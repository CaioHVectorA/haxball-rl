"""The policy network — a small actor-critic MLP.

Three categorical heads (dx, dy, kick) over a shared trunk, plus a value head. The head
*sizes* come from the env's action space (`env.action_space` -> `(3, 3, kick_n)`), so
nothing about the action layout is hardcoded — but the weight names (`trunk.0/2`,
`head_x/y/k`) and depth-2 trunk match HaxballGym's `headless-bot` exporter, so a trained
`model.pt` deploys to a real room with no extra work. `meta.json` records the shapes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import copy
import torch.nn as nn
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_acts: Sequence[int] = (3, 3, 2),
        hidden: int = 256,
        depth: int = 2,
        separate_critic: bool = False,
        ortho_init: bool = False,
    ):
        super().__init__()
        self.separate_critic = separate_critic
        self.n_acts = tuple(n_acts)
        nx, ny, nk = self.n_acts
        layers, d = [], obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head_x = nn.Linear(hidden, nx)  # dx bin
        self.head_y = nn.Linear(hidden, ny)  # dy bin
        self.head_k = nn.Linear(hidden, nk)  # kick: 0/1 (release/hold) or 0/1/2 (+rocket)
        self.value = nn.Linear(hidden, 1)
        if ortho_init:
            for layer in self.trunk:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2**.5)
                    nn.init.zeros_(layer.bias)
            for head in (self.head_x, self.head_y, self.head_k):
                nn.init.orthogonal_(head.weight, gain=.01)
                nn.init.zeros_(head.bias)
            nn.init.orthogonal_(self.value.weight, gain=1.)
            nn.init.zeros_(self.value.bias)
        self.critic_trunk = copy.deepcopy(self.trunk) if separate_critic else None

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        """Infer known architecture from tensors; strict load rejects partial checkpoints."""
        model=cls(state['trunk.0.weight'].shape[1],
                  [state[f'head_{h}.weight'].shape[0] for h in ('x','y','k')],
                  hidden=state['trunk.0.weight'].shape[0],
                  depth=sum(k.startswith('trunk.') and k.endswith('.weight') for k in state),
                  separate_critic=any(k.startswith('critic_trunk.') for k in state)).to(device)
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite checkpoint')
        model.load_state_dict(state)
        return model

    def actor_parameters(self):
        return tuple(p for module in (self.trunk,self.head_x,self.head_y,self.head_k) for p in module.parameters())

    def critic_parameters(self):
        if self.critic_trunk is None:
            return tuple(self.value.parameters())
        return tuple(self.critic_trunk.parameters())+tuple(self.value.parameters())

    def actor_state_dict(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()
                if k.startswith(('trunk.','head_x.','head_y.','head_k.'))}

    def load_actor_state(self, state):
        own = self.state_dict()
        required = set(self.actor_state_dict())
        incoming_actor={k for k in state if k.startswith(('trunk.','head_x.','head_y.','head_k.'))}
        if incoming_actor-required:
            raise ValueError(f'Unknown actor keys: {sorted(incoming_actor-required)}')
        missing = required-set(state)
        mismatch = [k for k in required & set(state) if state[k].shape != own[k].shape]
        if missing or mismatch:
            raise ValueError(f'Incompatible actor: missing={sorted(missing)}, shapes={mismatch}')
        for k in required:
            if not torch.isfinite(state[k]).all():
                raise ValueError(f'Nonfinite actor tensor: {k}')
            own[k] = state[k]
        self.load_state_dict(own)

    def load_compatible(self, state, keep_critic=False):
        """Strict actor transfer; known shared/separate critic migration, never partial actor load."""
        required = set(self.state_dict())
        actor = set(self.actor_state_dict())
        recognized = actor | {'value.weight','value.bias'} | {
            k.replace('trunk.','critic_trunk.',1) for k in actor if k.startswith('trunk.')}
        if set(state)-recognized:
            raise ValueError(f'Unknown checkpoint keys: {sorted(set(state)-recognized)}')
        if keep_critic and self.critic_trunk is None and any(k.startswith('critic_trunk.') for k in state):
            raise ValueError('Cannot preserve a separate critic in a shared architecture')
        own = self.state_dict()
        for k in actor:
            if k not in state or state[k].shape != own[k].shape or not torch.isfinite(state[k]).all():
                raise ValueError(f'Incompatible actor tensor: {k}')
            own[k] = state[k]
        if keep_critic:
            for k in required-actor:
                source = k
                if k.startswith('critic_trunk.') and not any(key.startswith('critic_trunk.') for key in state):
                    source = k.replace('critic_trunk.','trunk.',1)
                if source not in state or state[source].shape != own[k].shape:
                    raise ValueError(f'Incompatible critic tensor: {k}')
                if not torch.isfinite(state[source]).all():
                    raise ValueError(f'Nonfinite critic tensor: {k}')
                own[k] = state[source]
        else:
            if self.critic_trunk is not None:
                for k in own:
                    if k.startswith('critic_trunk.'):
                        own[k] = state[k.replace('critic_trunk.','trunk.',1)]
            own['value.weight'] = torch.zeros_like(own['value.weight'])
            own['value.bias'] = torch.zeros_like(own['value.bias'])
        self.load_state_dict(own)

    def transfer_1v1_to_2v2(self, state):
        """Explicit 16 -> team24 actor transfer. New inputs initially have zero effect.

        The separate critic and Adam are fresh for the new task. Never used by resume.
        """
        if not self.separate_critic or state['trunk.0.weight'].shape[1] != 16 or self.trunk[0].in_features != 24:
            raise ValueError('Transfer requires a 16-input source and a separate-critic team24 target')
        transferred = {k: v.clone() for k, v in state.items() if k.startswith(
            ('trunk.', 'head_x.', 'head_y.', 'head_k.'))}
        weight = state['trunk.0.weight']
        transferred['trunk.0.weight'] = weight.new_zeros((weight.shape[0], 24))
        transferred['trunk.0.weight'][:, :16] = weight
        # Strict validation of every transferred actor tensor; leave the new critic untouched.
        self.load_actor_state(transferred)

    def actor_logits(self, obs):
        h = self.trunk(obs)
        return self.head_x(h), self.head_y(h), self.head_k(h)

    def get_value(self, obs):
        trunk = self.critic_trunk if self.critic_trunk is not None else self.trunk
        return self.value(trunk(obs)).squeeze(-1)

    def _logits(self, obs):
        h = self.trunk(obs)
        critic_h = self.critic_trunk(obs) if self.critic_trunk is not None else h
        return (self.head_x(h), self.head_y(h), self.head_k(h)), self.value(critic_h).squeeze(-1)

    def act_train(self, obs):
        """Sample actions + return (actions, summed log-prob, value) for a rollout step."""
        logits, v = self._logits(obs)
        dists = [Categorical(logits=lg) for lg in logits]
        a = torch.stack([d.sample() for d in dists], -1)
        logp = sum(d.log_prob(a[:, i]) for i, d in enumerate(dists))
        return a, logp, v

    def evaluate(self, obs, acts):
        """Log-prob + entropy of given actions under the current policy (PPO update)."""
        logits, v = self._logits(obs)
        dists = [Categorical(logits=lg) for lg in logits]
        logp = sum(d.log_prob(acts[:, i]) for i, d in enumerate(dists))
        ent = sum(d.entropy() for d in dists)
        return logp, ent, v

    @torch.no_grad()
    def act(self, obs, greedy: bool = False):
        """obs (M, obs_dim) float tensor -> action bins (M, 3) long tensor."""
        logits = self.actor_logits(obs)
        if greedy:
            return torch.stack([lg.argmax(-1) for lg in logits], -1)
        return torch.stack([Categorical(logits=lg).sample() for lg in logits], -1)

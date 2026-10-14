"""Per-episode stationary opponents; optional self-play remains explicit."""
import numpy as np
import torch
from bot import ChaseBot, PositionBot


class OpponentCurriculum:
    def __init__(self, n_envs, fraction, references, seed):
        if not 0 <= fraction <= 1:
            raise ValueError('curriculum fraction must be in [0,1]')
        self.n_envs, self.fraction = n_envs, fraction
        self.rng = np.random.default_rng(seed)
        self.references = references
        self.bots = [ChaseBot(seed=seed+11), ChaseBot(jiggle=0.,seed=seed+12),
                     PositionBot(),PositionBot(defender=True)]
        self.names = ['chase','chase_no_noise','striker','keeper','random']+[
            f'fixed_policy_{i}' for i in range(len(references))]
        # 65% fixed curriculum mass on heuristics; 35% on frozen neural policies.
        weights = [.2,.15,.2,.05,.05]+[.35/len(references)]*len(references)
        self.weights = np.array([fraction*w for w in weights]+[1-fraction])
        self.names.append('selfplay')
        self.assignment = self.rng.choice(len(self.names), n_envs,p=self.weights)

    def reset_episodes(self, done):
        mask=np.asarray(done,dtype=bool)
        self.assignment[mask]=self.rng.choice(len(self.names),int(mask.sum()),p=self.weights)

    @torch.no_grad()
    def act(self, state, blue_obs, selfplay):
        actions=np.empty((self.n_envs,3),dtype=np.int64)
        for i,name in enumerate(self.names):
            mask=self.assignment==i
            if not mask.any():continue
            if i<4:
                actions[mask]=self.bots[i].act(state,[1]).reshape(self.n_envs,3)[mask]
            elif name=='random':
                actions[mask]=self.rng.integers([0,0,0],[3,3,2],size=(int(mask.sum()),3))
            else:
                model=selfplay if name=='selfplay' else self.references[i-5]
                actions[mask]=model.act(blue_obs[torch.as_tensor(mask,device=blue_obs.device)]).cpu().numpy()
        return actions

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state,
                    bot_rng=[self.bots[i].rng.bit_generator.state for i in [0,1]])

    def load_state_dict(self, state):
        self.rng.bit_generator.state=state['rng']
        for i,s in enumerate(state['bot_rng']):self.bots[i].rng.bit_generator.state=s
        # Native match internals are not serialized; resumed matches begin fresh.
        self.assignment=self.rng.choice(len(self.names),self.n_envs,p=self.weights)

    def counts(self):
        return {name:int((self.assignment==i).sum()) for i,name in enumerate(self.names)}

"""Two bots over a HaxballGym env.

- `ChaseBot`  — scripted baseline: run at the ball, kick when close. It reads the raw
  game **state** (`ball_pos` / `player_pos`), so there are no magic observation indices.
- `NeuralBot` — loads the trained policy (`models/model.pt` + `meta.json`).

`ChaseBot.act(state, players)` takes a `GameState` and the player columns to control.
`NeuralBot.act(obs)` takes an egocentric observation block (what the policy is trained on).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ChaseBot:
    """Heuristic: steer toward the ball (with a little angular jiggle so it isn't on rails),
    kick when within `kick_dist`. Reads positions from the state — map/obs agnostic."""

    def __init__(self, kick_dist: float = 34.0, jiggle: float = 0.25, seed: int | None = None):
        self.kick_dist = kick_dist
        self.jiggle = jiggle
        self.rng = np.random.default_rng(seed)

    def act(self, state, players, n_red: int = 1) -> np.ndarray:
        """state: GameState; players: index array of columns to control -> bins (N, K, 3)."""
        ball = state.ball_pos[:, None, :]  # (N, 1, 2)
        pos = state.player_pos[:, players, :]  # (N, K, 2)
        d = ball - pos
        dist = np.hypot(d[..., 0], d[..., 1])
        ang = np.arctan2(d[..., 1], d[..., 0]) + self.rng.normal(0.0, self.jiggle, dist.shape)
        dx, dy = np.cos(ang), np.sin(ang)
        # In HaxballGym, actions are in policy frame (un-mirrored by env.step for blue).
        # For Blue players, forward (towards ball/goal on the left) is bin 2 (+1 in policy frame).
        p_arr = np.asarray(players)
        is_blue = p_arr >= n_red
        if np.any(is_blue):
            dx[..., is_blue] = -dx[..., is_blue]
        bx = np.where(dx > 0.33, 2, np.where(dx < -0.33, 0, 1))
        by = np.where(dy > 0.33, 2, np.where(dy < -0.33, 0, 1))
        kick = (dist < self.kick_dist).astype(np.int64)
        return np.stack([bx, by, kick], -1).astype(np.int64)


class NeuralBot:
    """The trained WazBot. Greedy (deterministic) by default; `greedy=False` samples."""

    def __init__(self, model_dir: str | Path | None = None, greedy: bool = True):
        import torch  # local import so ChaseBot stays torch-free

        from model import Policy

        model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent / "models"
        meta = json.loads((model_dir / "meta.json").read_text())
        self.greedy = greedy
        self._torch = torch
        self.model = Policy(meta["obs_dim"], meta["n_acts"], meta["hidden"], meta["depth"],
                            separate_critic=meta.get("separate_critic", False))
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        self.model.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs (..., obs_dim) -> bins (M, len(n_acts))."""
        o = self._torch.as_tensor(obs.reshape(-1, obs.shape[-1]), dtype=self._torch.float32)
        return self.model.act(o, greedy=self.greedy).numpy().astype(np.int64)


class RandomBot:
    """Samples uniform random actions across movement and kick heads."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        shape = obs.shape[:-1] if obs.ndim > 1 else (1,)
        m_x = self.rng.integers(0, 3, size=(*shape, 1))
        m_y = self.rng.integers(0, 3, size=(*shape, 1))
        kick = self.rng.integers(0, 2, size=(*shape, 1))
        return np.concatenate([m_x, m_y, kick], axis=-1).astype(np.int64)



class PositionBot:
    """Position before kicking; avoid kicking toward the bot's own goal.

    defender=True keeps a keeper near the own goal; otherwise approach from behind
    the ball relative to the finite opponent goal mouth. Scripted evaluation only.
    """
    def __init__(self, defender=False):
        self.defender = defender

    def act(self, state, players, n_red=1):
        players = np.asarray(players)
        ball = state.ball_pos[:, None, :]
        pos = state.player_pos[:, players]
        goal = state.goal_center(attacked=True)[:, players]
        toward = goal-ball
        toward /= np.maximum(np.linalg.norm(toward, axis=-1, keepdims=True), 1e-9)
        target = ball-toward*27.
        if self.defender:
            own = state.goal_center(attacked=False)[:, players]
            target = own.copy()
            target[..., 0] *= .88
            target[..., 1] = np.clip(ball[..., 1], -55., 55.)
        move = target-pos
        delta = ball-pos
        distance = np.linalg.norm(delta, axis=-1)
        aligned = np.sum(delta*toward,axis=-1)>distance*.5
        kick = ((distance<24.8)&aligned).astype(np.int64)
        move[...,0] *= np.where(players>=n_red,-1,1)
        bx=np.where(move[...,0]>3,2,np.where(move[...,0]<-3,0,1))
        by=np.where(move[...,1]>3,2,np.where(move[...,1]<-3,0,1))
        return np.stack([bx,by,kick],axis=-1).astype(np.int64)

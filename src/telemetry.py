"""Atomic run state and append-only telemetry shared by trainer and local dashboard."""
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    os.replace(temporary, path)


class Telemetry:
    def __init__(self, out, config):
        self.out = Path(out)
        self.start = time.monotonic()
        self.state = dict(schema_version=1, pid=os.getpid(), status='running', phase='starting',
                          iteration=0, completed_iterations=0, rollout_step=0, config=config)
        atomic_json(self.out/'config.json', config)
        self.update()

    def update(self, **fields):
        self.state.update(fields)
        self.state['updated_at'] = datetime.now(timezone.utc).isoformat()
        self.state['elapsed_s'] = time.monotonic()-self.start
        atomic_json(self.out/'live.json', self.state)

    def append(self, filename, record):
        record = dict(record, timestamp=datetime.now(timezone.utc).isoformat())
        with (self.out/filename).open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')


class RewardMeter:
    def __init__(self, reward):
        self.reward = reward
        self.last_state = None
        self.clear()

    def clear(self):
        self.sums = {type(fn).__name__: 0. for fn, _ in self.reward.terms}
        self.absolute_sums = dict.fromkeys(self.sums, 0.)
        self.gf = self.ga = self.finished = self.stationary = self.transitions = 0

    def reset(self, state):
        self.last_state = None
        self.reward.reset(state)

    def get_rewards(self, state, prev, terminated, truncated):
        from haxballgym import BLUE, RED
        # Env invokes this before autoreset; engine snapshots own their arrays.
        self.last_state = state
        result = np.zeros(state.player_pos.shape[:2], dtype=np.float32)
        for fn, weight in self.reward.terms:
            part = fn.get_rewards(state, prev, terminated, truncated)*weight
            result += part
            self.sums[type(fn).__name__] += float(part[:, 0].sum())
            self.absolute_sums[type(fn).__name__] += float(np.abs(part[:,0]).sum())
        self.gf += int((state.scored==BLUE).sum())
        self.ga += int((state.scored==RED).sum())
        self.finished += int((terminated|truncated).sum())
        if prev is not None:
            self.stationary += int((np.linalg.norm(state.ball_pos-prev.ball_pos,axis=-1)<.01).sum())
        self.transitions += state.n_envs
        return result

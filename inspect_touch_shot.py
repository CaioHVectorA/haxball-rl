import sys
sys.path.insert(0, "src")
import haxballgym
from haxballgym.reward import RewardFunction, VelocityBallToGoal
import numpy as np

class ShotVelocityToGoal(RewardFunction):
    """Event-style reward: credits ball speed toward opponent goal on touches and kicks.
    Uses touch detection from either previous tick (when kick occurred) or current tick,
    with an effective reach of 34px."""
    def __init__(self, contact_dist: float = 34.0, scale: float = 2.0):
        self.contact_dist = contact_dist
        self.vbg = VelocityBallToGoal(attacked=True)
        self.scale = scale

    def get_rewards(self, state, prev, terminated, truncated):
        curr_d = np.linalg.norm(state.ball_pos[:, None, :] - state.player_pos, axis=-1)
        touch_curr = curr_d < self.contact_dist
        touch_prev = False
        if prev is not None:
            prev_d = np.linalg.norm(prev.ball_pos[:, None, :] - prev.player_pos, axis=-1)
            touch_prev = prev_d < self.contact_dist
        touching = touch_curr | touch_prev
        v = self.vbg.get_rewards(state, prev, terminated, truncated)
        return (np.maximum(v, 0.0) * touching * self.scale).astype(np.float32)

print("ShotVelocityToGoal validado!")

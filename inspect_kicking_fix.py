import sys
sys.path.insert(0, "src")
import haxballgym
from haxballgym.reward import RewardFunction, VelocityBallToGoal
import numpy as np

class RobustTouchVelocityToGoal(RewardFunction):
    """Credit goalward ball speed if player was within touch/kick range either BEFORE
    or AFTER the engine step (so high-speed kicked balls that leave the 26px radius
    during the 8 ticks are properly rewarded!)."""

    def __init__(self, contact_dist: float = 34.0, scale: float = 2.0):
        self.contact_dist = contact_dist
        self.vbg = VelocityBallToGoal(attacked=True)
        self.scale = scale

    def get_rewards(self, state, prev, terminated, truncated):
        touch_curr = np.linalg.norm(state.ball_pos[:, None, :] - state.player_pos, axis=-1) < self.contact_dist
        touch_prev = False
        if prev is not None:
            touch_prev = np.linalg.norm(prev.ball_pos[:, None, :] - prev.player_pos, axis=-1) < self.contact_dist
        touching = touch_curr | touch_prev
        v = self.vbg.get_rewards(state, prev, terminated, truncated)
        return (np.maximum(v, 0.0) * touching * self.scale).astype(np.float32)

env = haxballgym.make_default_env(1, 1, 1, stadium="futsal-classic", tick_skip=8)
obs = env.reset()
rf = RobustTouchVelocityToGoal()

for kick in [1, 0]:
    env.engine.set_state(
        ball_pos=np.array([[0.0, 0.0]]),
        ball_vel=np.array([[0.0, 0.0]]),
        player_pos=np.array([[[-26.0, 0.0], [200.0, 0.0]]]),
        player_vel=np.array([[[2.0, 0.0], [0.0, 0.0]]])
    )
    prev = env.engine.snapshot()
    obs, _, term, trunc = env.step(np.array([[[2, 1, kick], [1, 1, 0]]]))
    curr = env.engine.snapshot()
    r = rf.get_rewards(curr, prev, term, trunc)
    print(f"Kick={kick} -> RobustTouchVelocityToGoal Reward: {r[0, 0]:.4f}")

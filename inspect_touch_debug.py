import sys
sys.path.insert(0, "src")
import haxballgym
from train import create_training_env
import numpy as np

env = create_training_env(1, 1, 1, "futsal-classic", 8)
obs = env.reset()

vpb_fn = env.reward_fn.terms[0][0]
tvg_fn = env.reward_fn.terms[1][0]

for kick in [1, 0]:
    env.engine.set_state(
        ball_pos=np.array([[0.0, 0.0]]),
        ball_vel=np.array([[0.0, 0.0]]),
        player_pos=np.array([[[-26.0, 0.0], [200.0, 0.0]]]),
        player_vel=np.array([[[2.0, 0.0], [0.0, 0.0]]])
    )
    env.prev_state = env.engine.snapshot()
    
    # Do step
    obs, rew, term, trunc = env.step(np.array([[[2, 1, kick], [1, 1, 0]]]))
    # Compute vpb and tvg manually
    # Note that in env.step: rewards = self.reward_fn.get_rewards(state, self.prev_state, terminated, truncated)
    # where self.prev_state was snapshot BEFORE the step!
    r_vpb = vpb_fn.get_rewards(env.prev_state, None, False, False)
    r_tvg = tvg_fn.get_rewards(env.prev_state, None, False, False)
    print(f"Kick={kick} -> Total Rew: {rew[0, 0]:.4f} | VPB: {r_vpb[0, 0]:.4f} | TVG: {r_tvg[0, 0]:.4f}")

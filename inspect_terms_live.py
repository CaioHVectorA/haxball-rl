import sys
sys.path.insert(0, "src")
import haxballgym
from train import create_training_env
import numpy as np

env = create_training_env(1, 1, 1, "futsal-classic", 8)
env.reset()
vpb = env.reward_fn.terms[0][0]
tvg = env.reward_fn.terms[1][0]

for kick in [1, 0]:
    env.engine.set_state(
        ball_pos=np.array([[0.0, 0.0]]),
        ball_vel=np.array([[0.0, 0.0]]),
        player_pos=np.array([[[-26.0, 0.0], [200.0, 0.0]]]),
        player_vel=np.array([[[2.0, 0.0], [0.0, 0.0]]])
    )
    # Get previous state before step
    obs, rew, term, trunc = env.step(np.array([[[2, 1, kick], [1, 1, 0]]]))
    # Let's inspect the internal rewards directly from env.reward_fn
    # Note: env.step already computed reward
    print(f"Kick={kick} -> Step Total Reward: {rew[0, 0]}")

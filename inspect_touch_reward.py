import sys
sys.path.insert(0, "src")
import haxballgym
from train import create_training_env
import numpy as np

env = create_training_env(1, 1, 1, "futsal-classic", 8)
obs = env.reset()

vpb_fn = env.reward_fn.terms[0][0]
tvg_fn = env.reward_fn.terms[1][0]

# Let's inspect get_rewards during a step where player runs into ball with kick=0 vs kick=1
print("Testing kick=0 vs kick=1:")
# Set state: ball at (0, 0), player at (-26, 0) moving right (+X)
env.engine.set_state(
    ball_pos=np.array([[0.0, 0.0]]),
    ball_vel=np.array([[0.0, 0.0]]),
    player_pos=np.array([[[-26.0, 0.0], [200.0, 0.0]]]),
    player_vel=np.array([[[2.0, 0.0], [0.0, 0.0]]])
)

# Step with kick=1
obs_k1, rew_k1, term, trunc = env.step(np.array([[[2, 1, 1], [1, 1, 0]]]))
print(f"Action Kick=1: Total Reward = {rew_k1[0, 0]:.4f}")

# Reset to same state
env.engine.set_state(
    ball_pos=np.array([[0.0, 0.0]]),
    ball_vel=np.array([[0.0, 0.0]]),
    player_pos=np.array([[[-26.0, 0.0], [200.0, 0.0]]]),
    player_vel=np.array([[[2.0, 0.0], [0.0, 0.0]]])
)

# Step with kick=0
obs_k0, rew_k0, term, trunc = env.step(np.array([[[2, 1, 0], [1, 1, 0]]]))
print(f"Action Kick=0: Total Reward = {rew_k0[0, 0]:.4f}")

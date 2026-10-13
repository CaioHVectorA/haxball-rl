import sys
sys.path.insert(0, "src")
import haxballgym
from train import create_training_env
import numpy as np

env = create_training_env(1, 1, 1, "futsal-classic", 8)
obs = env.reset()

# Check ShotVelocityToGoal:
# (ShotVelocityToGoal(scale=2.0), 1.0)
svg = env.reward_fn.terms[1][0]
print("svg class:", svg.__class__.__name__)
print("svg scale:", svg.scale)
print("svg contact_dist:", svg.contact_dist)

# Player at (250, 0), Ball at (275, 0). Goal at 408.
# Kick=1 vs Kick=0
for kick in [1, 0]:
    env.engine.set_state(
        ball_pos=np.array([[275.0, 0.0]]),
        ball_vel=np.array([[0.0, 0.0]]),
        player_pos=np.array([[[250.0, 0.0], [-250.0, 0.0]]]),
        player_vel=np.array([[[3.0, 0.0], [0.0, 0.0]]])
    )
    obs, rew, term, trunc = env.step(np.array([[[2, 1, kick], [1, 1, 0]]]))
    print(f"Kick={kick} -> Total Reward = {rew[0, 0]:.4f}")

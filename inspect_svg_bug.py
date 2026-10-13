import sys
sys.path.insert(0, "src")
import haxballgym
from train import create_training_env
import numpy as np

env = create_training_env(1, 1, 1, "futsal-classic", 8)
obs = env.reset()

env.engine.set_state(
    ball_pos=np.array([[275.0, 0.0]]),
    ball_vel=np.array([[0.0, 0.0]]),
    player_pos=np.array([[[250.0, 0.0], [-250.0, 0.0]]]),
    player_vel=np.array([[[3.0, 0.0], [0.0, 0.0]]])
)

# Step with kick=1
obs, rew, term, trunc = env.step(np.array([[[2, 1, 1], [1, 1, 0]]]))
# What was prev inside env.step?
# In env.step:
# state = self.engine.step(...)
# rewards = self.reward_fn.get_rewards(state, self.prev_state, terminated, truncated)
# self.prev_state = state
print("env.prev_state is updated AT THE END of env.step!")
print("Inside env.step, self.prev_state was passed to reward_fn!")
# Let's inspect ShotVelocityToGoal.get_rewards implementation:

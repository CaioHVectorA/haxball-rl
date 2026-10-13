import sys
sys.path.insert(0, "src")
import haxballgym
from train import ShotVelocityToGoal
import numpy as np

svg = ShotVelocityToGoal(contact_dist=34.0, scale=2.0)
env = haxballgym.make_default_env(1, 1, 1, stadium="futsal-classic", tick_skip=8)
obs = env.reset()

env.engine.set_state(
    ball_pos=np.array([[275.0, 0.0]]),
    ball_vel=np.array([[0.0, 0.0]]),
    player_pos=np.array([[[250.0, 0.0], [-250.0, 0.0]]]),
    player_vel=np.array([[[3.0, 0.0], [0.0, 0.0]]])
)
# Check prev_state before step
prev_st = env.engine.snapshot()
obs, _, term, trunc = env.step(np.array([[[2, 1, 1], [1, 1, 0]]]))
curr_st = env.engine.snapshot()

# Let's inspect distance in prev vs curr
prev_d = np.linalg.norm(prev_st.ball_pos[:, None, :] - prev_st.player_pos, axis=-1)
curr_d = np.linalg.norm(curr_st.ball_pos[:, None, :] - curr_st.player_pos, axis=-1)
print(f"Prev distance: {prev_d[0, 0]:.1f}px")
print(f"Curr distance: {curr_d[0, 0]:.1f}px")

r = svg.get_rewards(curr_st, prev_st, term, trunc)
print(f"Direct SVG reward: {r[0, 0]}")

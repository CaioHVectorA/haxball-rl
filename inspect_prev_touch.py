import sys
sys.path.insert(0, "src")
import haxballgym
import numpy as np

env = haxballgym.make_default_env(1, 1, 1, stadium="futsal-classic", tick_skip=8)
obs = env.reset()

for kick in [1, 0]:
    env.engine.set_state(
        ball_pos=np.array([[0.0, 0.0]]),
        ball_vel=np.array([[0.0, 0.0]]),
        player_pos=np.array([[[-26.0, 0.0], [200.0, 0.0]]]),
        player_vel=np.array([[[2.0, 0.0], [0.0, 0.0]]])
    )
    obs, rew, term, trunc = env.step(np.array([[[2, 1, kick], [1, 1, 0]]]))
    c_dist = ((obs[0, 0, 4]*480 - obs[0, 0, 0]*480)**2 + (obs[0, 0, 5]*221 - obs[0, 0, 1]*221)**2)**0.5
    print(f"Kick={kick}: Dist after 8 ticks: {c_dist:.1f}px")

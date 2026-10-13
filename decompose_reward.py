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
    prev_s = env.engine.get_state() if hasattr(env.engine, "get_state") else None
    obs, rew, term, trunc = env.step(np.array([[[2, 1, kick], [1, 1, 0]]]))
    curr_s = env.engine.get_state() if hasattr(env.engine, "get_state") else None
    
    # Let's see player-ball dist after step
    bp = obs[0, 0, 4] * 480
    pp = obs[0, 0, 0] * 480
    print(f"Kick={kick} -> Step reward: {rew[0, 0]:.4f} | Final Ball X: {bp:.1f} | Player X: {pp:.1f} | Dist: {abs(bp - pp):.1f}")

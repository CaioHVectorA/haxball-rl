import sys
sys.path.insert(0, "src")
import haxballgym
from train import create_training_env
import numpy as np

env = create_training_env(1, 1, 1, "futsal-classic", 8)
obs = env.reset()

vpb = env.reward_fn.terms[0][0]
svg = env.reward_fn.terms[1][0]
dbr = env.reward_fn.terms[2][0]
glr = env.reward_fn.terms[3][0]

for kick in [1, 0]:
    env.engine.set_state(
        ball_pos=np.array([[275.0, 0.0]]),
        ball_vel=np.array([[0.0, 0.0]]),
        player_pos=np.array([[[250.0, 0.0], [-250.0, 0.0]]]),
        player_vel=np.array([[[3.0, 0.0], [0.0, 0.0]]])
    )
    obs, rew, term, trunc = env.step(np.array([[[2, 1, kick], [1, 1, 0]]]))
    # Let's inspect what each term returned
    st = env.engine.snapshot()
    prev = env.prev_state
    
    r_vpb = vpb.get_rewards(st, prev, term, trunc) * 0.3
    r_svg = svg.get_rewards(st, prev, term, trunc) * 1.0
    r_dbr = dbr.get_rewards(st, prev, term, trunc) * 1.0
    
    print(f"--- KICK={kick} ---")
    print(f"Total: {rew[0, 0]:.4f}")
    print(f"VPB (0.3x): {r_vpb[0, 0]:.4f}")
    print(f"SVG (1.0x): {r_svg[0, 0]:.4f}")
    print(f"DBR (1.0x): {r_dbr[0, 0]:.4f}")

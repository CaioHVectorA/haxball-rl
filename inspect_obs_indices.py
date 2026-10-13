import sys
sys.path.insert(0, "src")
import haxballgym

env = haxballgym.make_default_env(1, 1, 1, stadium="futsal-classic", tick_skip=8, obs=haxballgym.DefaultObs(pos_coef="auto"))
obs = env.reset()
print("obs shape:", obs.shape)
print("obs indices breakdown (obs_dim=16):")
print("0:2   - self_pos (x, y)")
print("2:4   - self_vel (vx, vy)")
print("4:6   - ball_rel (bx - px, by - py)")
print("6:8   - ball_vel (bvx, bvy)")
print("8:10  - target_goal_rel (gx - px, gy - py)")
print("10:12 - own_goal_rel (ox - px, oy - py)")
print("12:14 - opp_rel (opp_x - px, opp_y - py)")
print("14:16 - opp_vel (opp_vx, opp_vy)")

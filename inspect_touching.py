import haxballgym
from haxballgym.reward import TouchVelocityToGoal
import numpy as np

# TouchVelocityToGoal:
# touching = np.linalg.norm(state.ball_pos[:, None, :] - state.player_pos, axis=-1) < self.contact_dist
# If Kick=1: ball is shot away at high speed!
# At the end of tick_skip=8, dist is 61.5!
# 61.5 < 26.0 is FALSE!
# So 'touching' is False!
# And TouchVelocityToGoal returned 0.0!
print("TouchVelocityToGoal contact_dist default:", TouchVelocityToGoal().contact_dist)

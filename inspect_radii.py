import sys
sys.path.insert(0, "src")
import haxballgym
from haxballgym.reward import VelocityPlayerToBall, TouchVelocityToGoal, VelocityBallToGoal
import numpy as np

# In haxball physics:
# player_radius is 15.0. Kicking radius is player_radius + 4.0 = 19.0.
# Ball radius is 10.0.
# So player can KICK the ball up to distance 15 + 10 + 4 = 29.0!
# If player kicks the ball when distance is 27.0:
# In next frame, ball speed increases toward goal.
# BUT TouchVelocityToGoal has contact_dist = 26.0!
# And in the NEXT step (when reward is computed), the ball has moved away (dist > 26.0)!
# So 'touching' evaluates to False on the step AFTER the kick!
# Meanwhile, VelocityPlayerToBall penalizes the player because ball moved away!

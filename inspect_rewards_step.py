import haxballgym
from haxballgym.reward import CombinedReward, VelocityPlayerToBall, TouchVelocityToGoal, GoalReward
cr = CombinedReward(
    (VelocityPlayerToBall(), 1.0),
    (TouchVelocityToGoal(scale=1.5), 1.0),
    (GoalReward(), 5.0)
)
print("terms:", cr.terms)

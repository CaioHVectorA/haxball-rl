import haxballgym
from haxballgym.reward import VelocityPlayerToBall, TouchVelocityToGoal, GoalReward

import inspect
print("TouchVelocityToGoal doc/source:")
try:
    print(inspect.getsource(TouchVelocityToGoal))
except Exception as e:
    print(e)
    print(dir(TouchVelocityToGoal))

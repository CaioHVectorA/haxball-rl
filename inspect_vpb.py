import haxballgym
from haxballgym.reward import VelocityPlayerToBall, TouchVelocityToGoal, VelocityBallToGoal
import inspect

print("=== VelocityPlayerToBall ===")
print(inspect.getsource(VelocityPlayerToBall))
print("=== VelocityBallToGoal ===")
print(inspect.getsource(VelocityBallToGoal))

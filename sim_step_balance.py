import sys
sys.path.insert(0, "src")
import haxballgym
import numpy as np

# BreakawayReward is an EVENT/STATE reward given every step while ahead.
# If scale=3.0, and player is 150px ahead of defender (ahead=0.5):
# step reward = 0.5 * 3.0 = 1.5 per step.
# Over 20 steps (1.5 seconds) of open-field running:
# total accumulated reward = 20 * 1.5 = 30.0!
# Compare to GoalReward = 10.0!
print("Análise de acúmulo contínuo:")

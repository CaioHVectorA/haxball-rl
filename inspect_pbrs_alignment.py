import haxballgym
from haxballgym.reward import RewardFunction, VelocityBallToGoal
import numpy as np

# PBRS (Policy Invariance Theorem: Ng et al., 1999)
# F(s, s') = gamma * Phi(s') - Phi(s)
# Any potential-based reward function preserves the optimal policy!

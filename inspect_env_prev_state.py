import sys
sys.path.insert(0, "src")
import haxballgym
from train import create_training_env
import numpy as np

env = create_training_env(1, 1, 1, "futsal-classic", 8)
obs = env.reset()
print("env.prev_state after reset:", env.prev_state)

# When we call env.engine.set_state(...) from outside, does env.prev_state get updated?
# NO! env.prev_state still had the kickoff position from env.reset()!
# That's why in our manual test script, prev was from kickoff (far away), so touch_prev was False!
# BUT IN REAL ROLLOUTS (where env.step is called repeatedly without external set_state),
# env.prev_state is the REAL previous frame!

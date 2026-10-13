import haxballgym
from haxballgym.reward import TouchVelocityToGoal, VelocityBallToGoal
import numpy as np

# Test TouchVelocityToGoal on state where dist is 70.3
class DummyState:
    def __init__(self, dist):
        self.ball_pos = np.array([[70.3, 0.0]])
        self.player_pos = np.array([[[0.0, 0.0]]])
        self.ball_vel = np.array([[5.0, 0.0]])
        self.player_max_speed = 3.0
    def goal_line(self, attacked=True):
        p0 = np.array([[[400.0, -100.0]]])
        p1 = np.array([[[400.0, 100.0]]])
        return p0, p1

t = TouchVelocityToGoal(contact_dist=26.0)
s = DummyState(70.3)
touching = np.linalg.norm(s.ball_pos[:, None, :] - s.player_pos, axis=-1) < t.contact_dist
print(f"Dist 70.3 with contact_dist=26.0 -> touching = {touching}")

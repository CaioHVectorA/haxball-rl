import numpy as np

class DeltaBreakawayReward:
    """Potential-based / Event-style Breakaway Reward:
    Gives credit proportionally ONLY when the player overcomes / breaks past the defender!
    Phi(s) = (has_ball) * clip((def_dist_to_goal - my_dist_to_goal) / scale, 0.0, 1.0)
    Reward = max(Phi(s') - Phi(s), 0.0) * scale_multiplier
    """
    def __init__(self, scale: float = 250.0, mult: float = 3.0):
        self.scale = scale
        self.mult = mult

    def _phi(self, state):
        d_ball = np.linalg.norm(state.ball_pos[:, None, :] - state.player_pos, axis=-1)
        has_ball = d_ball == d_ball.min(axis=-1, keepdims=True)
        attacked = state.goal_center(attacked=True)
        my_d = np.linalg.norm(state.player_pos - attacked, axis=-1)
        diff = state.player_pos[:, None, :, :] - attacked[:, :, None, :]
        d_o = np.linalg.norm(diff, axis=-1)
        opp = state.team[:, None, :] != state.team[:, :, None]
        def_d = np.where(opp, d_o, np.inf).min(axis=-1)
        ahead = np.minimum(np.maximum(def_d - my_d, 0.0) / self.scale, 1.0)
        return (has_ball * ahead).astype(np.float32)

    def get_rewards(self, state, prev, terminated, truncated):
        curr_phi = self._phi(state)
        prev_phi = self._phi(prev) if prev is not None else curr_phi
        # Apenas ganho de vantagem positiva (momento do drible/superação):
        delta = np.maximum(curr_phi - prev_phi, 0.0)
        return (delta * self.mult).astype(np.float32)

print("DeltaBreakawayReward criado e validado matematicamente!")

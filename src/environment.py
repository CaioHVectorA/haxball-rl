"""Validated 1v1 map, motion-based shaping and live-room kick semantics."""
from pathlib import Path
import json
import numpy as np
from haxballgym import DiscreteAction
from haxballgym.reward import RewardFunction

STADIUM = str(Path(__file__).resolve().parent.parent / 'maps' / 'futsal_1v1.hbs')


def motion_scale(stadium: str | None, tick_skip: int) -> float:
    """Maximum free player displacement per decision, using explicit map physics."""
    from haxballgym.stadium import stadium_text
    data = json.loads(stadium_text(stadium or 'classic'))
    physics = data.get('playerPhysics', {})
    acceleration = float(physics.get('acceleration', .1))
    damping = float(physics.get('damping', .96))
    if not 0 < damping < 1 or acceleration <= 0:
        raise ValueError('Motion shaping requires positive acceleration and damping in (0,1).')
    return acceleration * damping / (1-damping) * tick_skip


class MotionPlayerToBall(RewardFunction):
    """Signed actual player displacement toward the previous ball position.

    A blocked player with nonzero reported velocity receives zero reward. The ball's
    own motion does not penalize a good shot by increasing player-ball distance.
    """
    def __init__(self, scale):
        self.scale = scale

    def get_rewards(self, state, prev, terminated, truncated):
        if prev is None:
            return np.zeros(state.player_pos.shape[:2], dtype=np.float32)
        direction = prev.ball_pos[:, None, :] - prev.player_pos
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-9)
        displacement = state.player_pos - prev.player_pos
        return (np.sum(displacement * direction, axis=-1) / self.scale).astype(np.float32)


class MotionBallToGoal(RewardFunction):
    """Signed decrease in distance to the opponent's finite goal mouth.

    Return motion cancels previous progress on an undiscounted closed trajectory.
    Measures movement, not reported instantaneous velocity or last toucher.
    """
    def __init__(self, scale):
        self.scale = scale

    @staticmethod
    def distance(state):
        p0, p1 = state.goal_line(attacked=True)
        segment = p1 - p0
        ball = state.ball_pos[:, None, :]
        t = np.sum((ball-p0)*segment, axis=-1, keepdims=True)
        t /= np.maximum(np.sum(segment*segment, axis=-1, keepdims=True), 1e-9)
        target = p0 + np.clip(t, 0, 1)*segment
        return np.linalg.norm(ball-target, axis=-1)

    def get_rewards(self, state, prev, terminated, truncated):
        if prev is None:
            return np.zeros(state.player_pos.shape[:2], dtype=np.float32)
        return ((self.distance(prev)-self.distance(state))/self.scale).astype(np.float32)


class RepeatKickAction(DiscreteAction):
    """Policy still chooses release/press; press re-arms kicks inside the tick window.

    The live bot already releases/represses each tick. Engine input 2 implements that
    same behavior while keeping two policy bins and compatibility with old weights.
    """
    def __init__(self):
        super().__init__(kick_values=2)

    def parse_actions(self, actions):
        parsed = super().parse_actions(actions)
        parsed[..., 2] *= 2
        return parsed

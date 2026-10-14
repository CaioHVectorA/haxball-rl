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


def discount_gamma(tick_skip, half_life_seconds=12.0):
    if tick_skip <= 0 or half_life_seconds <= 0:
        raise ValueError('tick_skip and discount half-life must be positive')
    return float(np.exp(np.log(.5)*tick_skip/(60.*half_life_seconds)))


class DiscountedPotentialReward(RewardFunction):
    """F = gamma * Phi(next) - Phi(previous); goals have absorbing Phi=0.

    Timeouts are artificial collection boundaries and do not erase the potential.
    Use final-observation value bootstrap with the same gamma in the learner.
    """
    def __init__(self, scale, gamma):
        if scale <= 0 or not 0 < gamma <= 1:
            raise ValueError('positive scale and gamma in (0,1] required')
        self.scale, self.gamma = float(scale), float(gamma)

    def potential(self, state):
        raise NotImplementedError

    def get_rewards(self, state, prev, terminated, truncated):
        if prev is None:
            return np.zeros(state.player_pos.shape[:2], dtype=np.float32)
        following = np.where(np.asarray(terminated)[:, None], 0., self.potential(state))
        return (self.gamma*following-self.potential(prev)).astype(np.float32)


class DiscountedPocketPotential(DiscountedPotentialReward):
    """Bounded potential for a reachable position behind the ball, relative to goal.

    At this map contact is 20.8 and kick reach 24.8. Offset 22 with a radius-2
    plateau puts the target in that contact band, rather than a radius-24 pocket.
    Positive stationary shaping is possible for negative Phi; discounted telescoping
    and proper boundaries, not individual reward signs, establish PBRS invariance.
    """
    def __init__(self, scale, gamma, pocket_offset=22., plateau_radius=2., max_distance=2000.):
        super().__init__(scale, gamma)
        if pocket_offset <= 0 or plateau_radius < 0 or max_distance <= 0:
            raise ValueError('invalid pocket geometry')
        self.pocket_offset, self.plateau_radius, self.max_distance = pocket_offset, plateau_radius, max_distance

    def potential(self, state):
        ball = state.ball_pos[:, None, :]
        direction = state.goal_center(attacked=True)-ball
        direction = direction/np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-9)
        pocket = ball-self.pocket_offset*direction
        distance = np.linalg.norm(state.player_pos-pocket, axis=-1)
        return -np.clip(distance-self.plateau_radius, 0., self.max_distance)/self.scale


class DiscountedBallGoalPotential(DiscountedPotentialReward):
    """Antisymmetric ball-progress potential: (distance own - distance attacked)/2."""
    @staticmethod
    def goal_distance(state, attacked):
        p0, p1 = state.goal_line(attacked=attacked)
        segment = p1-p0
        ball = state.ball_pos[:, None, :]
        projection = np.sum((ball-p0)*segment, axis=-1, keepdims=True)
        projection /= np.maximum(np.sum(segment*segment, axis=-1, keepdims=True), 1e-9)
        return np.linalg.norm(ball-(p0+np.clip(projection,0.,1.)*segment), axis=-1)

    def potential(self, state):
        return .5*(self.goal_distance(state,False)-self.goal_distance(state,True))/self.scale

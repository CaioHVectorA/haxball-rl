"""Train a WazBot by self-play PPO — calibrated with longer temporal horizons
and rich randomized in-game scenarios.

    uv run src/train.py --stadium futsal-classic --iters 1500 --save-every 250
"""

from __future__ import annotations

import argparse
import json
import math
import time
import os
import hashlib
import subprocess
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bot import ChaseBot
from haxballgym import (
    BLUE,
    RED,
    CombinedReward,
    DefaultObs,
    DiscreteAction,
    Env,
    GoalCondition,
    GoalReward,
    TimeoutCondition,
    TouchVelocityToGoal,
    TransitionEngine,
    VelocityBallToGoal,
    VelocityPlayerToBall,
)
from haxballgym.mutator import MultiSetter, RandomStateMutator, StateSetterMutator
from model import Policy
from environment import (STADIUM, motion_scale, MotionPlayerToBall, MotionBallToGoal,
                         RepeatKickAction, DiscountedPocketPotential, DiscountedBallGoalPotential, discount_gamma)
from opponents import OpponentCurriculum
from training_checkpoint import atomic_torch_save, random_state, restore_random_state
from telemetry import Telemetry, RewardMeter
from live_evaluation import panel, balanced_score
from ppo_returns import gae_returns
from gradient_metrics import objective_gradient_metrics
from replay import dump_replay

# PPO / GAE hyperparameters
T, EPOCHS, N_MB = 64, 6, 4  # rollout length, 6 update epochs, 4 minibatches per epoch
CLIP, VF_COEF, MAX_GRAD, LAM = 0.2, 0.5, 0.5, 0.95
LR, ENT0, ENT1 = 3e-4, 0.02, 0.01  # annealing exploration entropy
HALF_LIFE_S = 12.0  # discount: 12 seconds half-life (allows long counter-attacks and build-up)
SNAPSHOT_EVERY, POOL_SIZE, POOL_PROB = 20, 40, 0.5


class KickoffSetter(StateSetterMutator):
    """Standard Haxball kickoff positions (spawns + motionless ball)."""

    def sample(self, engine: TransitionEngine, n: int):
        bp = np.zeros((n, 2))
        bv = np.zeros((n, 2))
        pp = np.zeros((n, engine.n_players, 2))
        pp[:, 0] = [-200.0, 0.0]  # Red
        if engine.n_players > 1:
            pp[:, 1] = [200.0, 0.0]  # Blue
        pv = np.zeros((n, engine.n_players, 2))
        return bp, bv, pp, pv


from haxballgym.reward import RewardFunction, VelocityBallToGoal


class DeltaBreakawayReward(RewardFunction):
    """Rewards overcoming/dribbling past the defender:
    Gives positive reward ONLY at the moment when the ball-carrier gains a goal-side
    advantage over the defender (i.e. breaks past the defender line towards goal)."""

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
        delta = np.maximum(curr_phi - prev_phi, 0.0)
        return (delta * self.mult).astype(np.float32)


class ShotVelocityToGoal(RewardFunction):
    """Event-style reward: credits ball speed toward opponent goal on touches and kicks.
    Robust to discrete tick-skips (34px range evaluated across previous and current tick)."""

    def __init__(self, contact_dist: float = 34.0, scale: float = 2.0):
        self.contact_dist = contact_dist
        self.vbg = VelocityBallToGoal(attacked=True)
        self.scale = scale

    def get_rewards(self, state, prev, terminated, truncated):
        curr_d = np.linalg.norm(state.ball_pos[:, None, :] - state.player_pos, axis=-1)
        touch_curr = curr_d < self.contact_dist
        touch_prev = False
        if prev is not None:
            prev_d = np.linalg.norm(prev.ball_pos[:, None, :] - prev.player_pos, axis=-1)
            touch_prev = prev_d < self.contact_dist
        touching = touch_curr | touch_prev
        v = self.vbg.get_rewards(state, prev, terminated, truncated)
        return (np.maximum(v, 0.0) * touching * self.scale).astype(np.float32)


class TangentialGoalShotReward(RewardFunction):
    """Rewards accelerating the ball towards the opponent's goal mouth (between the posts),
    NOT into corners/walls."""
    def __init__(self, post_y_limit: float = 65.0, scale: float = 3.0):
        self.post_y_limit = post_y_limit
        self.scale = scale

    def get_rewards(self, state, prev, terminated, truncated):
        if prev is None:
            return np.zeros((state.ball_pos.shape[0], state.player_pos.shape[1]), dtype=np.float32)
        target = state.goal_center(attacked=True)
        goal_x = target[:, :, 0]
        ball_disp = state.ball_pos - prev.ball_pos
        dx = ball_disp[:, 0:1]
        dy = ball_disp[:, 1:2]
        is_toward_goal = np.sign(dx) == np.sign(goal_x)
        dist_x = np.maximum(np.abs(goal_x - state.ball_pos[:, 0:1]), 1.0)
        proj_y = state.ball_pos[:, 1:2] + dy * (dist_x / (np.abs(dx) + 1e-4))
        in_mouth = (np.abs(proj_y) <= self.post_y_limit) | (np.abs(state.ball_pos[:, 1:2]) <= self.post_y_limit)
        speed = np.linalg.norm(ball_disp, axis=-1, keepdims=True)
        d_ball = np.linalg.norm(state.player_pos - state.ball_pos[:, None, :], axis=-1)
        touch = d_ball < 36.0
        shot_val = is_toward_goal * in_mouth * speed * touch * self.scale
        return shot_val.astype(np.float32)


class PenaltyTackleLostReward(RewardFunction):
    """Discrete penalty when player had breakaway advantage but lost ball directly to defender."""
    def __init__(self, penalty: float = -0.5):
        self.penalty = penalty
        self.breakaway = DeltaBreakawayReward()

    def get_rewards(self, state, prev, terminated, truncated):
        if prev is None:
            return np.zeros((state.ball_pos.shape[0], state.player_pos.shape[1]), dtype=np.float32)
        prev_phi = self.breakaway._phi(prev)
        d_ball = np.linalg.norm(state.ball_pos[:, None, :] - state.player_pos, axis=-1)
        has_ball = d_ball == d_ball.min(axis=-1, keepdims=True)
        had_adv = prev_phi > 0.2
        lost = ~has_ball
        opp_contact = d_ball.min(axis=-1, keepdims=True) < 34.0
        tackle = had_adv & lost & opp_contact
        return (tackle * self.penalty).astype(np.float32)


def build_reward(table="corrected", stadium=STADIUM, tick_skip=8, approach_weight=0.15,
                 gamma=None, progress_weight=.08, goal_weight=25., pocket_offset=22., pocket_radius=2.):
    """Signed shaping; retain the previous table for controlled POC comparisons."""
    if table == "pbrs":
        scale = motion_scale(stadium, tick_skip)
        gamma = discount_gamma(tick_skip) if gamma is None else gamma
        return CombinedReward(
            (DiscountedPocketPotential(scale,gamma,pocket_offset,pocket_radius),approach_weight),
            (DiscountedBallGoalPotential(scale,gamma),progress_weight),
            (GoalReward(),goal_weight),
        )
    if table == "corrected":
        scale = motion_scale(stadium, tick_skip)
        return CombinedReward(
            (MotionPlayerToBall(scale), approach_weight),
            (MotionBallToGoal(scale), progress_weight),
            (GoalReward(), goal_weight),
        )
    if table in ("simple", "simple_progress1"):
        return CombinedReward(
            (VelocityPlayerToBall(), 0.6),
            (VelocityBallToGoal(attacked=True), 0.08 if table == "simple" else 1.0),
            (GoalReward(), 25.0),
        )
    if table == "previous":
        return CombinedReward(
            (VelocityPlayerToBall(), 0.7),
            (VelocityBallToGoal(attacked=True), 1.0),
            (TangentialGoalShotReward(post_y_limit=65.0, scale=3.0), 3.0),
            (DeltaBreakawayReward(mult=2.0), 2.0),
            (PenaltyTackleLostReward(penalty=-0.5), 1.0),
            (GoalReward(), 20.0),
        )
    raise ValueError(f"Unknown reward table: {table}")


def create_training_env(
    n_envs: int, n_red: int, n_blue: int, stadium: str | None, tick_skip: int,
    seed: int | None = None, table: str = "corrected", repeat_kick: bool = True, approach_weight: float = 0.15,
    gamma=None, progress_weight=.08, goal_weight=25., pocket_offset=22., pocket_radius=2.,
):
    """Creates training environment with auto-calibrated obs_builder and multi-scenario resets."""
    engine = TransitionEngine(n_envs, n_red, n_blue, step_limit=2000, tick_skip=tick_skip, stadium=stadium)
    obs_builder = DefaultObs(pos_coef="auto")
    mutator = MultiSetter(
        [KickoffSetter(), RandomStateMutator(ball_speed=3.0, rng=seed)],
        weights=[0.4, 0.6],
        rng=seed,
    )
    return Env(
        engine=engine,
        obs_builder=obs_builder,
        action_parser=RepeatKickAction() if repeat_kick else DiscreteAction(kick_values=2),
        reward_fn=build_reward(table, stadium, tick_skip, approach_weight, gamma, progress_weight, goal_weight, pocket_offset, pocket_radius),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(2000),
        state_mutator=mutator,
    )


def create_eval_env(
    n_envs: int, n_red: int, n_blue: int, stadium: str | None, tick_skip: int,
    table: str = "corrected", repeat_kick: bool = True, approach_weight: float = 0.15,
    gamma=None, progress_weight=.08, goal_weight=25., pocket_offset=22., pocket_radius=2.,
):
    """Creates evaluation environment with standard kickoff resets."""
    engine = TransitionEngine(n_envs, n_red, n_blue, step_limit=2000, tick_skip=tick_skip, stadium=stadium)
    obs_builder = DefaultObs(pos_coef="auto")
    mutator = KickoffSetter()
    return Env(
        engine=engine,
        obs_builder=obs_builder,
        action_parser=RepeatKickAction() if repeat_kick else DiscreteAction(kick_values=2),
        reward_fn=build_reward(table, stadium, tick_skip, approach_weight, gamma, progress_weight, goal_weight, pocket_offset, pocket_radius),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(2000),
        state_mutator=mutator,
    )


@torch.no_grad()
def eval_vs_chase(eval_env, model, dev, n_red, n_blue, steps=350):
    """Net goals of model (red) vs ChaseBot (blue) over persistent eval_env."""
    obs = eval_env.reset()
    chase = ChaseBot(seed=0)
    P, od = n_red + n_blue, obs.shape[-1]
    n_envs = eval_env.engine.n_envs
    red_i, blue_i = np.arange(n_red), np.arange(n_red, P)
    rg = bg = 0
    for _ in range(steps):
        red = model.act(torch.as_tensor(obs[:, red_i].reshape(-1, od), dtype=torch.float32, device=dev))
        full = np.empty((n_envs, P, 3), dtype=np.int64)
        full[:, red_i] = red.cpu().numpy().reshape(n_envs, n_red, 3)
        full[:, blue_i] = chase.act(eval_env.prev_state, blue_i, n_red=n_red)
        obs, rew, term, trunc = eval_env.step(full)
        sc = eval_env.prev_state.scored
        rg += int((sc == BLUE).sum())  # blue conceded -> red scored
        bg += int((sc == RED).sum())
    return rg - bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--envs", type=int, default=512, help="parallel games")
    ap.add_argument("--red", type=int, default=1)
    ap.add_argument("--blue", type=int, default=1)
    ap.add_argument("--stadium", default=STADIUM, help="bundled name or .hbs path")
    ap.add_argument("--tick-skip", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument("--save-every", type=int, default=250, help="snapshot ckpts/it{N}.pt every N iters")
    ap.add_argument("--replay-every", type=int, default=0, help="dump replay every N iters (0=off)")
    ap.add_argument("--record-js", default=None)
    ap.add_argument("--warm-start", default=None, help="path to existing .pt checkpoint to continue training")
    ap.add_argument("--approach-weight", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-envs", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=400)
    ap.add_argument("--keep-critic", action="store_true", help="keep old value head on warm-start")
    ap.add_argument("--timeout-bootstrap", choices=("zero", "final"), default=None,
                    help="B0: zero; E1: bootstrap artificial timeout from pre-reset observation")
    ap.add_argument("--stop-at", help="Timezone-aware ISO deadline; finish and save before this time")
    ap.add_argument("--deadline-reserve-seconds", type=float, default=120.,
                    help="reserve for final evaluation and checkpoint (default 120 s)")
    ap.add_argument("--sprint1", action="store_true", help="PBRS + separate critic + fixed curriculum + dual-mode validation")
    ap.add_argument("--reward-table", choices=("corrected","pbrs","simple","simple_progress1","previous"))
    group=ap.add_mutually_exclusive_group()
    group.add_argument("--separate-critic",dest="separate_critic",action="store_true")
    group.add_argument("--shared-critic",dest="separate_critic",action="store_false")
    ap.set_defaults(separate_critic=None)
    ap.add_argument("--curriculum-frac",type=float)
    ap.add_argument("--anchor-checkpoint",action="append",default=[])
    ap.add_argument("--progress-weight",type=float,default=.08)
    ap.add_argument("--goal-weight",type=float,default=25.)
    ap.add_argument("--pocket-offset",type=float,default=22.)
    ap.add_argument("--pocket-radius",type=float,default=2.)
    ap.add_argument("--actor-lr",type=float,default=LR)
    ap.add_argument("--critic-lr",type=float,default=LR)
    ap.add_argument("--target-kl",type=float,help="stop remaining minibatches when measured KL exceeds this threshold")
    ap.add_argument("--eval-modes",nargs='+',choices=("greedy","sampled"))
    ap.add_argument("--selection-metric",choices=("mean","minimum"))
    ap.add_argument("--checkpoint-every",type=int,default=10)
    ap.add_argument("--critic-warmup-iters",type=int,help="calibrate only the separate critic before actor updates")
    ap.add_argument("--resume",help="resume.pt: restore model, Adam, pool and RNG; matches reset")
    args = ap.parse_args()
    resume = torch.load(args.resume,map_location='cpu',weights_only=True) if args.resume else None
    if resume is not None:
        if resume.get('schema_version') != 1:
            ap.error('unsupported resume schema')
        protected=('envs','red','blue','tick_skip','seed','reward_table','separate_critic',
                   'curriculum_frac','approach_weight','progress_weight','goal_weight',
                   'pocket_offset','pocket_radius','actor_lr','critic_lr','target_kl',
                   'eval_modes','selection_metric','timeout_bootstrap','iters','eval_envs','eval_steps','eval_every','critic_warmup_iters')
        resume['config'].setdefault('critic_warmup_iters',0)
        for key in protected:
            flag='--'+key.replace('_','-')
            if flag in sys.argv and getattr(args,key)!=resume['config'][key]:
                ap.error(f'resume cannot change {flag}')
            setattr(args,key,resume['config'][key])
        args.warm_start=None
        args.stadium=resume['config']['stadium']
        if not Path(args.stadium).is_file():
            restored_map=Path(args.out)/'resume_stadium.hbs'
            restored_map.parent.mkdir(parents=True,exist_ok=True)
            restored_map.write_text(resume['stadium_text'])
            args.stadium=str(restored_map.resolve())
    args.reward_table = args.reward_table or ('pbrs' if args.sprint1 else 'corrected')
    args.separate_critic = args.sprint1 if args.separate_critic is None else args.separate_critic
    args.curriculum_frac = (1. if args.sprint1 else 0.) if args.curriculum_frac is None else args.curriculum_frac
    args.timeout_bootstrap = args.timeout_bootstrap or ('final' if args.sprint1 else 'zero')
    args.eval_modes = args.eval_modes or (['greedy','sampled'] if args.sprint1 else ['greedy'])
    args.selection_metric = args.selection_metric or ('minimum' if args.sprint1 else 'mean')
    args.target_kl = (.03 if args.sprint1 else 0.) if args.target_kl is None else args.target_kl
    args.critic_warmup_iters = (20 if args.sprint1 and args.separate_critic else 0) if args.critic_warmup_iters is None else args.critic_warmup_iters
    if args.critic_warmup_iters<0 or (args.critic_warmup_iters>0 and not args.separate_critic):
        ap.error('critic warmup requires a separate critic and nonnegative iteration count')
    if args.reward_table=='pbrs' and args.timeout_bootstrap!='final':
        ap.error('PBRS assumes artificial timeouts: use --timeout-bootstrap final')
    if not 0<=args.curriculum_frac<=1 or min(args.actor_lr,args.critic_lr)<=0 or args.target_kl<0:
        ap.error('invalid curriculum/LR/KL values')
    if args.checkpoint_every<1:
        ap.error('checkpoint-every must be positive')
    if (args.sprint1 or args.curriculum_frac>0) and (args.red,args.blue)!=(1,1):
        ap.error('Sprint 1 and curriculum are currently validated only for 1v1')
    deadline = None
    if args.stop_at:
        try:
            deadline_dt = datetime.fromisoformat(args.stop_at)
            if deadline_dt.tzinfo is None:
                raise ValueError("timezone required")
            deadline = deadline_dt.timestamp()
        except ValueError as error:
            ap.error(f"invalid stop-at: {error}")
        if args.deadline_reserve_seconds < 1:
            ap.error("deadline reserve must be at least one second")
        if deadline <= time.time() + args.deadline_reserve_seconds + 1:
            ap.error("stop-at must leave more time than deadline reserve")
    if args.envs < 1 or args.iters < 1 or args.eval_every < 1:
        ap.error("envs, iters and eval-every must be positive")

    dev = torch.device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    gamma = discount_gamma(args.tick_skip,HALF_LIFE_S)
    ppo_contract=dict(rollout_steps=T,epochs=EPOCHS,minibatches=N_MB,clip=CLIP,
                      value_coef=VF_COEF,max_grad=MAX_GRAD,lambda_gae=LAM,gamma=gamma)
    if resume and resume['config'].get('ppo_contract') != ppo_contract:
        ap.error('resume PPO contract differs from the checkpoint')
    env = create_training_env(args.envs,args.red,args.blue,args.stadium,args.tick_skip,
          seed=args.seed,table=args.reward_table,approach_weight=args.approach_weight,
          gamma=gamma,progress_weight=args.progress_weight,goal_weight=args.goal_weight,
          pocket_offset=args.pocket_offset,pocket_radius=args.pocket_radius)

    P = args.red + args.blue
    obs_dim = env.obs_dim
    n_acts = env.action_space

    slots = np.arange(args.envs * P).reshape(args.envs, P)
    learner_idx = slots[:, : args.red].reshape(-1)
    opp_idx = slots[:, args.red :].reshape(-1)
    learner_t = torch.as_tensor(learner_idx, device=dev)
    opp_t = torch.as_tensor(opp_idx, device=dev)
    B = learner_idx.size

    model = Policy(obs_dim, n_acts,separate_critic=args.separate_critic).to(dev)
    if args.warm_start:
        ws_path = Path(args.warm_start)
        if not ws_path.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {ws_path}")
        if ws_path.exists():
            print(f"Warm-starting policy weights from {ws_path}")
            model.load_compatible(torch.load(ws_path,map_location=dev,weights_only=True),keep_critic=args.keep_critic)
    opp = Policy(obs_dim, n_acts).to(dev)
    opp.eval()
    if args.separate_critic:
        opt=torch.optim.Adam([{'params':model.actor_parameters(),'lr':args.actor_lr,'base_lr':args.actor_lr},
                              {'params':model.critic_parameters(),'lr':args.critic_lr,'base_lr':args.critic_lr}])
    else:
        opt=torch.optim.Adam(model.parameters(),lr=args.actor_lr)
        opt.param_groups[0]['base_lr']=args.actor_lr
    pool: list[dict] = []
    if args.warm_start and Path(args.warm_start).exists():
        pool.append(model.actor_state_dict())

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"self-play PPO (snapshot pool) | {args.red}v{args.blue} {args.stadium or 'classic'} | "
        f"obs_dim={obs_dim} n_acts={n_acts} train_slots={B} gamma={gamma:.4f} device={dev} iters={args.iters}"
    )

    S_obs = torch.zeros(T, B, obs_dim, device=dev)
    S_act = torch.zeros(T, B, 3, dtype=torch.long, device=dev)
    S_logp = torch.zeros(T, B, device=dev)
    S_val = torch.zeros(T, B, device=dev)
    S_rew = torch.zeros(T, B, device=dev)
    S_done = torch.zeros(T, B, dtype=torch.bool, device=dev)
    S_term = torch.zeros_like(S_done)
    S_final_val = torch.zeros(T, B, device=dev)

    obs = env.reset().reshape(args.envs * P, obs_dim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists():
        raise FileExistsError('Use a new output directory; existing telemetry must not be overwritten')
    best = -1.0
    t0 = time.perf_counter()
    config = dict(vars(args), obs_dim=obs_dim, n_acts=list(n_acts), hidden=256, depth=2,
                  rollout_steps=T, epochs=EPOCHS, minibatches=N_MB, gamma=gamma,
                  reward_weights={"goal":args.goal_weight,"approach":args.approach_weight,"progress":args.progress_weight},
                  obs_schema="default16" if obs_dim==16 else "default",ppo_contract=ppo_contract,
                  repeat_kick=True, samples_per_iteration=T*B,
                  gradient_metric_sampling='first_minibatch_each_epoch',
                  actor_gradient_objective='ppo_surrogate_minus_entropy',
                  critic_gradient_objective='VF_COEF_times_value_loss',
                  revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    config['warm_start_sha256'] = hashlib.sha256(Path(args.warm_start).read_bytes()).hexdigest() if args.warm_start else None
    telemetry = Telemetry(out, config)
    meter = RewardMeter(env.reward_fn)
    env.reward_fn = meter
    reference = Policy(obs_dim,n_acts).to(dev)
    reference.load_actor_state(model.actor_state_dict()); reference.eval(); reference.requires_grad_(False)
    anchors=[]
    for state in (resume['anchors'] if resume else [torch.load(path,map_location='cpu',weights_only=True)
                                                   for path in args.anchor_checkpoint]):
        anchor=Policy(obs_dim,n_acts).to(dev);anchor.load_actor_state(state);anchor.eval();anchor.requires_grad_(False);anchors.append(anchor)
    curriculum=OpponentCurriculum(args.envs,args.curriculum_frac,[reference,*anchors],args.seed+101)
    config['fixed_opponent_probabilities']=dict(zip(curriculum.names,curriculum.weights.tolist()))
    from haxballgym.stadium import stadium_text
    config['stadium_sha256']=hashlib.sha256(stadium_text(args.stadium).encode()).hexdigest()
    if resume and config['stadium_sha256'] != resume['config']['stadium_sha256']:
        raise ValueError('Resumed stadium differs from checkpoint')
    config['resume_semantics']='model/Adam/schedule/pool/RNG preserved; native matches reset'
    # Rewrite config once additional provenance has been computed.
    from telemetry import atomic_json
    atomic_json(out/'config.json',config,durable=True)
    atomic_torch_save(snapshot(), out/'initial_policy.pt')
    train_seconds=0.
    start_iteration=0
    best_state=None
    best_iteration=0
    if resume:
        model.load_state_dict(resume['model'])
        reference.load_actor_state(resume['reference'])
        opt.load_state_dict(resume['optimizer'])
        pool=resume['pool'];best=resume['best'];start_iteration=resume['iteration']
        best_state=resume['best_model'];best_iteration=resume['best_iteration']
        train_seconds=resume['train_seconds']
        restore_random_state(resume['random'],rng)
        curriculum.load_state_dict(resume['curriculum'])
        atomic_torch_save(model.state_dict(),out/'initial_policy.pt')
        if start_iteration>=args.iters:
            raise ValueError('Resume checkpoint already reached its iteration budget')
    stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)

    def save_resume(iteration):
        payload=dict(schema_version=1,config=config,iteration=iteration,model=snapshot(),
            optimizer=opt.state_dict(),reference=reference.actor_state_dict(),
            anchors=[a.actor_state_dict() for a in anchors],pool=pool,best=best,
            best_model=best_state,best_iteration=best_iteration,
            train_seconds=train_seconds,random=random_state(rng),curriculum=curriculum.state_dict(),
            stadium_text=stadium_text(args.stadium))
        atomic_torch_save(payload,out/'resume.pt')

    def evaluate(iteration):
        telemetry.update(phase='evaluating', evaluation_iteration=iteration, evaluation_completed=0)
        rows=[]
        for mode in args.eval_modes:
            rows.extend(panel(model,reference,args.stadium,args.seed,
              lambda row:telemetry.append('evaluation_cases.jsonl',dict(iteration=iteration,**row)),
              lambda case,total,bot,scenario:telemetry.update(phase='evaluating',evaluation_completed=case,
                evaluation_total=total,evaluation_opponent=bot,evaluation_scenario=scenario,evaluation_mode=mode),
              n=args.eval_envs,steps=args.eval_steps,tick_skip=args.tick_skip,greedy=mode=='greedy',
              extra_opponents={f'anchor_{i}':a for i,a in enumerate(anchors)}))
        mean_score=balanced_score(rows)
        groups={f'{mode}_{scenario}':balanced_score([r for r in rows if r['policy_mode']==mode and r['scenario']==scenario])
                for mode in args.eval_modes for scenario in ('random','kickoff')}
        valid=[v for v in groups.values() if v is not None]
        score=min(valid) if valid and args.selection_metric=='minimum' else mean_score
        if len(valid)!=len(groups):score=None
        scenario_scores={f'eval_{scenario}_score':balanced_score([r for r in rows if r['scenario']==scenario])
                         for scenario in ('random','kickoff')}
        telemetry.append('evaluations.jsonl',dict(iteration=iteration,score=score,mean_score=mean_score,
                             mode_scenario_scores=groups,**scenario_scores,cases=rows))
        return score

    try:
        baseline_score=evaluate(start_iteration)
        if not resume:
            best=baseline_score if baseline_score is not None else -1.
            best_state=snapshot();best_iteration=start_iteration
        elif baseline_score is not None and baseline_score>best:
            best=baseline_score;best_state=snapshot();best_iteration=start_iteration
        atomic_torch_save(best_state,out/'model.pt')
        atomic_json(out/'meta.json',dict(config,it=best_iteration,evaluation_score=best),durable=True)

        completed = start_iteration
        save_resume(completed)
        for it in range(start_iteration+1,args.iters+1):
            if stop_requested:break
            if deadline is not None and time.time() >= deadline - args.deadline_reserve_seconds:
                break
            iteration_start=time.perf_counter()
            meter.clear()
            telemetry.update(phase="rollout",iteration=it,rollout_step=0)
            frac = (it - 1) / max(1, args.iters - 1)
            for g in opt.param_groups:
                base=g["base_lr"]
                g["lr"] = base * (1 - frac) + (base / 3) * frac
            ent_coef = ENT0 * (1 - frac) + ENT1 * frac

            # pick frozen opponent: current copy or past snapshot
            opp.load_actor_state(pool[rng.integers(len(pool))]
                                 if (pool and rng.random()<POOL_PROB) else model.actor_state_dict())

            # --- rollout ---
            for t in range(T):
                o = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                with torch.no_grad():
                    act, logp, val = model.act_train(o[learner_t])
                    o_act = (torch.as_tensor(curriculum.act(env.prev_state,o[opp_t],opp),device=dev)
                             if args.curriculum_frac>0 else opp.act(o[opp_t]))
                S_obs[t], S_act[t], S_logp[t], S_val[t] = o[learner_t], act, logp, val
                full = np.empty((args.envs * P, 3), dtype=np.int64)
                full[learner_idx] = act.cpu().numpy()
                full[opp_idx] = o_act.cpu().numpy()
                nobs, rew, term, trunc = env.step(full.reshape(args.envs, P, 3))
                obs = nobs.reshape(args.envs * P, obs_dim)
                if args.curriculum_frac>0:curriculum.reset_episodes(term|trunc)
                done = np.repeat(term | trunc, P)
                S_term[t] = torch.as_tensor(np.repeat(term, P)[learner_idx], device=dev)
                S_final_val[t].zero_()
                timeout = trunc & ~term
                if timeout.any():
                    # Evaluate final observations in BOTH arms for comparable diagnostics.
                    final_obs = env.obs_builder.build_obs(meter.last_state)
                    final_red = final_obs[timeout, :args.red].reshape(-1, obs_dim)
                    timeout_slots = torch.as_tensor(np.repeat(timeout, args.red), device=dev)
                    with torch.no_grad():
                        S_final_val[t, timeout_slots] = model.get_value(
                            torch.as_tensor(final_red,dtype=torch.float32,device=dev))
                S_rew[t] = torch.as_tensor(rew.reshape(-1)[learner_idx], dtype=torch.float32, device=dev)
                S_done[t] = torch.as_tensor(done[learner_idx], device=dev)
                if stop_requested:break
                if (t+1)%8==0:
                    telemetry.update(phase='rollout',rollout_step=t+1)

            if stop_requested:break
            rollout_seconds=time.perf_counter()-iteration_start
            telemetry.update(phase="updating")
            # --- GAE advantages ---
            with torch.no_grad():
                last_val = model.get_value(torch.as_tensor(obs,dtype=torch.float32,device=dev)[learner_t])
            adv, ret = gae_returns(S_rew, S_val, S_term, S_done, S_final_val,
                                   last_val, gamma, LAM, args.timeout_bootstrap == 'final')
            timeout_values = S_final_val[S_done & ~S_term]

            # --- PPO update ---
            bo, ba, bl = S_obs.reshape(-1, obs_dim), S_act.reshape(-1, 3), S_logp.reshape(-1)
            badv, bret = adv.reshape(-1), ret.reshape(-1)
            badv = (badv - badv.mean()) / (badv.std() + 1e-8)
            idx = np.arange(T * B)
            mb = len(idx) // N_MB
            ppo_metrics=[]
            gradient_metrics=[]
            combined_norms=[]
            kl_stopped=False
            actor_update_norms=[]
            for _ in range(EPOCHS):
                np.random.shuffle(idx)
                for s in range(0, len(idx), mb):
                    mi = torch.as_tensor(idx[s : s + mb], device=dev)
                    logp, ent, val = model.evaluate(bo[mi], ba[mi])
                    ratio = (logp - bl[mi]).exp()
                    a = badv[mi]
                    pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
                    vl = 0.5 * (val - bret[mi]).pow(2).mean()
                    actor_loss = pl - ent_coef * ent.mean()
                    critic_loss = VF_COEF * vl
                    if it<=args.critic_warmup_iters:
                        actor_loss=0.*actor_loss
                    loss = actor_loss + critic_loss
                    if s == 0:
                        gradient_metrics.append(objective_gradient_metrics(model, actor_loss, critic_loss))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite PPO loss; last durable resume remains available')
                    measured_kl=float(((ratio-1)-(logp-bl[mi])).mean().detach())
                    if args.target_kl and measured_kl>args.target_kl:
                        kl_stopped=True
                        break
                    before=[p.detach().clone() for p in model.actor_parameters()] if s==0 else None
                    opt.zero_grad()
                    loss.backward()
                    if it<=args.critic_warmup_iters:
                        # No zero-gradient Adam steps: actor moments start only when actor learning starts.
                        for parameter in model.actor_parameters():parameter.grad=None
                    if args.separate_critic:
                        an=nn.utils.clip_grad_norm_(model.actor_parameters(),MAX_GRAD,error_if_nonfinite=True)
                        cn=nn.utils.clip_grad_norm_(model.critic_parameters(),MAX_GRAD,error_if_nonfinite=True)
                        combined_norms.append(float(torch.sqrt(an.square()+cn.square())))
                    else:
                        combined_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD,error_if_nonfinite=True)))
                    opt.step()
                    if before is not None:
                        actor_update_norms.append(float(torch.sqrt(sum((p.detach()-v).square().sum()
                                                   for p,v in zip(model.actor_parameters(),before)))))
                    with torch.no_grad():
                        lr_delta=logp-bl[mi]
                        kl=((lr_delta.exp()-1)-lr_delta).mean()
                        clipped=((ratio-1).abs()>CLIP).float().mean()
                        ppo_metrics.append([float(pl),float(vl),float(ent.mean()),float(kl),float(clipped)])

                if kl_stopped:break
            iteration_seconds=time.perf_counter()-iteration_start
            train_seconds+=iteration_seconds
            means=np.mean(ppo_metrics,axis=0) if ppo_metrics else np.zeros(5)
            yvar=float(bret.var(unbiased=False))
            explained=1-float((bret-S_val.reshape(-1)).var(unbiased=False))/yvar if yvar>1e-9 else None
            grad_summary = {}
            for key in gradient_metrics[0]:
                measured = [row[key] for row in gradient_metrics if row[key] is not None]
                grad_summary[key] = float(np.mean(measured)) if measured else None
            record=dict(**grad_summary,critic_warmup=it<=args.critic_warmup_iters,kl_early_stop=kl_stopped,optimizer_minibatches=len(ppo_metrics),
               actor_update_norm=float(np.mean(actor_update_norms)) if actor_update_norms else 0.,
               opponents=curriculum.counts() if args.curriculum_frac>0 else {'selfplay':args.envs},
               gradient_metric_minibatches=len(gradient_metrics),
               combined_grad_norm_pre_clip=float(np.mean(combined_norms)) if combined_norms else 0.,
               timeout_count=int(timeout_values.numel()),
               timeout_bootstrap_mean=float(timeout_values.mean()) if timeout_values.numel() else None,
               timeout_bootstrap_applied=args.timeout_bootstrap == 'final',iteration=it,samples=it*T*B,iteration_seconds=iteration_seconds,
               rollout_seconds=rollout_seconds,update_seconds=iteration_seconds-rollout_seconds,
               samples_per_second=T*B/iteration_seconds,training_samples_per_second=it*T*B/train_seconds,
               wall_samples_per_second=(it-start_iteration)*T*B/(time.perf_counter()-t0),
               reward_mean=float(S_rew.mean()),reward_terms_per_sample={k:v/(T*B) for k,v in meter.sums.items()},
               reward_terms_absolute_per_sample={k:v/(T*B) for k,v in meter.absolute_sums.items()},
               goals_for=meter.gf,goals_against=meter.ga,completed_episodes=meter.finished,
               stationary_fraction=meter.stationary/max(1,meter.transitions),
               policy_loss=float(means[0]),value_loss=float(means[1]),entropy=float(means[2]),
               approximate_kl=float(means[3]),clip_fraction=float(means[4]),explained_variance=explained,
               learning_rate=opt.param_groups[0]['lr'],critic_learning_rate=opt.param_groups[-1]['lr'],snapshot_pool=len(pool))
            telemetry.append('training.jsonl',record)
            telemetry.update(completed_iterations=it,samples=it*T*B,phase='checkpoint',last_training=record)

            if it % SNAPSHOT_EVERY == 0:
                pool.append(model.actor_state_dict())
                if len(pool) > POOL_SIZE:
                    pool.pop(0)

            if args.save_every and it % args.save_every == 0:
                (out / "ckpts").mkdir(exist_ok=True)
                atomic_torch_save(model.state_dict(), out / "ckpts" / f"it{it}.pt")

            completed = it
            if not stop_requested and (it % args.eval_every == 0 or it == args.iters or (deadline is not None and time.time() >= deadline - args.deadline_reserve_seconds)):
                score=evaluate(it)
                meta=dict(config,it=it,evaluation_score=score)
                atomic_torch_save(snapshot(),out/'model_final.pt')
                atomic_json(out/'meta_final.json',meta,durable=True)
                tag=''
                if score is not None and score>best:
                    best=score;best_state=snapshot();best_iteration=it
                    atomic_torch_save(best_state,out/'model.pt')
                    atomic_json(out/'meta.json',meta,durable=True);tag=' <- saved best'
                score_label=f'{score:.3f}' if score is not None else 'n/a'
                print(f'it {it:4d} | {record["samples_per_second"]/1000:.1f}k samples/s | balanced eval {score_label} (best {best:.3f}){tag}',flush=True)

            if it % args.checkpoint_every==0:
                save_resume(completed)

        # If the deadline was reached between iterations, preserve the exact endpoint.
        final_meta = out/'meta_final.json'
        final_it = json.loads(final_meta.read_text())['it'] if final_meta.exists() else -1
        if final_it != completed:
            score = evaluate(completed) if not stop_requested and time.time() < (deadline or float('inf')) - 30 else None
            atomic_torch_save(snapshot(), out/'model_final.pt')
            atomic_json(final_meta,dict(config,it=completed,evaluation_score=score),durable=True)
        save_resume(completed)
        telemetry.update(status="completed",phase="completed",completed_iterations=completed,
                         stop_reason='signal' if stop_requested else ('deadline' if completed<args.iters else 'iteration_budget'))
        print(f"done. balanced score {best:.3f} -> {out / 'model.pt'}",flush=True)
    except BaseException as error:
        telemetry.update(status='failed',phase='failed',error=repr(error))
        raise


if __name__ == "__main__":
    main()

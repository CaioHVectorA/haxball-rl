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
from environment import STADIUM, motion_scale, MotionPlayerToBall, MotionBallToGoal, RepeatKickAction
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


def build_reward(table="corrected", stadium=STADIUM, tick_skip=8, approach_weight=0.15):
    """Signed shaping; retain the previous table for controlled POC comparisons."""
    if table == "corrected":
        scale = motion_scale(stadium, tick_skip)
        return CombinedReward(
            (MotionPlayerToBall(scale), approach_weight),
            (MotionBallToGoal(scale), 0.08),
            (GoalReward(), 25.0),
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
        reward_fn=build_reward(table, stadium, tick_skip, approach_weight),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(2000),
        state_mutator=mutator,
    )


def create_eval_env(
    n_envs: int, n_red: int, n_blue: int, stadium: str | None, tick_skip: int,
    table: str = "corrected", repeat_kick: bool = True, approach_weight: float = 0.15,
):
    """Creates evaluation environment with standard kickoff resets."""
    engine = TransitionEngine(n_envs, n_red, n_blue, step_limit=2000, tick_skip=tick_skip, stadium=stadium)
    obs_builder = DefaultObs(pos_coef="auto")
    mutator = KickoffSetter()
    return Env(
        engine=engine,
        obs_builder=obs_builder,
        action_parser=RepeatKickAction() if repeat_kick else DiscreteAction(kick_values=2),
        reward_fn=build_reward(table, stadium, tick_skip, approach_weight),
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
    ap.add_argument("--timeout-bootstrap", choices=("zero", "final"), default="zero",
                    help="B0: zero; E1: bootstrap artificial timeout from pre-reset observation")
    ap.add_argument("--stop-at", help="Timezone-aware ISO deadline; finish and save before this time")
    ap.add_argument("--deadline-reserve-seconds", type=float, default=120.,
                    help="reserve for final evaluation and checkpoint (default 120 s)")
    args = ap.parse_args()
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

    env = create_training_env(args.envs, args.red, args.blue, args.stadium, args.tick_skip, seed=args.seed, approach_weight=args.approach_weight)
    eval_env = create_eval_env(128, args.red, args.blue, args.stadium, args.tick_skip)

    P = args.red + args.blue
    obs_dim = env.obs_dim
    n_acts = env.action_space
    gamma = math.exp(math.log(0.5) / ((60 / args.tick_skip) * HALF_LIFE_S))

    slots = np.arange(args.envs * P).reshape(args.envs, P)
    learner_idx = slots[:, : args.red].reshape(-1)
    opp_idx = slots[:, args.red :].reshape(-1)
    learner_t = torch.as_tensor(learner_idx, device=dev)
    opp_t = torch.as_tensor(opp_idx, device=dev)
    B = learner_idx.size

    model = Policy(obs_dim, n_acts).to(dev)
    if args.warm_start:
        ws_path = Path(args.warm_start)
        if not ws_path.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {ws_path}")
        if ws_path.exists():
            print(f"Warm-starting policy weights from {ws_path}")
            model.load_state_dict(torch.load(ws_path, map_location=dev))
            if not args.keep_critic:
                nn.init.zeros_(model.value.weight)
                nn.init.zeros_(model.value.bias)
    opp = Policy(obs_dim, n_acts).to(dev)
    opp.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pool: list[dict] = []
    if args.warm_start and Path(args.warm_start).exists():
        pool.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

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
    best = -1.0
    t0 = time.perf_counter()
    config = dict(vars(args), obs_dim=obs_dim, n_acts=list(n_acts), hidden=256, depth=2,
                  rollout_steps=T, epochs=EPOCHS, minibatches=N_MB, gamma=gamma,
                  reward_weights={"goal":25,"approach":args.approach_weight,"progress":0.08},
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
    reference.load_state_dict(model.state_dict()); reference.eval()
    torch.save(snapshot(), out/'initial_policy.pt')
    train_seconds=0.

    def evaluate(iteration):
        telemetry.update(phase='evaluating', evaluation_iteration=iteration, evaluation_completed=0)
        rows=panel(model, reference, args.stadium, args.seed,
          lambda row:telemetry.append('evaluation_cases.jsonl',dict(iteration=iteration,**row)),
          lambda case,total,bot,scenario:telemetry.update(phase='evaluating',evaluation_completed=case,
            evaluation_total=total,evaluation_opponent=bot,evaluation_scenario=scenario),
          n=args.eval_envs,steps=args.eval_steps,tick_skip=args.tick_skip)
        score=balanced_score(rows)
        scenario_scores={f'eval_{scenario}_score':balanced_score([r for r in rows if r['scenario']==scenario])
                         for scenario in ('random','kickoff')}
        telemetry.append('evaluations.jsonl',dict(iteration=iteration,score=score,**scenario_scores,cases=rows))
        return score

    try:
        baseline_score=evaluate(0)
        best=baseline_score if baseline_score is not None else -1.
        torch.save(snapshot(),out/'model.pt')
        (out/'meta.json').write_text(json.dumps(dict(config, it=0, evaluation_score=baseline_score)))

        completed = 0
        for it in range(1, args.iters + 1):
            if deadline is not None and time.time() >= deadline - args.deadline_reserve_seconds:
                break
            iteration_start=time.perf_counter()
            meter.clear()
            telemetry.update(phase="rollout",iteration=it,rollout_step=0)
            frac = (it - 1) / max(1, args.iters - 1)
            for g in opt.param_groups:
                g["lr"] = LR * (1 - frac) + (LR / 3) * frac
            ent_coef = ENT0 * (1 - frac) + ENT1 * frac

            # pick frozen opponent: current copy or past snapshot
            opp.load_state_dict(
                pool[rng.integers(len(pool))] if (pool and rng.random() < POOL_PROB) else snapshot()
            )

            # --- rollout ---
            for t in range(T):
                o = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                with torch.no_grad():
                    act, logp, val = model.act_train(o[learner_t])
                    o_act = opp.act(o[opp_t])
                S_obs[t], S_act[t], S_logp[t], S_val[t] = o[learner_t], act, logp, val
                full = np.empty((args.envs * P, 3), dtype=np.int64)
                full[learner_idx] = act.cpu().numpy()
                full[opp_idx] = o_act.cpu().numpy()
                nobs, rew, term, trunc = env.step(full.reshape(args.envs, P, 3))
                obs = nobs.reshape(args.envs * P, obs_dim)
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
                        S_final_val[t, timeout_slots] = model._logits(
                            torch.as_tensor(final_red, dtype=torch.float32, device=dev))[1]
                S_rew[t] = torch.as_tensor(rew.reshape(-1)[learner_idx], dtype=torch.float32, device=dev)
                S_done[t] = torch.as_tensor(done[learner_idx], device=dev)
                if (t+1)%8==0:
                    telemetry.update(phase='rollout',rollout_step=t+1)

            rollout_seconds=time.perf_counter()-iteration_start
            telemetry.update(phase="updating")
            # --- GAE advantages ---
            with torch.no_grad():
                last_val = model._logits(torch.as_tensor(obs, dtype=torch.float32, device=dev)[learner_t])[1]
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
                    loss = actor_loss + critic_loss
                    if s == 0:
                        gradient_metrics.append(objective_gradient_metrics(model, actor_loss, critic_loss))
                    opt.zero_grad()
                    loss.backward()
                    combined_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)))
                    opt.step()
                    with torch.no_grad():
                        lr_delta=logp-bl[mi]
                        kl=((lr_delta.exp()-1)-lr_delta).mean()
                        clipped=((ratio-1).abs()>CLIP).float().mean()
                        ppo_metrics.append([float(pl),float(vl),float(ent.mean()),float(kl),float(clipped)])

            iteration_seconds=time.perf_counter()-iteration_start
            train_seconds+=iteration_seconds
            means=np.mean(ppo_metrics,axis=0)
            yvar=float(bret.var(unbiased=False))
            explained=1-float((bret-S_val.reshape(-1)).var(unbiased=False))/yvar if yvar>1e-9 else None
            grad_summary = {}
            for key in gradient_metrics[0]:
                measured = [row[key] for row in gradient_metrics if row[key] is not None]
                grad_summary[key] = float(np.mean(measured)) if measured else None
            record=dict(**grad_summary,
               gradient_metric_minibatches=len(gradient_metrics),
               combined_grad_norm_pre_clip=float(np.mean(combined_norms)),
               timeout_count=int(timeout_values.numel()),
               timeout_bootstrap_mean=float(timeout_values.mean()) if timeout_values.numel() else None,
               timeout_bootstrap_applied=args.timeout_bootstrap == 'final',iteration=it,samples=it*T*B,iteration_seconds=iteration_seconds,
               rollout_seconds=rollout_seconds,update_seconds=iteration_seconds-rollout_seconds,
               samples_per_second=T*B/iteration_seconds,training_samples_per_second=it*T*B/train_seconds,
               wall_samples_per_second=it*T*B/(time.perf_counter()-t0),
               reward_mean=float(S_rew.mean()),reward_terms_per_sample={k:v/(T*B) for k,v in meter.sums.items()},
               reward_terms_absolute_per_sample={k:v/(T*B) for k,v in meter.absolute_sums.items()},
               goals_for=meter.gf,goals_against=meter.ga,completed_episodes=meter.finished,
               stationary_fraction=meter.stationary/max(1,meter.transitions),
               policy_loss=float(means[0]),value_loss=float(means[1]),entropy=float(means[2]),
               approximate_kl=float(means[3]),clip_fraction=float(means[4]),explained_variance=explained,
               learning_rate=opt.param_groups[0]['lr'],snapshot_pool=len(pool))
            telemetry.append('training.jsonl',record)
            telemetry.update(completed_iterations=it,samples=it*T*B,phase='checkpoint',last_training=record)

            if it % SNAPSHOT_EVERY == 0:
                pool.append(snapshot())
                if len(pool) > POOL_SIZE:
                    pool.pop(0)

            if args.save_every and it % args.save_every == 0:
                (out / "ckpts").mkdir(exist_ok=True)
                torch.save(model.state_dict(), out / "ckpts" / f"it{it}.pt")

            completed = it
            if it % args.eval_every == 0 or it == args.iters or (deadline is not None and time.time() >= deadline - args.deadline_reserve_seconds):
                score=evaluate(it)
                meta=dict(config,it=it,evaluation_score=score)
                torch.save(snapshot(),out/'model_final.pt')
                (out/'meta_final.json').write_text(json.dumps(meta))
                tag=''
                if score is not None and score>best:
                    best=score
                    torch.save(snapshot(),out/'model.pt')
                    (out/'meta.json').write_text(json.dumps(meta));tag=' <- saved best'
                score_label=f'{score:.3f}' if score is not None else 'n/a'
                print(f'it {it:4d} | {record["samples_per_second"]/1000:.1f}k samples/s | balanced eval {score_label} (best {best:.3f}){tag}',flush=True)

        # If the deadline was reached between iterations, preserve the exact endpoint.
        final_meta = out/'meta_final.json'
        final_it = json.loads(final_meta.read_text())['it'] if final_meta.exists() else -1
        if final_it != completed:
            score = evaluate(completed) if time.time() < (deadline or float('inf')) - 30 else None
            torch.save(snapshot(), out/'model_final.pt')
            final_meta.write_text(json.dumps(dict(config,it=completed,evaluation_score=score)))
        telemetry.update(status="completed",phase="completed",completed_iterations=completed,
                         stop_reason='deadline' if completed < args.iters else 'iteration_budget')
        print(f"done. balanced score {best:.3f} -> {out / 'model.pt'}",flush=True)
    except BaseException as error:
        telemetry.update(status='failed',phase='failed',error=repr(error))
        raise


if __name__ == "__main__":
    main()

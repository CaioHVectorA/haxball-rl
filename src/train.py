"""Train a WazBot by self-play PPO — calibrated with longer temporal horizons
and rich randomized in-game scenarios.

    uv run src/train.py --stadium futsal-classic --iters 1500 --save-every 250
"""

from __future__ import annotations

import argparse
import json
import math
import time
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


def create_training_env(n_envs: int, n_red: int, n_blue: int, stadium: str | None, tick_skip: int):
    """Creates training environment with auto-calibrated obs_builder and multi-scenario resets."""
    engine = TransitionEngine(n_envs, n_red, n_blue, step_limit=2000, tick_skip=tick_skip, stadium=stadium)
    obs_builder = DefaultObs(pos_coef="auto")
    mutator = MultiSetter(
        [KickoffSetter(), RandomStateMutator(ball_speed=3.0)],
        weights=[0.4, 0.6],
    )
    return Env(
        engine=engine,
        obs_builder=obs_builder,
        action_parser=DiscreteAction(kick_values=2),
        reward_fn=CombinedReward(
            (VelocityPlayerToBall(), 1.0),
            (TouchVelocityToGoal(scale=1.5), 1.0),
            (GoalReward(), 5.0),
        ),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(2000),
        state_mutator=mutator,
    )


def create_eval_env(n_envs: int, n_red: int, n_blue: int, stadium: str | None, tick_skip: int):
    """Creates evaluation environment with standard kickoff resets."""
    engine = TransitionEngine(n_envs, n_red, n_blue, step_limit=2000, tick_skip=tick_skip, stadium=stadium)
    obs_builder = DefaultObs(pos_coef="auto")
    mutator = KickoffSetter()
    return Env(
        engine=engine,
        obs_builder=obs_builder,
        action_parser=DiscreteAction(kick_values=2),
        reward_fn=CombinedReward(
            (VelocityPlayerToBall(), 1.0),
            (TouchVelocityToGoal(scale=1.5), 1.0),
            (GoalReward(), 5.0),
        ),
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
    ap.add_argument("--stadium", default="futsal-classic", help="bundled name or .hbs path")
    ap.add_argument("--tick-skip", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument("--save-every", type=int, default=250, help="snapshot ckpts/it{N}.pt every N iters")
    ap.add_argument("--replay-every", type=int, default=0, help="dump replay every N iters (0=off)")
    ap.add_argument("--record-js", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = create_training_env(args.envs, args.red, args.blue, args.stadium, args.tick_skip)
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
    opp = Policy(obs_dim, n_acts).to(dev)
    opp.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pool: list[dict] = []

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
    S_done = torch.zeros(T, B, device=dev)

    obs = env.reset().reshape(args.envs * P, obs_dim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -(10**9)
    t0 = time.perf_counter()

    for it in range(1, args.iters + 1):
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
            done = np.repeat((term | trunc).astype(np.float32), P)
            S_rew[t] = torch.as_tensor(rew.reshape(-1)[learner_idx], dtype=torch.float32, device=dev)
            S_done[t] = torch.as_tensor(done[learner_idx], device=dev)

        # --- GAE advantages ---
        with torch.no_grad():
            last_val = model._logits(torch.as_tensor(obs, dtype=torch.float32, device=dev)[learner_t])[1]
        adv = torch.zeros(T, B, device=dev)
        gae = torch.zeros(B, device=dev)
        for t in reversed(range(T)):
            nonterm = 1.0 - S_done[t]
            nextval = last_val if t == T - 1 else S_val[t + 1]
            delta = S_rew[t] + gamma * nextval * nonterm - S_val[t]
            gae = delta + gamma * LAM * nonterm * gae
            adv[t] = gae
        ret = adv + S_val

        # --- PPO update ---
        bo, ba, bl = S_obs.reshape(-1, obs_dim), S_act.reshape(-1, 3), S_logp.reshape(-1)
        badv, bret = adv.reshape(-1), ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        idx = np.arange(T * B)
        mb = len(idx) // N_MB
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, len(idx), mb):
                mi = torch.as_tensor(idx[s : s + mb], device=dev)
                logp, ent, val = model.evaluate(bo[mi], ba[mi])
                ratio = (logp - bl[mi]).exp()
                a = badv[mi]
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
                vl = 0.5 * (val - bret[mi]).pow(2).mean()
                loss = pl + VF_COEF * vl - ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                opt.step()

        if it % SNAPSHOT_EVERY == 0:
            pool.append(snapshot())
            if len(pool) > POOL_SIZE:
                pool.pop(0)

        if args.save_every and it % args.save_every == 0:
            (out / "ckpts").mkdir(exist_ok=True)
            torch.save(model.state_dict(), out / "ckpts" / f"it{it}.pt")

        if it % 50 == 0 or it == args.iters:
            net = eval_vs_chase(eval_env, model, dev, args.red, args.blue)
            sps = (it * T * B) / (time.perf_counter() - t0)

            def meta(n):
                return json.dumps(
                    {
                        "obs_dim": obs_dim,
                        "n_acts": list(n_acts),
                        "hidden": 256,
                        "depth": 2,
                        "n_red": args.red,
                        "n_blue": args.blue,
                        "stadium": args.stadium or "classic",
                        "net_vs_chase": n,
                        "it": it,
                    }
                )

            torch.save(model.state_dict(), out / "model_final.pt")
            (out / "meta_final.json").write_text(meta(net))
            tag = ""
            if net > best:
                best = net
                torch.save(model.state_dict(), out / "model.pt")
                (out / "meta.json").write_text(meta(net))
                tag = "  <- saved best"
            print(f"it {it:4d} | {sps / 1e3:4.0f}k steps/s | net vs chase {net:+d} (best {best:+d}){tag}")

    print(f"done. best-by-chase {best:+d} -> {out / 'model.pt'} ; final -> {out / 'model_final.pt'}")


if __name__ == "__main__":
    main()

"""Paired short reward experiment; writes only to a dedicated output directory."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from bot import ChaseBot, RandomBot
from haxballgym import BLUE, RED
from model import Policy
from environment import STADIUM
from train import (build_reward, create_training_env, create_eval_env, T, EPOCHS,
                   N_MB, CLIP, VF_COEF, MAX_GRAD, LAM, LR, ENT0, ENT1, HALF_LIFE_S)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def controlled_checks():
    env = create_eval_env(1, 1, 1, "futsal-classic", 8, table="simple")
    env.reset()
    state = env.engine.snapshot()
    state = replace(state, ball_pos=np.array([[0., 0.]]),
                    player_pos=np.array([[[-100., 0.], [100., 0.]]]),
                    player_vel=np.zeros((1, 2, 2)), scored=np.array([-1]))
    reward = build_reward("simple")
    def value(vx=0., pv=0., scored=-1):
        st = replace(state, ball_vel=np.array([[vx, 0.]]),
                     player_vel=np.array([[[pv, 0.], [0., 0.]]]), scored=np.array([scored]))
        return reward.get_rewards(st, state, np.array([scored != -1]), np.array([False]))[0].tolist()
    forward, backward = value(3.), value(-3.)
    assert forward[0] > 0 and backward[0] < 0
    assert np.isclose(forward[0] + backward[0], 0)
    assert value(scored=BLUE) == [25., -25.]
    assert value(scored=RED) == [-25., 25.]
    assert value(pv=1.)[0] > 0 and value(pv=-1.)[0] < 0
    # Actual engine trajectories: equal initial kick, defender on/off the shot line.
    shots = {}
    for name, defender_y in (("blocked", 0.), ("open", 150.)):
        env.reset()
        env.engine.set_state(ball_pos=np.array([[0., 0.]]), ball_vel=np.zeros((1, 2)),
            player_pos=np.array([[[-24., 0.], [80., defender_y]]]), player_vel=np.zeros((1, 2, 2)))
        env.prev_state = env.engine.snapshot()
        progress, goals, vx = 0., 0., []
        for step in range(100):
            act = np.array([[[1, 1, int(step == 0)], [1, 1, 1]]])
            previous = env.prev_state
            # Capture terminal state inside reward callback, before autoreset.
            captured = []
            original = env.reward_fn
            class Capture:
                def get_rewards(self, st, prev, term, trunc):
                    captured.append(st)
                    return original.get_rewards(st, prev, term, trunc)
            env.reward_fn = Capture()
            _, _, term, trunc = env.step(act)
            env.reward_fn = original
            st = captured[0]
            progress += float(original.terms[1][0].get_rewards(st, previous, term, trunc)[0, 0]) * original.terms[1][1]
            vx.append(float(st.ball_vel[0, 0]))
            goals += int(st.scored[0] == BLUE)
            if term[0] or trunc[0]:
                break
        shots[name] = dict(progress_sum=progress, red_goals=goals, min_vx=min(vx), max_vx=max(vx), decisions=len(vx))
    assert shots["open"]["max_vx"] > 0
    assert shots["blocked"]["min_vx"] < 0
    assert shots["open"]["progress_sum"] > shots["blocked"]["progress_sum"]
    return dict(player_max_speed=float(state.player_max_speed), forward=forward, backward=backward, undiscounted_roundtrip=forward[0]+backward[0],
                scored=value(scored=BLUE), conceded=value(scored=RED), engine_shots=shots)


@torch.no_grad()
def evaluate(model, seeds, n, steps, stadium=STADIUM):
    rows = []
    for seed in seeds:
        for opponent in ("chase", "chase_no_noise", "random"):
            for side in (0, 1):
                seed_all(seed)
                env = create_eval_env(n, 1, 1, stadium, 8)
                obs = env.reset()
                bot = RandomBot(seed) if opponent == "random" else ChaseBot(seed=seed, jiggle=0. if opponent == "chase_no_noise" else .25)
                gf = ga = 0
                for _ in range(steps):
                    actions = np.empty((n, 2, 3), dtype=np.int64)
                    actions[:, side] = model.act(torch.as_tensor(obs[:, side], dtype=torch.float32), greedy=True).numpy()
                    other = 1-side
                    actions[:, other] = (bot.act(obs[:, other]) if opponent == "random" else bot.act(env.prev_state, [other])).reshape(n, 3)
                    obs, _, _, _ = env.step(actions)
                    scored = env.prev_state.scored
                    gf += int((scored == (BLUE if side == 0 else RED)).sum())
                    ga += int((scored == (RED if side == 0 else BLUE)).sum())
                rows.append(dict(seed=seed, opponent=opponent, side=side, gf=gf, ga=ga, net=gf-ga))
    return rows


def train(table, seed, args, initial):
    seed_all(seed)
    env = create_training_env(args.envs, 1, 1, args.stadium, 8, seed=seed, table=table)
    env.reward_fn = build_reward(table, args.stadium, 8)
    model = Policy(env.obs_dim, env.action_space)
    model.load_state_dict(initial)
    if getattr(args, "reset_value", False):
        torch.nn.init.zeros_(model.value.weight)
        torch.nn.init.zeros_(model.value.bias)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    chase = ChaseBot(seed=seed)
    obs = env.reset()
    gamma = math.exp(math.log(.5) / (7.5 * HALF_LIFE_S))
    totals = {type(fn).__name__: 0. for fn, _ in env.reward_fn.terms}
    # Capture contributions before automatic episode resets.
    original = env.reward_fn
    class Meter:
        def get_rewards(self, st, prev, term, trunc):
            result = np.zeros((args.envs, 2), dtype=np.float32)
            for fn, weight in original.terms:
                part = fn.get_rewards(st, prev, term, trunc) * weight
                totals[type(fn).__name__] += float(part[:, 0].sum())
                result += part
            return result
    env.reward_fn = Meter()
    for it in range(args.iters):
        frac = it / max(1, args.iters-1)
        for group in opt.param_groups:
            group['lr'] = LR*(1-frac) + LR/3*frac
        ent_coef = ENT0*(1-frac) + ENT1*frac
        so, sa, sl, sv, sr, sd = [], [], [], [], [], []
        for _ in range(T):
            o = torch.as_tensor(obs[:, 0], dtype=torch.float32)
            with torch.no_grad():
                a, logp, v = model.act_train(o)
            actions = np.empty((args.envs, 2, 3), dtype=np.int64)
            actions[:, 0] = a.numpy()
            actions[:, 1] = chase.act(env.prev_state, [1]).reshape(args.envs, 3)
            obs, rew, term, trunc = env.step(actions)
            so.append(o); sa.append(a); sl.append(logp); sv.append(v)
            sr.append(torch.as_tensor(rew[:, 0])); sd.append(torch.as_tensor(term | trunc))
        so, sa, sl, sv, sr, sd = map(torch.stack, (so, sa, sl, sv, sr, sd))
        with torch.no_grad():
            nextv = model._logits(torch.as_tensor(obs[:, 0], dtype=torch.float32))[1]
        adv = torch.zeros_like(sr); gae = torch.zeros(args.envs)
        for t in reversed(range(T)):
            mask = (~sd[t]).float()
            delta = sr[t] + gamma*nextv*mask - sv[t]
            gae = delta + gamma*LAM*mask*gae
            adv[t] = gae; nextv = sv[t]
        ret = (adv+sv).flatten()
        adv = adv.flatten(); adv = (adv-adv.mean())/(adv.std()+1e-8)
        bo, ba, bl = so.reshape(-1, env.obs_dim), sa.reshape(-1, 3), sl.flatten()
        for _ in range(EPOCHS):
            for mi in torch.randperm(T*args.envs).chunk(N_MB):
                logp, entropy, val = model.evaluate(bo[mi], ba[mi])
                ratio = (logp-bl[mi]).exp()
                loss = -torch.minimum(ratio*adv[mi], ratio.clamp(1-CLIP, 1+CLIP)*adv[mi]).mean()
                loss += VF_COEF*.5*(val-ret[mi]).square().mean() - ent_coef*entropy.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD); opt.step()
        if (it+1) % 10 == 0:
            print(f'{table} seed={seed} iteration={it+1}/{args.iters}', flush=True)
    torch.save(model.state_dict(), args.out / f'{table}_seed{seed}.pt')
    return dict(table=table, seed=seed, contributions=totals,
                evaluation=evaluate(model, args.eval_seeds, args.eval_envs, args.eval_steps, args.stadium))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--envs', type=int, default=64)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--eval-seeds', type=int, nargs='+', default=[10, 11])
    ap.add_argument('--eval-envs', type=int, default=16)
    ap.add_argument('--eval-steps', type=int, default=300)
    ap.add_argument('--warm-start', type=Path, default=Path('models/model.pt'))
    ap.add_argument('--tables', nargs='+', choices=['previous', 'simple_progress1', 'simple', 'corrected'], default=['previous', 'simple'])
    ap.add_argument('--stadium', default=STADIUM)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.out.exists():
        raise SystemExit('Output already exists; choose a new directory.')
    if not args.warm_start.is_file():
        raise SystemExit('Warm-start checkpoint does not exist.')
    initial = torch.load(args.warm_start, map_location='cpu', weights_only=True)
    args.out.mkdir(parents=True)
    torch.save(initial, args.out / "initial.pt")
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  reward_tables={table: {type(fn).__name__: weight for fn, weight in build_reward(table).terms} for table in args.tables},
                  checkpoint_sha256=hashlib.sha256(args.warm_start.read_bytes()).hexdigest(),
                  controlled=controlled_checks(),
                  limitation='Short fine-tuning against Chase, not self-play convergence; greedy evaluation. Goals are aggregated over fixed windows, not match win rates.', runs=[])
    baseline = Policy(16); baseline.load_state_dict(initial)
    report['baseline'] = evaluate(baseline, args.eval_seeds, args.eval_envs, args.eval_steps, args.stadium)
    start = time.perf_counter()
    for seed in args.seeds:
        for table in args.tables:
            report['runs'].append(train(table, seed, args, initial))
            report['elapsed_s'] = time.perf_counter()-start
            (args.out/'results.json').write_text(json.dumps(report, indent=2))
    print(f'Results: {args.out}/results.json', flush=True)

if __name__ == '__main__':
    main()

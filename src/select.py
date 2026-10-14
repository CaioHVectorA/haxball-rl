"""Pick the strongest checkpoint from a training run via Round-Robin tournament.

    uv run src/select.py            # round-robin models/ckpts/*.pt (+ model_final.pt), winner -> model.pt

Plays every saved checkpoint against every other (head-to-head, both colors; goals counted
from `state.scored`, so it's reward-independent) and copies the one with the best overall
goal share to `models/model.pt` (+ `meta.json`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from haxballgym import BLUE, RED
from train import create_eval_env
from environment import STADIUM
from model import Policy

MODELS = Path(__file__).resolve().parent.parent / "models"


def load(path: Path):
    """Rebuild a Policy from a checkpoint, inferring the architecture from the weights."""
    sd = torch.load(path, map_location="cpu")
    obs_dim = sd["trunk.0.weight"].shape[1]
    hidden = sd["trunk.0.weight"].shape[0]
    depth = sum(1 for k in sd if k.startswith("trunk.") and k.endswith(".weight"))
    nk = sd["head_k.weight"].shape[0]
    m = Policy.from_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def h2h(a, b, stadium: str | None = None, n_envs: int = 128, steps: int = 350):
    """Goals (a, b) with `a`=red, `b`=blue over one set of games."""
    env = create_eval_env(n_envs, 1, 1, stadium or STADIUM, 8)
    obs = env.reset()
    od = obs.shape[-1]
    ag = bg = 0
    for _ in range(steps):
        ra = a.act(torch.as_tensor(obs[:, 0:1].reshape(-1, od), dtype=torch.float32)).numpy()
        rb = b.act(torch.as_tensor(obs[:, 1:2].reshape(-1, od), dtype=torch.float32)).numpy()
        obs, rew, term, trunc = env.step(np.stack([ra, rb], axis=1).reshape(n_envs, 2, 3))
        sc = env.prev_state.scored
        ag += int((sc == BLUE).sum())  # blue conceded -> a (red) scored
        bg += int((sc == RED).sum())
    return ag, bg


def main():
    meta_path = MODELS / "meta.json"
    default_stadium = STADIUM
    if meta_path.exists():
        try:
            default_stadium = json.loads(meta_path.read_text()).get("stadium", default_stadium)
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--stadium", default=default_stadium, help="stadium name or .hbs path")
    ap.add_argument("--envs", type=int, default=128, help="parallel tournament games")
    ap.add_argument("--steps", type=int, default=300, help="steps per match")
    args = ap.parse_args()

    paths = sorted((MODELS / "ckpts").glob("it*.pt"), key=lambda p: int(p.stem[2:]))
    if (MODELS / "model_final.pt").exists() and (MODELS / "model_final.pt") not in paths:
        paths.append(MODELS / "model_final.pt")

    if len(paths) < 2:
        print("need >=2 checkpoints in models/ckpts/ (train with --save-every)")
        return

    print(f"\n{'='*70}")
    print(f"TOURNAMENT SELECTION — {len(paths)} Checkpoints on {args.stadium}")
    print(f"{'='*70}\n")

    nets = {p.stem: load(p) for p in paths}
    gf = dict.fromkeys(nets, 0)
    ga = dict.fromkeys(nets, 0)
    names = list(nets)

    # Round-robin, each pair once with color swap
    total_pairs = len(names) * (len(names) - 1) // 2
    pair_count = 0
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            pair_count += 1
            xr, yb = h2h(nets[x], nets[y], stadium=args.stadium, n_envs=args.envs, steps=args.steps)
            yr, xb = h2h(nets[y], nets[x], stadium=args.stadium, n_envs=args.envs, steps=args.steps)
            gf[x] += xr + xb
            ga[x] += yb + yr
            gf[y] += yr + yb
            ga[y] += xb + xr

    ranked = sorted(names, key=lambda k: (gf[k] / max(gf[k] + ga[k], 1), gf[k] - ga[k]), reverse=True)
    print(f"{'RANK':<5} {'CHECKPOINT':<18} {'GOAL SHARE':<12} {'GOLS PRO':<10} {'GOLS CONTRA':<12} {'SALDO':<8}")
    print("-" * 70)
    for rank, k in enumerate(ranked, 1):
        share = gf[k] / max(gf[k] + ga[k], 1)
        net_diff = gf[k] - ga[k]
        print(f"{rank:<5} {k:<18} {share:<12.3f} {gf[k]:<10} {ga[k]:<12} {net_diff:+<8d}")
    print("-" * 70)

    win = ranked[0]
    torch.save(nets[win].state_dict(), MODELS / "model.pt")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta.update(obs_dim=nets[win].trunk[0].in_features, hidden=nets[win].trunk[0].out_features,
                depth=len(nets[win].trunk)//2, n_acts=list(nets[win].n_acts),
                n_red=1, n_blue=1, stadium=args.stadium, repeat_kick=True)
    meta.pop("it", None)
    meta.pop("net_vs_chase", None)
    meta["selected"] = win
    (MODELS / "meta.json").write_text(json.dumps(meta))
    print(f"\n🏆 Campeão Absoluto: [{win}] -> copiado para models/model.pt e models/meta.json\n")


if __name__ == "__main__":
    main()

"""Headless evaluation and benchmarking across multiple bots and stadiums.

    uv run src/eval.py                              # Current NeuralBot vs ChaseBot
    uv run src/eval.py --benchmark                  # Comprehensive benchmark: Random vs Chase, Neural vs Chase, Neural vs Random
    uv run src/eval.py --stadium futsal-classic     # Custom stadium evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bot import ChaseBot, NeuralBot, RandomBot
from haxballgym import BLUE, RED
from train import create_eval_env
from environment import STADIUM


def play_matchup(red_bot, blue_bot, stadium: str | None, n_envs: int, steps: int, team_size: int = 1):
    env = create_eval_env(n_envs, team_size, team_size, stadium or STADIUM, 8)
    obs = env.reset()
    rg = bg = 0
    captured = []
    reward = env.reward_fn
    class Capture:
        def get_rewards(self, state, prev, term, trunc):
            captured.append(state)
            return reward.get_rewards(state, prev, term, trunc)
    env.reward_fn = Capture()
    for _ in range(steps):
        actions=[]
        for bot, players in [(red_bot, list(range(team_size))),
                             (blue_bot, list(range(team_size, 2*team_size)))]:
            action = (bot.act(env.prev_state, players, n_red=team_size) if isinstance(bot, ChaseBot)
                      else bot.act(obs[:, players]))
            actions.append(action.reshape(n_envs, team_size, 3))
        obs, _, _, _ = env.step(np.concatenate(actions, axis=1))
        scored=captured.pop().scored
        rg += int((scored == BLUE).sum())
        bg += int((scored == RED).sum())
    return rg, bg


def main():
    models_dir = Path(__file__).resolve().parent.parent / "models"
    meta_path = models_dir / "meta.json"
    default_stadium = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            default_stadium = meta.get("stadium")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="parallel games")
    ap.add_argument("--steps", type=int, default=600, help="decisions per game")
    ap.add_argument("--stadium", default=None, help="stadium name or .hbs path; default from model metadata")
    ap.add_argument("--benchmark", action="store_true", help="run full 3-way comparative benchmark")
    ap.add_argument("--self", action="store_true", help="NeuralBot vs NeuralBot")
    ap.add_argument('--model-dir', type=Path, default=models_dir)
    ap.add_argument('--team-size', type=int, choices=(1,2))
    args = ap.parse_args()
    meta=json.loads((args.model_dir/'meta.json').read_text())
    team_size=args.team_size or meta.get('red',meta.get('n_red',1))

    stadium_name = args.stadium or meta.get("stadium") or STADIUM

    if args.benchmark:
        print(f"\n{'='*65}")
        print(f"HAXBALL BENCHMARK — Arena: {stadium_name} ({args.n} games x {args.steps} steps)")
        print(f"{'='*65}\n")

        matchups = [
            ("RandomBot", RandomBot(seed=42), "ChaseBot", ChaseBot(seed=0)),
            ("NeuralBot (Trained)", NeuralBot(model_dir=args.model_dir,greedy=True), "ChaseBot", ChaseBot(seed=0)),
            ("NeuralBot (Trained)", NeuralBot(model_dir=args.model_dir,greedy=True), "RandomBot", RandomBot(seed=123)),
        ]

        for r_name, r_bot, b_name, b_bot in matchups:
            rg, bg = play_matchup(r_bot, b_bot, stadium=stadium_name, n_envs=args.n, steps=args.steps, team_size=team_size)
            net = rg - bg
            total = rg + bg
            winrate = (rg / total * 100) if total > 0 else 50.0
            print(f"[{r_name}] vs [{b_name}]:")
            print(f"  Gols: {rg} - {bg} (Saldo: {net:+d}) | Proporção dos gols: {winrate:.1f}%\n")
        print(f"{'='*65}\n")
    else:
        red_bot = NeuralBot(model_dir=args.model_dir,greedy=True)
        blue_bot = NeuralBot(model_dir=args.model_dir,greedy=True) if args.self else ChaseBot(seed=0)
        b_name = "NeuralBot (Self)" if args.self else "ChaseBot"
        rg, bg = play_matchup(red_bot, blue_bot, stadium=stadium_name, n_envs=args.n, steps=args.steps, team_size=team_size)
        print(f"NeuralBot {rg} - {bg} {b_name} | Saldo: {rg - bg:+d} (em {args.n} jogos x {args.steps} steps no {stadium_name})")


if __name__ == "__main__":
    main()

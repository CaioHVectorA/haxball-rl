"""Evaluate trained 1v1 policies in fixed wall-corner attacks against a still opponent."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from environment import STADIUM
from model import Policy
from train import create_eval_env
from haxballgym import BLUE, RED


def evaluate(checkpoint: Path, *, envs: int = 8, max_steps: int = 350):
    model = Policy.from_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    model.eval()
    rows = []
    for side in (0, 1):
        for wall in (-1, 1):
            env = create_eval_env(envs, 1, 1, STADIUM, 8)
            env.reset()
            initial = env.engine.snapshot()
            goal_x = float(initial.goal_center(attacked=True)[0, side, 0])
            attack_dir = 1.0 if goal_x > 0 else -1.0
            offsets = np.linspace(-12., 12., envs)
            # Approximate the supplied match image in map coordinates: the ball is
            # within 10 units of the lower wall, with the bot above-left and the
            # stationary player well behind it. Mirror for the other sides/wall.
            ball = np.tile([attack_dir * 304., wall * 192.], (envs, 1))
            ball_vel = np.zeros((envs, 2))
            players = np.zeros((envs, 2, 2))
            players[:, side, 0] = attack_dir * 284. + offsets
            players[:, side, 1] = wall * 146.
            players[:, 1-side, 0] = attack_dir * 115.
            players[:, 1-side, 1] = wall * 115.
            velocities = np.zeros((envs, 2, 2))
            env.prev_state = env.engine.set_state(ball, ball_vel, players, velocities)
            obs = env.obs_builder.build_obs(env.prev_state)
            capture = []
            original_reward = env.reward_fn

            class CaptureState:
                def get_rewards(self, state, previous, terminated, truncated):
                    capture.append(state)
                    return original_reward.get_rewards(state, previous, terminated, truncated)

            env.reward_fn = CaptureState()
            done = np.zeros(envs, dtype=bool)
            first_contact = np.full(envs, -1, dtype=int)
            near_steps = np.zeros(envs, dtype=int)
            near_presses = np.zeros(envs, dtype=int)
            aimed_impulses = np.zeros(envs, dtype=int)
            goals_for = np.zeros(envs, dtype=int)
            goals_against = np.zeros(envs, dtype=int)
            ball_goal_delta = np.zeros(envs, dtype=float)
            decisions = np.zeros(envs, dtype=int)
            for step in range(max_steps):
                previous = env.prev_state
                action = model.act(torch.as_tensor(obs[:, side], dtype=torch.float32), greedy=True).numpy()
                action[done] = [1, 1, 0]
                delta = previous.ball_pos - previous.player_pos[:, side]
                distance = np.linalg.norm(delta, axis=-1)
                near = (distance < 24.8) & ~done
                new_contact = near & (first_contact < 0)
                first_contact[new_contact] = step
                near_steps += near.astype(int)
                pressed = (action[:, 2] == 1) & near
                near_presses += pressed.astype(int)
                opponent_action = np.zeros((envs, 3), dtype=np.int64)
                opponent_action[:, 0:2] = 1
                actions = np.empty((envs, 2, 3), dtype=np.int64)
                actions[:, side] = action
                actions[:, 1-side] = opponent_action
                obs, _, terminated, truncated = env.step(actions)
                post = capture.pop()
                active = ~done
                goals_for += ((post.scored == (BLUE if side == 0 else RED)) & active).astype(int)
                goals_against += ((post.scored == (RED if side == 0 else BLUE)) & active).astype(int)
                to_goal = previous.goal_center(attacked=True)[:, side] - previous.ball_pos
                to_goal /= np.maximum(np.linalg.norm(to_goal, axis=-1, keepdims=True), 1e-9)
                impulse = post.ball_vel - previous.ball_vel
                toward = np.sum(impulse * to_goal, axis=-1) > .2
                aimed_impulses += (pressed & toward).astype(int)
                ball_goal_delta += (np.linalg.norm(previous.goal_center(attacked=True)[:, side] - previous.ball_pos, axis=-1)
                    - np.linalg.norm(post.goal_center(attacked=True)[:, side] - post.ball_pos, axis=-1)) * active
                newly_done = (terminated | truncated) & active
                decisions[active] += 1
                done |= newly_done
                if done.all():
                    break
            rows.append({
                "side": side,
                "wall": "bottom" if wall > 0 else "top",
                "start_ball": [float(ball[0, 0]), float(ball[0, 1])],
                "start_player": [float(players[0, side, 0]), float(players[0, side, 1])],
                "episodes": int(envs),
                "completed": int(done.sum()),
                "goals_for": int(goals_for.sum()),
                "goals_against": int(goals_against.sum()),
                "contact_fraction": float((first_contact >= 0).mean()),
                "mean_first_contact_decisions": float(first_contact[first_contact >= 0].mean()) if (first_contact >= 0).any() else None,
                "near_contact_decisions": int(near_steps.sum()),
                "near_contact_presses": int(near_presses.sum()),
                "press_with_goalward_impulse": int(aimed_impulses.sum()),
                "mean_center_goal_distance_reduction": float(ball_goal_delta.sum() / max(1, decisions.sum())),
                "mean_episode_decisions": float(decisions.mean()),
            })
    return {"checkpoint": str(checkpoint), "envs_per_case": envs, "max_steps": max_steps,
            "opponent": "stationary (no movement, no kick)", "stadium": STADIUM, "rows": rows,
            "limitation": "Fixed scripted states test this corner pattern only; a goalward impulse is not proof that the ball entered the goal."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=350)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = []
    for item in args.policy:
        name, path = item.split("=", 1)
        result = evaluate(Path(path), envs=args.envs, max_steps=args.steps)
        result["name"] = name
        results.append(result)
        print(json.dumps({"name": name, "rows": result["rows"]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()

# Haxball RL ⚽🤖

Autonomous Reinforcement Learning agents for [Haxball](https://www.haxball.com) trained via PPO self-play and deployed straight into live multiplayer rooms.

---

## ⚡ Architecture Overview

1. **Analytical Engine**: Powered by [HaxballGym](https://github.com/HaxballGym/HaxballGym) — Rust simulation core delivering bit-exact ($10^{-9}$) analytical collision physics identical to the official Haxball game.
2. **High-Throughput PPO**: Runs 512 parallel game instances using Rayon multithreading, reaching up to **50,000 steps/sec** on CPU.
3. **Multi-Agent Self-Play**: Self-play training with snapshot pools, goal-dominated rewards, and symmetric observation normalization.
4. **Zero-Lag Headless Bot Integration**: Policies export directly to lightweight JSON (`policy.json`) for seamless real-time execution in official Haxball rooms via the Headless Room API.

---

## 🚀 Quick Start

### 1. Requirements & Setup

This repository uses [uv](https://github.com/astral-sh/uv) for fast and reproducible Python package management.

```bash
# Clone the repository
git clone https://github.com/CaioHVectorA/haxball-rl.git
cd haxball-rl

# Sync Python virtual environment
uv sync
```

### 2. Training an Agent

Train a 1v1 PPO policy on the futsal stadium across 512 parallel environments:

```bash
uv run src/train.py --stadium futsal-classic --red 1 --blue 1 --envs 512 --iters 4000
```

Checkpoints are automatically saved to `models/`.

### 3. Deploying to Live Haxball Rooms

You can deploy your trained policy directly against real human players:

```bash
cd headless-bot
npm install

# Export the latest PyTorch checkpoint to JSON policy
uv run export_policy.py

# Launch the official Headless Room
HEADLESS_TOKEN="<your_token_from_haxball_com>" node bot.js
```

Obtain your room token at [haxball.com/headlesstoken](https://www.haxball.com/headlesstoken). Join the room via the printed URL and play 1v1 against the RL agent!

---

## 📂 Repository Structure

- `src/` — PPO training loop, neural network architecture, evaluation suites, and self-play management.
- `headless-bot/` — Node.js loader and export pipeline for live room deployment.
- `maps/` — Standard and custom Haxball stadium definitions (`.hbs`).
- `models/` — Saved model weights and metadata.
- `CONTEXT.md` / `PROTOCOL.md` / `AGENTS.md` — In-depth architectural details, protocol specs, and agent roles.

---

## 📜 License
MIT

import sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
import numpy as np
from train import create_eval_env
from model import Policy

dev = torch.device("cpu")
model = Policy(16, (3, 3, 2), 256, 2).to(dev)
model.load_state_dict(torch.load("models/model_final.pt", map_location=dev))
model.eval()

# Run 32 games of it4350 vs it4350 (Self-Play puro)
env = create_eval_env(32, 1, 1, "futsal-classic", 8)
obs = env.reset()

goals = [0, 0]
steps = 800

for _ in range(steps):
    flat_obs = obs.reshape(-1, 16)
    with torch.no_grad():
        acts = model.act(torch.tensor(flat_obs, dtype=torch.float32, device=dev), greedy=True).numpy()
    obs, rews, terms, truncs = env.step(acts.reshape(32, 2, 3))
    st = env.engine.snapshot()
    for s in st.scored:
        if s == 2:
            goals[0] += 1
        elif s == 1:
            goals[1] += 1

print(f"Self-Play (it4350 vs it4350 em 32 jogos de 800 passos): Gols Red {goals[0]} x Gols Blue {goals[1]} | Total Gols: {sum(goals)}")

import sys
sys.path.insert(0, "src")
import torch
import numpy as np
from train import create_eval_env
from model import Policy
from bot import ChaseBot

dev = torch.device("cpu")
model = Policy(16, (3, 3, 2), 256, 2).to(dev)
model.load_state_dict(torch.load("models/model.pt", map_location=dev))
model.eval()

# Run 32 games of it300 vs ChaseBot and check how many goals Red scored
env = create_eval_env(32, 1, 1, "futsal-classic", 8)
obs = env.reset()
chase = ChaseBot(seed=123)

rg = bg = 0
for _ in range(350):
    o_red = obs[:, 0].reshape(-1, 16)
    with torch.no_grad():
        a_red = model.act(torch.tensor(o_red, dtype=torch.float32, device=dev), greedy=True).numpy()
    st = env.engine.snapshot()
    a_blue = chase.act(st, [1], n_red=1)
    joint = np.concatenate([a_red[:, None, :], a_blue], axis=1)
    obs, _, _, _ = env.step(joint)
    sc = env.engine.snapshot().scored
    rg += int((sc == 2).sum())
    bg += int((sc == 1).sum())

print(f"Resultado it300 vs ChaseBot (32 partidas simultâneas):")
print(f"Gols RL (Red): {rg} | Gols Chase (Blue): {bg} | Saldo: {rg - bg}")

import sys
sys.path.insert(0, "src")
import torch
import numpy as np
from train import create_eval_env
from model import Policy
from bot import ChaseBot

dev = torch.device("cpu")
model = Policy(16, (3, 3, 2), 256, 2).to(dev)
model.load_state_dict(torch.load("models/model_final.pt", map_location=dev))
model.eval()

# Let's run 64 parallel matches
env = create_eval_env(64, 1, 1, "futsal-classic", 8)
obs = env.reset()
chase = ChaseBot(seed=42)

rg = bg = 0
touches = 0
kicks_fired = 0

for _ in range(400):
    o_red = obs[:, 0].reshape(-1, 16)
    with torch.no_grad():
        a_red = model.act(torch.tensor(o_red, dtype=torch.float32, device=dev), greedy=False).numpy()
    st = env.engine.snapshot()
    a_blue = chase.act(st, [1], n_red=1)
    joint = np.concatenate([a_red[:, None, :], a_blue], axis=1)
    obs, _, _, _ = env.step(joint)
    sc = env.engine.snapshot().scored
    rg += int((sc == 2).sum())
    bg += int((sc == 1).sum())
    
    for i in range(64):
        d = np.hypot(st.player_pos[i, 0, 0] - st.ball_pos[i, 0], st.player_pos[i, 0, 1] - st.ball_pos[i, 1])
        if d < 35.0:
            touches += 1
            if a_red[i, 2] == 1:
                kicks_fired += 1

print(f"Em 64 partidas (it2100 vs ChaseBot):")
print(f"Gols RL (Red): {rg} | Gols Chase (Blue): {bg}")
print(f"Total Contatos: {touches} | Chutes em contato: {kicks_fired} ({kicks_fired/max(1, touches)*100:.1f}%)")

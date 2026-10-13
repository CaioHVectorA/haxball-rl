import sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
from train import create_eval_env
from model import Policy

dev = torch.device("cpu")

for path, label in [
    ("models/model.pt", "Melhor Atual (it1300 Novo Treino)"),
    ("models/ckpts/it1000.pt", "it1000 (Novo Treino)"),
    ("models/ckpts/it400.pt", "it400 (Antigo)"),
    ("models/ckpts/it1200.pt", "it1200 (Antigo)")
]:
    p = Path(path)
    if not p.exists():
        continue
    model = Policy(16, (3, 3, 2), 256, 2).to(dev)
    model.load_state_dict(torch.load(p, map_location=dev))
    model.eval()
    
    env = create_eval_env(64, 1, 1, "futsal-classic", 8)
    obs = env.reset()
    kicks = 0
    near_touches = 0
    
    with torch.no_grad():
        for _ in range(250):
            o_tensor = torch.tensor(obs.reshape(-1, 16), dtype=torch.float32, device=dev)
            acts = model.act(o_tensor, greedy=True).numpy()
            
            flat_obs = obs.reshape(-1, 16)
            for i in range(len(flat_obs)):
                dist = ((flat_obs[i][0] * 480 - flat_obs[i][4] * 480)**2 + (flat_obs[i][1] * 221 - flat_obs[i][5] * 221)**2)**0.5
                if dist < 36.0:
                    near_touches += 1
                    if acts[i][2] == 1:
                        kicks += 1
            obs, _, _, _ = env.step(acts.reshape(64, 2, 3))
            
    pct = (kicks / near_touches * 100) if near_touches > 0 else 0
    print(f"[{label}] Contatos: {near_touches} | Chutes: {kicks} | Taxa de Chute: {pct:.1f}%")

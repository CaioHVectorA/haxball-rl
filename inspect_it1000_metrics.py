import sys
sys.path.insert(0, "src")
import torch
import numpy as np
from model import Policy

dev = torch.device("cpu")
model = Policy(16, (3, 3, 2), 256, 2).to(dev)
model.load_state_dict(torch.load("models/ckpts/it1000.pt", map_location=dev))
model.eval()

# Let's inspect kick logits and avoidance of opponent
print("=== Telemetria do it1000 (Novo Ciclo) ===")
for dist_x in [20, 25, 30, 35]:
    o = torch.zeros(1, 16)
    o[0, 4] = dist_x / 480.0
    with torch.no_grad():
        (lx, ly, lk), _ = model._logits(o)
        pk = torch.softmax(lk, -1)[0, 1].item() * 100
        print(f"Bola a {dist_x}px à frente -> Probabilidade de Chute: {pk:.4f}%")

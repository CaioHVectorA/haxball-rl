import sys
sys.path.insert(0, "src")
import torch
import numpy as np
from model import Policy

dev = torch.device("cpu")
model = Policy(16, (3, 3, 2), 256, 2).to(dev)
model.load_state_dict(torch.load("models/model_final.pt", map_location=dev))
model.eval()

print("Telemetria do Modelo Atual (it2100):")
for dist_x in [15, 20, 25, 30, 40]:
    o = torch.zeros(1, 16)
    o[0, 4] = dist_x / 480.0
    with torch.no_grad():
        (lx, ly, lk), _ = model._logits(o)
        pk = torch.softmax(lk, -1)[0, 1].item() * 100
        print(f"Distância {dist_x}px -> Probabilidade Chute: {pk:.4f}% | Logits: {lk[0].numpy()}")

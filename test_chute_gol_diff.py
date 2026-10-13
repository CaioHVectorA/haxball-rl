import sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
import numpy as np
import haxballgym
from model import Policy

model = Policy(16, (3, 3, 2), 256, 2)
model.load_state_dict(torch.load("models/model_final.pt", map_location="cpu"))
model.eval()

# Let's see logits for different distances to ball
# Obs: [px, py, vx, vy, bx, by, bvx, bvy, ...]
# In policy frame, +X is forward towards opponent goal!
for dist_x in [15, 20, 25, 30, 40, 60]:
    o = torch.zeros(1, 16)
    o[0, 0] = 0.0 # player at (0, 0)
    o[0, 4] = dist_x / 480.0 # ball directly ahead
    with torch.no_grad():
        (lx, ly, lk), v = model._logits(o)
        probs_k = torch.softmax(lk, -1)[0].numpy()
        print(f"Distância {dist_x:2d}px à frente -> Logits Kick: {lk[0].numpy()} | Prob Kick=1: {probs_k[1]*100:.3f}%")

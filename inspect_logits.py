import sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
from model import Policy

model = Policy(16, (3, 3, 2), 256, 2)
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()

# Dummy obs: player at (0, 0), ball at (10/480, 0) - directly touching in front!
obs = torch.zeros(1, 16)
obs[0, 0] = 0.0 # player x
obs[0, 1] = 0.0 # player y
obs[0, 4] = 20.0 / 480 # ball x
obs[0, 5] = 0.0 # ball y

with torch.no_grad():
    (lx, ly, lk), v = model._logits(obs)
    print("Logits X:", lx)
    print("Logits Y:", ly)
    print("Logits Kick:", lk)
    print("Kick Probabilities:", torch.softmax(lk, -1))

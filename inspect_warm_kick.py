import sys
sys.path.insert(0, "src")
import torch
import numpy as np
from model import Policy

dev = torch.device("cpu")
model = Policy(16, (3, 3, 2), 256, 2).to(dev)

# Load warm-start baseline vs current live model.pt
ws = Policy(16, (3, 3, 2), 256, 2).to(dev)
ws.load_state_dict(torch.load("models/warm_start.pt", map_location=dev))
ws.eval()

curr = Policy(16, (3, 3, 2), 256, 2).to(dev)
curr.load_state_dict(torch.load("models/model.pt", map_location=dev))
curr.eval()

# Check kick probability at different distances
for d in [20, 25, 30, 35]:
    o = torch.zeros(1, 16)
    o[0, 4] = d / 480.0 # ball ahead
    with torch.no_grad():
        _, _, lk_ws = ws._logits(o)[0]
        _, _, lk_curr = curr._logits(o)[0]
        p_ws = torch.softmax(lk_ws, -1)[0, 1].item() * 100
        p_curr = torch.softmax(lk_curr, -1)[0, 1].item() * 100
        print(f"Dist {d}px -> Prob Chute Antes: {p_ws:.4f}% | Prob Chute Agora (it350): {p_curr:.4f}%")

import sys
sys.path.insert(0, "src")
import torch
import numpy as np
from model import Policy

dev = torch.device("cpu")
ws = Policy(16, (3, 3, 2), 256, 2).to(dev)
ws.load_state_dict(torch.load("models/warm_start.pt", map_location=dev))
ws.eval()

m21 = Policy(16, (3, 3, 2), 256, 2).to(dev)
m21.load_state_dict(torch.load("models/model_final.pt", map_location=dev))
m21.eval()

# Check lateral movement when ball is slightly off-center
# Player at (0, 0), Ball at (25px ahead, 15px up)
o = torch.zeros(1, 16)
o[0, 4] = 25.0 / 480.0
o[0, 5] = 15.0 / 221.0

with torch.no_grad():
    (lx_ws, ly_ws, _), _ = ws._logits(o)
    (lx_21, ly_21, _), _ = m21._logits(o)
    print("Previsão com bola angulada (Finta/Ajuste):")
    print(f"Warm Start Antigo -> Prob Y Up: {torch.softmax(ly_ws, -1)[0, 0].item()*100:.1f}% | Down: {torch.softmax(ly_ws, -1)[0, 2].item()*100:.1f}%")
    print(f"Modelo it2100     -> Prob Y Up: {torch.softmax(ly_21, -1)[0, 0].item()*100:.1f}% | Down: {torch.softmax(ly_21, -1)[0, 2].item()*100:.1f}%")

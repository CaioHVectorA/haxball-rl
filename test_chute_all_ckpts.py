import sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
from model import Policy

model = Policy(16, (3, 3, 2), 256, 2)

for it in [1000, 2000, 3000, 4000]:
    p = Path(f"models/ckpts/it{it}.pt")
    if not p.exists(): continue
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.eval()
    
    o = torch.zeros(1, 16)
    o[0, 4] = 25.0 / 480.0 # ball ahead
    with torch.no_grad():
        (lx, ly, lk), _ = model._logits(o)
        probs_k = torch.softmax(lk, -1)[0].numpy()
        print(f"ckpt it{it}: Prob Kick na bola = {probs_k[1]*100:.4f}%")

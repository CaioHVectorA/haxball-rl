import sys
sys.path.insert(0, "src")
import torch
import numpy as np
from model import Policy

dev = torch.device("cpu")
ws = Policy(16, (3, 3, 2), 256, 2).to(dev)
ws.load_state_dict(torch.load("models/warm_start.pt", map_location=dev))
ws.eval()

curr = Policy(16, (3, 3, 2), 256, 2).to(dev)
curr.load_state_dict(torch.load("models/model.pt", map_location=dev))
curr.eval()

# Cenário: Adversário está bloqueando o caminho direto para a bola!
# Jogador em (0, 0)
# Adversário em (20px à frente) -> (20/480, 0)
# Bola em (40px à frente, atrás do adversário) -> (40/480, 0)
# Se o bot continuar indo reto (dx=1, dy=0), ele bate de cabeça no adversário ("beijo").
# Se ele desviar/contornar, dy != 0!
o = torch.zeros(1, 16)
o[0, 0] = 0.0 # jogador
o[0, 4] = 40.0 / 480.0 # bola x
o[0, 5] = 0.0 # bola y
o[0, 12] = 20.0 / 480.0 # adversario x no meio
o[0, 13] = 0.0 # adversario y

with torch.no_grad():
    (lx_ws, ly_ws, _), _ = ws._logits(o)
    (lx_c, ly_c, _), _ = curr._logits(o)
    act_ws = ws.act(o, greedy=True)[0].numpy()
    act_c = curr.act(o, greedy=True)[0].numpy()
    
    print(f"Antes (warm_start): Ação [dx={act_ws[0]-1}, dy={act_ws[1]-1}] | Probabilidade dy!=0: {(1 - torch.softmax(ly_ws, -1)[0, 1].item())*100:.1f}%")
    print(f"Agora (it350):      Ação [dx={act_c[0]-1}, dy={act_c[1]-1}] | Probabilidade dy!=0: {(1 - torch.softmax(ly_c, -1)[0, 1].item())*100:.1f}%")

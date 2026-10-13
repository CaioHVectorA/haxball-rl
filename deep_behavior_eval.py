import sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
import numpy as np
import haxballgym
from train import create_eval_env
from model import Policy
from bot import ChaseBot

dev = torch.device("cpu")

# Load it4350 (or model_final)
model = Policy(16, (3, 3, 2), 256, 2).to(dev)
model.load_state_dict(torch.load("models/model_final.pt", map_location=dev))
model.eval()

# Let's run 64 parallel matches against ChaseBot for 1000 steps each
N_ENVS = 64
STEPS = 1000
env = create_eval_env(N_ENVS, 1, 1, "futsal-classic", 8)
obs = env.reset()
chase = ChaseBot(seed=42)

goals_red = 0
goals_blue = 0

# Metrics:
# 1. Possession / proximity to ball
# 2. Time spent in own half vs opponent half
# 3. Kicking behaviour near ball vs away from ball
# 4. Ball velocity produced on contact
touches = 0
kicks_on_touch = 0
kicks_away = 0
time_opp_half = 0
time_own_half = 0
ball_vel_on_touch = []

for step in range(STEPS):
    # Model is RED (player 0), ChaseBot is BLUE (player 1)
    red_obs = obs[:, 0].reshape(-1, 16)
    with torch.no_grad():
        red_acts = model.act(torch.tensor(red_obs, dtype=torch.float32, device=dev), greedy=True).numpy()
    
    # ChaseBot needs state
    st = env.engine.snapshot()
    blue_acts = chase.act(st, [1], n_red=1)
    
    joint_acts = np.concatenate([red_acts[:, None, :], blue_acts], axis=1)
    obs, rews, terms, truncs = env.step(joint_acts)
    
    # Analyze state
    # Goal check
    st_post = env.engine.snapshot()
    scored = st_post.scored
    for s in scored:
        if s == 2: # Blue conceded -> Red scored
            goals_red += 1
        elif s == 1: # Red conceded -> Blue scored
            goals_blue += 1
            
    # Spatial metrics for Red
    for i in range(N_ENVS):
        px, py = st_post.player_pos[i, 0]
        bx, by = st_post.ball_pos[i]
        bvx, bvy = st_post.ball_vel[i]
        dist = np.hypot(px - bx, py - by)
        kicked = red_acts[i, 2] == 1
        
        if px > 0: # opponent half (attack)
            time_opp_half += 1
        else:
            time_own_half += 1
            
        if dist < 35.0: # touching/in range
            touches += 1
            if kicked:
                kicks_on_touch += 1
            ball_vel_on_touch.append(np.hypot(bvx, bvy))
        else:
            if kicked:
                kicks_away += 1

print("="*50)
print(f"ANÁLISE DE COMPORTAMENTO — it4350 vs ChaseBot ({N_ENVS} jogos x {STEPS} passos)")
print("="*50)
print(f"Placar: Bot RL {goals_red} x {goals_blue} ChaseBot | Saldo: {goals_red - goals_blue}")
print(f"Taxa de Vitória/Empate em gols: {goals_red / max(1, goals_red + goals_blue) * 100:.1f}%")
print(f"Contatos com a bola: {touches}")
print(f"Chutes em contato com a bola: {kicks_on_touch} ({kicks_on_touch / max(1, touches) * 100:.2f}%)")
print(f"Chutes sem a bola: {kicks_away}")
print(f"Tempo no Ataque (+X): {time_opp_half / (time_opp_half + time_own_half) * 100:.1f}%")
print(f"Tempo na Defesa (-X): {time_own_half / (time_opp_half + time_own_half) * 100:.1f}%")
if ball_vel_on_touch:
    print(f"Velocidade média da bola ao tocar: {np.mean(ball_vel_on_touch):.2f} (Máx: {np.max(ball_vel_on_touch):.2f})")
print("="*50)

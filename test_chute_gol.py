import sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
import numpy as np
import haxballgym
from model import Policy

dev = torch.device("cpu")
model = Policy(16, (3, 3, 2), 256, 2).to(dev)
model.load_state_dict(torch.load("models/model_final.pt", map_location=dev))
model.eval()

# Let's test a scenario: player is right in front of the opponent goal with the ball!
# Red goal is at -408, Blue goal is at +408
# Player at (250, 0), Ball at (275, 0)
env = haxballgym.make_default_env(1, 1, 1, stadium="futsal-classic", tick_skip=8)
obs = env.reset()

env.engine.set_state(
    ball_pos=np.array([[275.0, 0.0]]),
    ball_vel=np.array([[0.0, 0.0]]),
    player_pos=np.array([[[250.0, 0.0], [-250.0, 0.0]]]),
    player_vel=np.array([[[0.0, 0.0], [0.0, 0.0]]])
)

# Step 10 times with model
print("Testando modelo de cara pro gol adversário:")
for s in range(5):
    o = env.obs_builder.build_obs(env.engine.snapshot())
    # player 0 is Red
    with torch.no_grad():
        a = model.act(torch.tensor(o[:, 0], dtype=torch.float32), greedy=True).numpy()
    st = env.engine.snapshot()
    px, py = st.player_pos[0, 0]
    bx, by = st.ball_pos[0]
    print(f"Passo {s}: Player=({px:.1f}, {py:.1f}) | Bola=({bx:.1f}, {by:.1f}) | Ação=(dx={a[0,0]-1}, dy={a[0,1]-1}, kick={a[0,2]})")
    env.step(np.array([[[a[0, 0], a[0, 1], a[0, 2]], [1, 1, 0]]]))

"""Parity fixture from simulator observations and a separate-critic policy."""
import json
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from train import create_eval_env
from environment import STADIUM
from haxballgym.mutator import RandomStateMutator
from model import Policy
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path,default=root/'models/sprint1_latest/model.pt')
parser.add_argument('--out',type=Path,default=root/'runs/sprint1_structural_export/fixture.json')
args=parser.parse_args()
env=create_eval_env(128,1,1,STADIUM,8)
env.state_mutator=RandomStateMutator(ball_speed=3.,rng=8101)
obs=env.reset();state=env.prev_state
model=Policy.from_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
cases=[]
with torch.no_grad():
 logits=model.actor_logits(torch.tensor(obs.reshape(-1,16),dtype=torch.float32))
 actions=model.act(torch.tensor(obs.reshape(-1,16),dtype=torch.float32),greedy=True)
 def disc(pos,vel):return dict(pos=dict(x=float(pos[0]),y=float(pos[1])),speed=dict(x=float(vel[0]),y=float(vel[1])))
 for i in range(128):
  for side in [0,1]:
   index=2*i+side
   cases.append(dict(self=disc(state.player_pos[i,side],state.player_vel[i,side]),
       ball=disc(state.ball_pos[i],state.ball_vel[i]),opp=disc(state.player_pos[i,1-side],state.player_vel[i,1-side]),
       team=side+1,obs=obs[i,side].tolist(),logits=[h[index].tolist() for h in logits],action=actions[index].tolist()))
path=args.out;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(cases=cases)))
print(path)

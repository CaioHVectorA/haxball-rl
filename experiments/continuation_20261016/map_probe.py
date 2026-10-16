import sys,json,torch,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from train import create_eval_env
from haxballgym.mutator import RandomStateMutator
from model import Policy
from bot import ChaseBot
from haxballgym import BLUE,RED
torch.set_num_threads(2)
m=Policy(16,(3,3,2),separate_critic=True);m.load_state_dict(torch.load('models/sprint1_10000/model.pt',weights_only=True,map_location='cpu'));m.eval()
results=[]
for path in ['models/sprint1_10000/stadium.hbs','maps/futsal_1v1_contained.hbs']:
 for side in [0,1]:
  e=create_eval_env(8,1,1,path,8);e.state_mutator=RandomStateMutator(ball_speed=3.,rng=1701);o=e.reset();captured=[];last=e.reward_fn
  class Capture:
   def reset(self,state):last.reset(state)
   def get_rewards(self,state,prev,term,trunc):
    captured.append(state);return last.get_rewards(state,prev,term,trunc)
  e.reward_fn=Capture();bot=ChaseBot(seed=1701);gf=ga=0
  for t in range(200):
   a=np.empty((8,2,3),dtype=np.int64)
   with torch.no_grad():a[:,side]=m.act(torch.as_tensor(o[:,side],dtype=torch.float32),greedy=True).numpy()
   a[:,1-side]=bot.act(e.prev_state,[1-side]).reshape(8,3)
   o,r,term,trunc=e.step(a)
   # Capture actual goals before autoreset.
   state=captured.pop();gf+=int((state.scored==(BLUE if side==0 else RED)).sum());ga+=int((state.scored==(RED if side==0 else BLUE)).sum())
  results.append(dict(stadium=path,side=side,goals_for=gf,goals_against=ga))
print(json.dumps(results,indent=2))
Path('experiments/continuation_20261016/map_probe.json').write_text(json.dumps(results,indent=2))

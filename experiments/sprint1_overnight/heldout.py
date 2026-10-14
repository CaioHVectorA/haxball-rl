"""Independent seeds to gate overnight continuation; criteria fixed before evaluation."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from model import Policy
from environment import STADIUM
from live_evaluation import panel, balanced_score

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/sprint1_timeout_full_20261014'
OUT = ROOT/'runs/sprint1_overnight_validation_20261014'
OUT.mkdir(exist_ok=False)
torch.set_num_threads(2)
initial = Policy(16)
initial.load_state_dict(torch.load(RUN/'source_policy.pt', weights_only=True))
initial.eval()
candidates = {'source':RUN/'source_policy.pt',
              'B0_seed2_best':RUN/'B0_seed2/model.pt',
              'E1_seed0_best':RUN/'E1_seed0/model.pt'}
config = dict(seeds=[2701,2702,2703,2704,2705],envs=8,steps=600,
              candidates={k:str(v) for k,v in candidates.items()},
              criterion='Both scenarios in both modes must exceed source by >=0.02; select largest minimum scenario/mode score, then mean.',
              rationale='Preselect one strongest existing-panel candidate per arm; new seeds and sampled actions gate further exploration, not deployment.')
(OUT/'config.json').write_text(json.dumps(config,indent=2))
results=[]
for name, path in candidates.items():
 model=Policy(16);model.load_state_dict(torch.load(path,weights_only=True));model.eval()
 for greedy in [True,False]:
  mode='greedy' if greedy else 'sampled'
  print(name,mode,flush=True)
  def capture(row):
   with (OUT/'cases.jsonl').open('a') as f:
    f.write(json.dumps(dict(candidate=name,**row))+'\n')
  rows=panel(model, initial, STADIUM,0,capture,lambda *args:None,
             n=config['envs'], steps=config['steps'],eval_seeds=config['seeds'],greedy=greedy)
  record=dict(candidate=name,mode=mode,score=balanced_score(rows),
              **{s:balanced_score([r for r in rows if r['scenario']==s]) for s in ['random','kickoff']})
  results.append(record)
  (OUT/'results.json').write_text(json.dumps(results,indent=2))
  print(record,flush=True)
base={(r['mode'],s):r[s] for r in results if r['candidate']=='source' for s in ['random','kickoff']}
eligible=[]
for name in candidates:
 if name=='source':continue
 rows=[r for r in results if r['candidate']==name]
 deltas=[r[s]-base[r['mode'],s] for r in rows for s in ['random','kickoff']]
 if min(deltas)>=.02:
  eligible.append((min(r[s] for r in rows for s in ['random','kickoff']),
                   sum(r['score'] for r in rows)/2,name))
winner=max(eligible)[2] if eligible else None
(OUT/'decision.json').write_text(json.dumps(dict(selected=winner,checkpoint=str(candidates[winner]) if winner else None,
                     eligible=eligible,gate=config['criterion']),indent=2))
print('Decision',winner,flush=True)

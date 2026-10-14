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
OUT = ROOT/'runs/sprint1_overnight_final_validation_20261014'
OUT.mkdir(exist_ok=False)
torch.set_num_threads(2)
initial = Policy(16)
initial.load_state_dict(torch.load(RUN/'source_policy.pt', weights_only=True))
initial.eval()
CONTINUATION=ROOT/'runs/sprint1_overnight_model_20261014'
candidates = {'origin':CONTINUATION/'source_policy.pt',
              'best_it3650':CONTINUATION/'train/model.pt',
              'final_it5236':CONTINUATION/'train/model_final.pt'}
config = dict(seeds=[3701,3702,3703,3704,3705],envs=8,steps=600,
              candidates={k:str(v) for k,v in candidates.items()},
              reference=str(RUN/'source_policy.pt'),
              criterion='Final descriptive comparison: best and endpoint versus continuation origin, per scenario/mode. No further training or automatic promotion.',
              rationale='Unseen seeds after checkpoint selection; evaluation seeds are not independent training replicates.')
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

base={(r['mode'],s):r[s] for r in results if r['candidate']=='origin' for s in ['random','kickoff']}
deltas=[dict(candidate=r['candidate'],mode=r['mode'],
         **{s:r[s]-base[r['mode'],s] for s in ['random','kickoff']})
        for r in results if r['candidate']!='origin']
(OUT/'comparison.json').write_text(json.dumps(dict(results=results,deltas_vs_origin=deltas),indent=2))

"""Unseen-seed check of the new learner against its origin, including neural anchors."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from model import Policy
from live_evaluation import panel,balanced_score
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/sprint1_structural_main_20261014'
OUT=ROOT/'runs/sprint1_structural_heldout_20261014'
OUT.mkdir(exist_ok=False)
torch.set_num_threads(2)
payload=torch.load(RUN/'train/resume.pt',map_location='cpu',weights_only=True)
start=torch.load(RUN/'sources/start.pt',weights_only=True)
reference=Policy.from_state_dict(start).eval()
anchors={f'anchor_{i}':Policy.from_state_dict(torch.load(RUN/'sources'/name,weights_only=True)).eval()
         for i,name in enumerate(['deployed_anchor.pt','historical_anchor.pt'])}
config=dict(iteration=payload['best_iteration'],seeds=[5701,5702,5703],envs=8,steps=500,
            criterion='Descriptive unseen-seed comparison; no automatic promotion and no single-intervention causal claim')
(OUT/'config.json').write_text(json.dumps(config,indent=2))
results=[]
for name,state in [('origin',start),('structural',payload['best_model'])]:
 torch.save(state,OUT/f'{name}.pt')
 model=Policy.from_state_dict(state).eval()
 for mode in ['greedy','sampled']:
  def capture(row):
   with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(dict(candidate=name,**row))+'\n')
  print(name,mode,flush=True)
  cases=panel(model,reference,str(RUN/'maps/futsal_1v1.hbs'),0,capture,lambda *args:None,
              n=8,steps=500,eval_seeds=config['seeds'],greedy=mode=='greedy',extra_opponents=anchors)
  record=dict(candidate=name,mode=mode,score=balanced_score(cases),
              **{s:balanced_score([r for r in cases if r['scenario']==s]) for s in ['random','kickoff']})
  results.append(record);print(record,flush=True)
  (OUT/'results.json').write_text(json.dumps(results,indent=2))
base={(r['mode'],s):r[s] for r in results if r['candidate']=='origin' for s in ['random','kickoff']}
(OUT/'comparison.json').write_text(json.dumps(dict(config=config,results=results,
    deltas=[dict(mode=r['mode'],**{s:r[s]-base[r['mode'],s] for s in ['random','kickoff']})
            for r in results if r['candidate']=='structural']),indent=2))

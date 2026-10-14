"""Short paired ablation; no claim of convergence or attribution from the main preset."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/sprint1_structural_ablation_20261014'
OUT.mkdir(exist_ok=False)
source=OUT/'source';source.mkdir()
for p in (ROOT/'src').glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
(OUT/'maps').mkdir();(OUT/'maps/futsal_1v1.hbs').write_bytes((ROOT/'maps/futsal_1v1.hbs').read_bytes())
variants={'C1_corrected_shared':('corrected','--shared-critic'),
          'C2_pbrs_shared':('pbrs','--shared-critic'),
          'C3_pbrs_separate':('pbrs','--separate-critic')}
config=dict(seeds=[0,1],iters=40,envs=64,critic_warmup_iters=0,
             curriculum_frac=1.,target_kl=.03,variants=variants,
             samples_per_arm_seed=40*64*64,status='running',completed=[])
(OUT/'config.json').write_text(json.dumps(config,indent=2))
for seed in config['seeds']:
 for name,(table,critic) in variants.items():
  p=OUT/f'{name}_seed{seed}'
  command=[sys.executable,str(source/'train.py'),'--sprint1','--reward-table',table,critic,
     '--critic-warmup-iters','0','--envs','64','--iters','40','--threads','1','--seed',str(seed),
     '--warm-start',str(ROOT/'runs/sprint1_structural_main_20261014/sources/start.pt'),
     '--anchor-checkpoint',str(ROOT/'runs/sprint1_structural_main_20261014/sources/deployed_anchor.pt'),
     '--anchor-checkpoint',str(ROOT/'runs/sprint1_structural_main_20261014/sources/historical_anchor.pt'),
     '--out',str(p),'--stadium',str(OUT/'maps/futsal_1v1.hbs'),
     '--eval-envs','2','--eval-steps','300','--eval-every','40','--save-every','40']
  print(name,seed,flush=True)
  with (OUT/f'{name}_seed{seed}.log').open('w') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
  if result.returncode:
   config.update(status='failed',failed=f'{name}_seed{seed}')
   (OUT/'config.json').write_text(json.dumps(config,indent=2))
   raise RuntimeError(f'failed {p}; inspect log')
  config['completed'].append(f'{name}_seed{seed}')
  (OUT/'config.json').write_text(json.dumps(config,indent=2))
config['status']='completed';(OUT/'config.json').write_text(json.dumps(config,indent=2))

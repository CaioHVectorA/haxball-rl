"""Test the already-implemented critic calibration, at equal total sample budget."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/sprint1_structural_ablation_20261014'
config=dict(seeds=[0,1],iters=40,envs=64,critic_warmup_iters=20,
       rationale='Calibrated preset component was implemented before C1/C2/C3 results; compare to C3 with warmup0, same sample budget, fewer actor updates.')
(BASE/'calibration_config.json').write_text(json.dumps(config,indent=2))
for seed in config['seeds']:
 name=f'C4_pbrs_separate_calibrated_seed{seed}'
 out=BASE/name
 cmd=[sys.executable,str(BASE/'source/train.py'),'--sprint1','--critic-warmup-iters','20',
      '--envs','64','--iters','40','--threads','1','--seed',str(seed),
      '--warm-start',str(ROOT/'runs/sprint1_structural_main_20261014/sources/start.pt'),
      '--anchor-checkpoint',str(ROOT/'runs/sprint1_structural_main_20261014/sources/deployed_anchor.pt'),
      '--anchor-checkpoint',str(ROOT/'runs/sprint1_structural_main_20261014/sources/historical_anchor.pt'),
      '--out',str(out),'--stadium',str(BASE/'maps/futsal_1v1.hbs'),
      '--eval-envs','2','--eval-steps','300','--eval-every','40','--save-every','40']
 print(name,flush=True)
 with (BASE/f'{name}.log').open('w') as log:
  run=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 if run.returncode:raise RuntimeError(f'failed {name}; inspect log')
config['status']='completed';(BASE/'calibration_config.json').write_text(json.dumps(config,indent=2))

"""Launch the actual Sprint 1 learner with frozen code, map and source checkpoints."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/sprint1_structural_main_20261014'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=OUT/'source';source.mkdir()
    for p in (ROOT/'src').glob('*.py'):
        (source/p.name).write_bytes(p.read_bytes())
    maps=OUT/'maps';maps.mkdir()
    (maps/'futsal_1v1.hbs').write_bytes((ROOT/'maps/futsal_1v1.hbs').read_bytes())
    starts=[ROOT/'runs/sprint1_overnight_model_20261014/train/model.pt',
            ROOT/'models/model.pt',ROOT/'runs/sprint1_overnight_model_20261014/source_policy.pt']
    checkpoints=OUT/'sources';checkpoints.mkdir()
    names=['start.pt','deployed_anchor.pt','historical_anchor.pt']
    for path,name in zip(starts,names):(checkpoints/name).write_bytes(path.read_bytes())
    cmd=[sys.executable,str(source/'train.py'),'--sprint1','--envs','512','--iters','10000',
         '--threads','2','--seed','0','--warm-start',str(checkpoints/'start.pt'),
         '--anchor-checkpoint',str(checkpoints/'deployed_anchor.pt'),
         '--anchor-checkpoint',str(checkpoints/'historical_anchor.pt'),
         '--out',str(OUT/'train'),'--stadium',str(maps/'futsal_1v1.hbs'),
         '--eval-every','50','--eval-envs','16','--eval-steps','400',
         '--checkpoint-every','5','--save-every','50']
    with (OUT/'process.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=process.pid,started_at=datetime.now(timezone.utc).isoformat(),command=cmd,
       objective='Actual Sprint 1: PBRS, separate critic/clipping, stationary curriculum with fixed neural anchors',
       target_iterations=10000,stop_policy='No forced 10:30 cutoff; SIGTERM saves and exits; hard shutdown resumes from last durable checkpoint',
       source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
       source_checkpoint_sha256={n:hashlib.sha256((checkpoints/n).read_bytes()).hexdigest() for n in names},
       stadium_sha256=hashlib.sha256((maps/'futsal_1v1.hbs').read_bytes()).hexdigest())
    (OUT/'launch.json').write_text(json.dumps(record,indent=2))
    (OUT/'dependencies.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}'
                                          for d in importlib.metadata.distributions()))+'\n')
    (OUT/'working_tree.diff').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    print(json.dumps(dict(pid=process.pid,run=str(OUT),command=cmd),indent=2))

if __name__=='__main__':main()

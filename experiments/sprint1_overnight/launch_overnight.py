"""Launch one heldout-qualified continuation, bounded by an absolute 07:00 cutoff."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
VALIDATION=ROOT/'runs/sprint1_overnight_validation_20261014'
FULL=ROOT/'runs/sprint1_timeout_full_20261014'
OUT=ROOT/'runs/sprint1_overnight_model_20261014'
DEADLINE='2026-10-14T07:00:00-03:00'


def main():
    decision=json.loads((VALIDATION/'decision.json').read_text())
    if not decision['selected']:
        raise RuntimeError('No candidate qualified; do not launch a continuation.')
    winner=decision['selected']
    checkpoint=Path(decision['checkpoint'])
    arm=winner.split('_best')[0]
    source_config=json.loads((FULL/arm/'config.json').read_text())
    elapsed=json.loads((FULL/arm/'live.json').read_text())['elapsed_s']
    seconds_per_iteration=elapsed/source_config['iters']
    remaining=datetime.fromisoformat(DEADLINE).timestamp()-datetime.now(timezone.utc).timestamp()-120
    if remaining<300:raise RuntimeError('Insufficient time for meaningful continuation.')
    iterations=max(1,math.ceil(remaining/seconds_per_iteration))
    OUT.mkdir(exist_ok=False)
    source=OUT/'source';source.mkdir()
    for f in (ROOT/'src').glob('*.py'):
        (source/f.name).write_bytes(f.read_bytes())
    maps=OUT/'maps';maps.mkdir()
    stadium=maps/'futsal_1v1.hbs'
    stadium.write_bytes((ROOT/'maps/futsal_1v1.hbs').read_bytes())
    frozen=OUT/'source_policy.pt';frozen.write_bytes(checkpoint.read_bytes())
    (OUT/'source_meta.json').write_bytes(checkpoint.with_name('meta.json').read_bytes())
    supervisor=OUT/'deadline_supervisor.py'
    supervisor.write_bytes(Path(__file__).with_name('deadline_supervisor.py').read_bytes())
    (OUT/'launcher.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'working_tree.diff').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    (OUT/'dependencies.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}'
                         for d in importlib.metadata.distributions()))+'\n')
    command=[sys.executable,str(source/'train.py'),'--warm-start',str(frozen),'--keep-critic',
             '--timeout-bootstrap',source_config['timeout_bootstrap'],'--approach-weight','0.15',
             '--seed',str(source_config['seed']),'--iters',str(iterations),'--envs','512','--threads','2',
             '--save-every','50','--eval-every','50','--eval-envs','16','--eval-steps','400',
             '--out',str(OUT/'train'),'--stadium',str(stadium),'--stop-at',DEADLINE]
    record=dict(selected=winner,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
       source_metadata=json.loads((OUT/'source_meta.json').read_text()),
       deadline=DEADLINE,estimated_seconds_per_iteration=seconds_per_iteration,
       remaining_seconds=remaining,target_iterations=iterations,target_samples=iterations*512*64,
       selection_criteria=decision['gate'],
       source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.py')},
       stadium_sha256=hashlib.sha256(stadium.read_bytes()).hexdigest(),
       warm_start_semantics='actor and critic preserved; Adam, pool, schedules and iteration counter restarted',
       task='Exploratory continuation; not an E1-vs-B0 experiment or deployment certification',
       command=command,started_at=datetime.now(timezone.utc).isoformat())
    log=(OUT/'supervisor.log').open('w')
    proc=subprocess.Popen([sys.executable,str(supervisor),'--deadline',DEADLINE,
                     '--out',str(OUT/'supervision'),'--',*command],cwd=ROOT,
                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    record['supervisor_pid']=proc.pid
    (OUT/'launch.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:record[k] for k in ['selected','target_iterations','target_samples','supervisor_pid','deadline']},indent=2))

if __name__=='__main__':main()

"""Sequential matched self-play reward trials; keep the live-room policy unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from telemetry import atomic_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--iters',type=int,default=300)
    ap.add_argument('--envs',type=int,default=256)
    ap.add_argument('--weights',type=float,nargs='+',default=[.6,.15,.05])
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    args=ap.parse_args()
    if args.out.exists():raise SystemExit('Output already exists.')
    if not args.checkpoint.is_file():raise SystemExit('Checkpoint missing.')
    args.out.mkdir(parents=True)
    status=dict(pid=os.getpid(),status='running',config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},trials=[])
    atomic_json(args.out/'sweep.json',status)
    for seed in args.seeds:
        # Reverse ordering for the second seed to avoid confounding weight with wall time.
        weights=args.weights if seed==args.seeds[0] else list(reversed(args.weights))
        for weight in weights:
            run=(args.out/f'approach_{weight:.2f}_seed{seed}').resolve()
            run.mkdir()
            command=[sys.executable,'-u','src/train.py','--envs',str(args.envs),'--iters',str(args.iters),
                '--seed',str(seed),'--threads','2','--warm-start',str(args.checkpoint.resolve()),
                '--approach-weight',str(weight),'--save-every','50','--eval-every','100','--out',str(run)]
            child_env=os.environ.copy();child_env['RAYON_NUM_THREADS']='4'
            with (run/'training.log').open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=child_env)
            atomic_json(Path('runs/active.json'),{'path':str(run),'pid':child.pid})
            status['active']=dict(seed=seed,weight=weight,path=str(run),pid=child.pid)
            atomic_json(args.out/'sweep.json',status)
            code=child.wait()
            result=dict(seed=seed,weight=weight,path=str(run),exit_code=code)
            status['trials'].append(result)
            if code:
                status['status']='failed';atomic_json(args.out/'sweep.json',status)
                raise SystemExit(f'Trial failed; inspect {run}/training.log')
            rows=[json.loads(s) for s in (run/'evaluations.jsonl').read_text().splitlines()]
            # Evaluate endpoints; report best separately to avoid hiding regressions.
            for label,row in [('initial',rows[0]),('final',rows[-1])]:
                values=[c['score'] for c in row['cases'] if c['scenario']=='random' and c['score'] is not None]
                result[label+'_random_score']=sum(values)/len(values) if values else None
            inspection=run/'behavior_final.json'
            with (run/'inspection.log').open('w') as log:
                inspected=subprocess.run([sys.executable,'src/inspect_policy.py',str(run/'model_final.pt'),'--out',str(inspection)],stdout=log,stderr=subprocess.STDOUT,env=child_env)
            result['inspection_exit_code']=inspected.returncode
            result['inspection']=str(inspection)
            atomic_json(args.out/'sweep.json',status)
            print(json.dumps(result),flush=True)
    status['status']='completed';status.pop('active',None)
    atomic_json(args.out/'sweep.json',status)

if __name__=='__main__':main()

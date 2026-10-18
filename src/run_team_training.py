"""Start one 2v2 transfer run; maintain a portable local bundle and publish its endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from telemetry import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--iters',type=int,default=10000)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    if (run/'live.json').exists():raise FileExistsError('Use a new run directory')
    if subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()!='gpt_2v2':
        raise ValueError('Expected branch gpt_2v2')
    original=hashlib.sha256((root/'models/model.pt').read_bytes()).hexdigest()
    log=(run/'supervisor.log').open('a',buffering=1)
    def call(parts):subprocess.run(parts,cwd=root,stdout=log,stderr=log,check=True)
    def bundle():
        call([sys.executable,'src/backup_training.py','--run',str(run),'--dest','models/team2v2_latest'])
        for weights,name in [('model.pt','policy.json'),('latest.pt','policy_latest.json')]:
            call([sys.executable,'headless-bot/export_policy.py',f'models/team2v2_latest/{weights}',
                  '--out',f'models/team2v2_latest/{name}'])
    try:
        with (run/'train.log').open('a',buffering=1) as output:
            child=subprocess.Popen([sys.executable,'-u','src/train.py','--team2v2','--transfer-1v1',
                '--warm-start','models/sprint1_10000/model.pt','--device','cuda','--envs','256',
                '--iters',str(args.iters),'--eval-every','100','--out',str(run),
                '--checkpoint-every','5','--save-every','50'],cwd=root,stdout=output,stderr=subprocess.STDOUT)
            atomic_json(run/'supervisor.json',dict(pid=child.pid,status='running'))
            last=-100
            while child.poll() is None:
                try:
                    live=json.loads((run/'live.json').read_text())
                    iteration=live.get('completed_iterations',0)
                    if (run/'resume.pt').exists() and iteration>=last+100:
                        bundle();last=iteration
                except (FileNotFoundError,json.JSONDecodeError):pass
                except subprocess.CalledProcessError as error:
                    # A temporary export failure must not abandon the live trainer.
                    log.write(f'Local bundle update failed: {error}\n')
                time.sleep(20)
            code=child.wait()
        if (run/'resume.pt').exists():bundle()
        if code:raise RuntimeError(f'Trainer exited {code}; inspect train.log; no automatic restart')
        live=json.loads((run/'live.json').read_text())
        archive=root/'experiments'/'team2v2'/run.name
        archive.mkdir(parents=True,exist_ok=True)
        files=[run/name for name in ['config.json','live.json','training.jsonl','evaluations.jsonl',
            'evaluation_cases.jsonl','train.log','initial_policy.pt','model_final.pt','meta_final.json'] if (run/name).exists()]
        files+=sorted((run/'ckpts').glob('*.pt'))
        groups=[[]];size=0
        for file in files:
            if size+file.stat().st_size>40_000_000 and groups[-1]:groups.append([]);size=0
            groups[-1].append(file);size+=file.stat().st_size
        for i,group in enumerate(groups,1):
            with zipfile.ZipFile(archive/f'records_{i:02}.zip','w',zipfile.ZIP_DEFLATED) as output:
                for file in group:output.write(file,file.relative_to(run))
        atomic_json(archive/'manifest.json',dict(iteration=live['completed_iterations'],
            files=[dict(path=str(p.relative_to(run)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]))
        if hashlib.sha256((root/'models/model.pt').read_bytes()).hexdigest()!=original:
            raise RuntimeError('Original 1v1 model changed')
        if subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()!='gpt_2v2':
            raise RuntimeError('Branch changed; endpoint preserved locally')
        call(['git','add','--','models/team2v2_latest',str(archive)])
        call(['git','add','-f','--',*[str(p) for p in archive.glob('records_*.zip')]])
        call(['git','commit','-m',f'Publish 2v2 self-play endpoint at {live["completed_iterations"]}'])
        call(['git','push','origin','gpt_2v2'])
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        remote=subprocess.check_output(['git','ls-remote','--heads','origin','gpt_2v2'],cwd=root,text=True).split()[0]
        if head!=remote:raise RuntimeError('Remote commit not confirmed')
        atomic_json(run/'publication.json',dict(status='published',iteration=live['completed_iterations'],commit=head))
    except BaseException as error:
        atomic_json(run/'publication.json',dict(status='failed',error=repr(error)))
        raise
    finally:log.close()


if __name__=='__main__':main()

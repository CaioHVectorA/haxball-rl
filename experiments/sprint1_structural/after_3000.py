"""Wait for the authorized endpoint, freeze its bundle, benchmark, and prepare a human room."""
import json
import os
from pathlib import Path
import subprocess
import time
import hashlib
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/sprint1_structural_main_20261014/train'
OUT=ROOT/'runs/sprint1_benchmark_3000_20261014'
STATE=ROOT/'runs/sprint1_after_3000.json'
PY=ROOT/'.venv/bin/python'


def update(**fields):
    data=json.loads(STATE.read_text()) if STATE.exists() else {}
    data.update(fields,updated_at=time.time());temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(STATE)


def run(*args):
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)


def main():
    update(status='waiting_for_3000',pid=os.getpid())
    while True:
        live=json.loads((RUN/'live.json').read_text())
        if live['status']=='failed':raise RuntimeError('Trainer failed before benchmark')
        if live['status']=='completed':break
        if live.get('completed_iterations',0)>3000:raise RuntimeError('Iteration stop missed target')
        time.sleep(2)
    if live['completed_iterations']!=3000:raise RuntimeError('Unexpected endpoint')
    checkpoint=torch.load(RUN/'resume.pt',weights_only=True,map_location='cpu')
    if checkpoint['iteration']!=3000:raise RuntimeError('Final durable save missing')
    bundle=ROOT/'models/sprint1_3000'
    run(PY,'src/backup_training.py','--run',RUN,'--dest',bundle)
    run(PY,'src/backup_training.py','--run',RUN,'--dest',ROOT/'models/sprint1_latest')
    backup=json.loads((bundle/'backup.json').read_text())
    if hashlib.sha256((bundle/'resume.pt').read_bytes()).hexdigest()!=backup['resume_sha256']:raise RuntimeError('Bundle hash mismatch')
    for file,key in [('model.pt','best_model'),('latest.pt','model')]:
        sd=torch.load(bundle/file,weights_only=True,map_location='cpu')
        if sd.keys()!=checkpoint[key].keys() or not all(torch.equal(v,checkpoint[key][k]) and torch.isfinite(v).all() for k,v in sd.items()):raise RuntimeError('Bundle weights mismatch')
    update(status='benchmarking',bundle=str(bundle),best_iteration=checkpoint['best_iteration'],benchmark=str(OUT))
    run(PY,'experiments/sprint1_structural/benchmark_3000.py','--out',OUT)
    config=json.loads((OUT/'config.json').read_text())
    if config['status']!='completed':raise RuntimeError('Incomplete benchmark')
    update(status='preparing_human_room',cases=config['cases'])
    run(PY,'headless-bot/export_policy.py',bundle/'model.pt','--out',bundle/'policy.json')
    run(PY,'experiments/sprint1_structural/export_fixture.py','--checkpoint',bundle/'model.pt','--out',OUT/'export_fixture.json')
    run('node','tests/verify_export.cjs',OUT/'export_fixture.json',bundle/'policy.json')
    # Private room uses the exact frozen pre-benchmark best, not a checkpoint chosen retrospectively.
    env=dict(os.environ,POLICY_PATH=str(bundle/'policy.json'),ROOM_NAME='Sprint 1 — 3000 / modelo avaliado',ROOM_PASS='bot')
    log=ROOT/'runs/sprint1_3000_room.log'
    with log.open('w') as f:
        process=subprocess.Popen(['node','bot.js'],cwd=ROOT/'headless-bot',env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    update(status='completed',room_pid=process.pid,room_log=str(log),room_status='starting',
           room_policy=str(bundle/'policy.json'),room_checkpoint_iteration=checkpoint['best_iteration'])


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        update(status='failed',error=repr(exc));raise

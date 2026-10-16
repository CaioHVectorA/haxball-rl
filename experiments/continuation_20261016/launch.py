"""Launch one frozen-source warm start and keep periodic coherent backups."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.venv/bin/python'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    # Inspect real process arguments rather than trusting a stale PID file.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(arg.endswith(b'/train.py') or arg == b'src/train.py' for arg in args):
            raise RuntimeError(f'Existing trainer PID {proc.name}; refusing duplicate')
    stamp = datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%Y%m%d_%H%M%S')
    run = ROOT / 'runs' / f'continued_warmstart_{stamp}'
    run.mkdir(parents=True, exist_ok=False)
    frozen = run / 'frozen'
    shutil.copytree(ROOT / 'src', frozen / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'models/sprint1_10000/stadium.hbs', frozen / 'stadium.hbs')
    shutil.copy2(ROOT / 'models/sprint1_10000/model.pt', frozen / 'origin.pt')
    shutil.copy2(ROOT / 'models/sprint1_10000/meta.json', frozen / 'origin_meta.json')
    import torch
    resume = torch.load(ROOT / 'models/sprint1_10000/resume.pt', map_location='cpu', weights_only=True)
    source = torch.load(frozen / 'origin.pt', map_location='cpu', weights_only=True)
    assert all(torch.isfinite(v).all().item() for v in source.values())
    assert all(torch.equal(v, resume['best_model'][k]) for k,v in source.items())
    assert resume['best_iteration'] == 9900
    for i, anchor in enumerate(resume['anchors']):
        torch.save(anchor, frozen / f'anchor_{i}.pt')
    command = [str(PYTHON), '-u', str(frozen / 'src/train.py'), '--sprint1',
               '--warm-start', str(frozen / 'origin.pt'), '--keep-critic',
               '--stadium', str(frozen / 'stadium.hbs'), '--device', 'cpu',
               '--envs', '512', '--threads', '2', '--seed', '16',
               '--continuous', '--iters', '10000', '--actor-lr', '0.0001', '--critic-lr', '0.0001',
               '--eval-every', '50', '--checkpoint-every', '5', '--save-every', '1000',
               '--out', str(run / 'train')]
    for i in range(len(resume['anchors'])):
        command += ['--anchor-checkpoint', str(frozen / f'anchor_{i}.pt')]
    manifest = dict(created=datetime.now(ZoneInfo('America/Sao_Paulo')).isoformat(),
                    source_checkpoint='models/sprint1_10000/model.pt', source_iteration=9900,
                    command=command, supervisor_pid=os.getpid(), torch=torch.__version__,
                    cuda_available=torch.cuda.is_available(),
                    semantics='Warm start: actor and critic retained; Adam/pool/RNG/schedules reset; physical matches reset',
                    map_change='Exact original checkpoint stadium; contained map rejected after native-engine side asymmetry probe',
                    evaluation_reference='Warm-start policy (9900); old two anchors preserved; baseline measured before updates',
                    budget='Continuous until user interruption or failure; --iters 10000 sets schedule horizon only, not stopping budget',
                    hashes={str(p.relative_to(run)):digest(p) for p in frozen.rglob('*') if p.is_file()})
    with (run / 'train.log').open('w') as log:
        trainer = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
    manifest['trainer_pid'] = trainer.pid
    (run / 'launch.json').write_text(json.dumps(manifest, indent=2))
    (ROOT / 'runs/active_local_warmstart.txt').write_text(str(run)+'\n')
    print(json.dumps(dict(run=str(run), trainer_pid=trainer.pid)), flush=True)
    last_backup = 0
    while True:
        code = trainer.poll()
        if (run / 'train/resume.pt').exists() and (time.monotonic()-last_backup > 1800 or code is not None):
            with (run / 'backup.log').open('a') as output:
                result = subprocess.run([str(PYTHON), str(ROOT / 'src/backup_training.py'),
                       '--run', str(run / 'train'), '--dest', str(ROOT / 'models/local_warmstart_latest')],
                       cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            last_backup = time.monotonic()
            print(json.dumps(dict(backup_returncode=result.returncode)), flush=True)
        if code is not None:
            (run / 'exit.json').write_text(json.dumps(dict(returncode=code, ended=datetime.now(ZoneInfo('America/Sao_Paulo')).isoformat())))
            return code
        time.sleep(30)

if __name__ == '__main__':
    sys.exit(main())

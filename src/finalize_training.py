"""Watch one existing run, publish its portable bundle and records, then optionally shut down."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (FileNotFoundError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--shutdown-after-push', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    run = args.run.resolve()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    lock = run/'completion_watch.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    original = hashlib.sha256((root/'models/model.pt').read_bytes()).hexdigest()

    def command(*parts):
        subprocess.run(list(parts), cwd=root, check=True)

    def status(**values):
        from telemetry import atomic_json
        atomic_json(run/'delivery.json', dict(updated_at=datetime.now(timezone.utc).isoformat(), **values))

    try:
        status(status='watching', deadline=args.deadline)
        while True:
            live = read_json(run/'live.json')
            if live.get('status') in ('completed', 'failed') or time.time() >= deadline-1200:
                break
            time.sleep(30)
        # Let the original supervisor finish its existing bundle/commit/push first.
        if live.get('status') == 'completed':
            until = min(time.time()+600, deadline-600)
            while time.time() < until:
                publication = read_json(run/'publication.json')
                if publication.get('status') == 'published' or 'failed' in publication.get('status', ''):
                    break
                time.sleep(10)
        import torch
        checkpoint = torch.load(run/'resume.pt', map_location='cpu', weights_only=True)
        iteration = checkpoint['iteration']
        complete = iteration == 10000 and live.get('status') == 'completed'
        bundle = Path('models')/f'sprint1_{iteration}'
        status(status='preparing', iteration=iteration)
        command(sys.executable, 'src/backup_training.py', '--run', str(run), '--dest', str(bundle))
        for weights, policy in [('model.pt', 'policy.json'), ('latest.pt', 'policy_latest.json')]:
            command(sys.executable, 'headless-bot/export_policy.py', str(bundle/weights), '--out', str(bundle/policy))
        for filename in ('policy.json', 'policy_latest.json'):
            policy = read_json(bundle/filename)
            if policy.get('obs_dim') != 16 or not policy.get('repeat_kick') or not policy.get('stadium_hbs'):
                raise ValueError('Export does not preserve the expected policy contract')
        archive = Path('experiments/sprint1_structural')/f'gpu_continuation_{iteration}'
        archive.mkdir(parents=True, exist_ok=True)
        # Explicit allowlist: never archive env files, tokens, node_modules or credentials.
        filenames = ['training.jsonl', 'evaluations.jsonl', 'evaluation_cases.jsonl',
                     'config.json', 'live.json', 'train.log', 'meta.json', 'meta_final.json',
                     'initial_policy.pt', 'model_final.pt', 'resume_stadium.hbs',
                     'initial_policy_pre_recovery.pt', 'train_pre_recovery.log', 'train_error.log']
        files = [run/name for name in filenames if (run/name).is_file()]
        files += sorted((run/'ckpts').glob('*.pt'))
        manifest = []
        groups = [[]]
        size = 0
        for file in files:
            if size + file.stat().st_size > 40_000_000 and groups[-1]:
                groups.append([])
                size = 0
            groups[-1].append(file)
            size += file.stat().st_size
            manifest.append(dict(path=str(file.relative_to(run)).replace('\\', '/'),
                                 size=file.stat().st_size,
                                 sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        for index, group in enumerate(groups, 1):
            with zipfile.ZipFile(archive/f'records_{index:02}.zip', 'w', zipfile.ZIP_DEFLATED) as output:
                for file in group:
                    output.write(file, file.relative_to(run))
        (archive/'manifest.json').write_text(json.dumps(dict(iteration=iteration, complete=complete,
            source_run=run.name, bundle=str(bundle), files=manifest), indent=2), encoding='utf-8')
        (bundle/'USAGE.md').write_text(f'''# Sprint 1: iteration {iteration}

`model.pt` is the best internally selected policy; `latest.pt` is the endpoint.
`resume.pt` includes Adam, schedules, references, opponent pool and RNGs.
The original `models/model.pt` is preserved. Records and intermediate checkpoints:
`{archive.as_posix()}` (ZIP parts with SHA-256 manifest).

## Web test

Pre-exported `policy.json` uses the best model; `policy_latest.json` uses the endpoint.
From the repository root in PowerShell, install the existing room dependencies:

```powershell
npm --prefix headless-bot ci
$env:POLICY_PATH = (Resolve-Path '{bundle.as_posix()}/policy.json').Path
node headless-bot/bot.js
```

Supply HEADLESS_TOKEN only in the process environment, using a fresh Haxball token.
The bot prints the room link to open in the web client. No token is stored here.
Export compatibility was checked; human gameplay has not been validated.

## Warm start (a new training run, fresh optimizer and schedules)

```powershell
python src/train.py --sprint1 --warm-start {bundle.as_posix()}/latest.pt --device cuda --out runs/new_finetune
```

Use `model.pt` instead to start from the internally selected best policy.
For exact training continuation, use `--resume`, which preserves the saved recipe.
The 10000-iteration resume has already exhausted its original budget; extending it
requires an explicit new budget decision rather than silently changing schedules.
''', encoding='utf-8')
        if hashlib.sha256((root/'models/model.pt').read_bytes()).hexdigest() != original:
            raise RuntimeError('Original model changed; refusing publication')
        if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() != 'gpt':
            raise RuntimeError('Expected branch gpt')
        paths = ['pyproject.toml', 'uv.lock', 'src/telemetry.py', 'src/training_checkpoint.py',
                 'src/select.py', 'src/select_checkpoint.py', 'src/finalize_training.py',
                 'live_stats_server.py', 'README.md', 'PROTOCOL.md',
                 'experiments/sprint1_structural/README.md', str(bundle), str(archive)]
        # A deleted path may already have been committed by the first supervisor.
        paths = [p for p in paths if Path(p).exists() or subprocess.run(
            ['git', 'ls-files', '--error-unmatch', p], capture_output=True).returncode == 0]
        command('git', 'add', '--', *paths)
        command('git', 'add', '-f', '--', *[str(p) for p in archive.glob('records_*.zip')])
        if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode:
            command('git', 'commit', '-m', f'Archive CUDA training through {iteration}; prepare web and warm start')
        command('git', 'push', 'origin', 'gpt')
        local = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'origin', 'gpt'], text=True).split()[0]
        if local != remote:
            raise RuntimeError('Remote does not match the published commit')
        status(status='published', iteration=iteration, complete=complete, commit=local,
               bundle=str(bundle), archive=str(archive))
        if complete and args.shutdown_after_push:
            # Allow the 15-minute thread monitor to report before the host goes offline.
            command('shutdown.exe', '/a')
            seconds = max(0, min(1200, int(deadline-time.time())))
            command('shutdown.exe', '/s', '/t', str(seconds))
            status(status=f'published; shutdown scheduled in {seconds} seconds', iteration=iteration,
                   complete=complete, commit=local, bundle=str(bundle), archive=str(archive))
    except BaseException as error:
        status(status='failed', error=repr(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

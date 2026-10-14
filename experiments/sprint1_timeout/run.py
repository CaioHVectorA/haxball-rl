"""Run paired B0/E1 experiments in a new directory (never promote checkpoints)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os
from datetime import datetime, timezone
import importlib.metadata

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--envs', type=int, default=512)
    parser.add_argument('--eval-envs', type=int, default=16)
    parser.add_argument('--eval-steps', type=int, default=400)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--approach-weight', type=float, default=.15)
    args = parser.parse_args()
    if min(args.iters, args.envs, args.eval_envs, args.eval_steps, args.eval_every) < 1:
        parser.error("budgets must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    checkpoint = args.checkpoint.resolve()
    payload = checkpoint.read_bytes()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out/'source_policy.pt'
    frozen.write_bytes(payload)
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
                    seeds=args.seeds, iters=args.iters, envs=args.envs,
                    samples_per_arm_seed=args.iters*args.envs*64,
                    approach_weight=args.approach_weight, critic_reset=True,
                    variants={'B0':'zero', 'E1':'final'}, status='running', completed=[],
                    pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                    eval_envs=args.eval_envs, eval_steps=args.eval_steps, eval_every=args.eval_every)
    # Archive implementation because HEAD alone does not identify uncommitted work.
    source = out/'source'; source.mkdir()
    for path in sorted((ROOT/'src').glob('*.py')):
        (source/path.name).write_bytes(path.read_bytes())
    (out/'working_tree.diff').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    (out/'maps').mkdir()
    stadium = out/'maps/futsal_1v1.hbs'
    stadium.write_bytes((ROOT/'maps/futsal_1v1.hbs').read_bytes())
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'summarize.py').write_bytes(Path(__file__).with_name('summarize.py').read_bytes())
    manifest['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(source.glob('*.py'))}
    manifest['stadium_sha256'] = hashlib.sha256(stadium.read_bytes()).hexdigest()
    (out/'dependencies.txt').write_text('\n'.join(sorted(
        f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()))+'\n')
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    save()
    try:
        for seed in args.seeds:
            for name, mode in manifest['variants'].items():
                folder = out/f'{name}_seed{seed}'
                command = [sys.executable, str(source/'train.py'), '--warm-start', str(frozen),
                           '--out', str(folder), '--seed', str(seed), '--timeout-bootstrap', mode,
                           '--iters', str(args.iters), '--envs', str(args.envs),
                           '--eval-envs', str(args.eval_envs), '--eval-steps', str(args.eval_steps),
                           '--eval-every', str(args.eval_every), '--approach-weight', str(args.approach_weight),
                           '--save-every', '50', '--stadium', str(stadium)]
                manifest['current']=f'{name}_seed{seed}'
                save()
                print(f'Starting {name}, seed {seed}', flush=True)
                with (out/f'{name}_seed{seed}.log').open('w') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                manifest['completed'].append(f'{name}_seed{seed}')
                save()
                subprocess.run([sys.executable,str(out/'summarize.py'),str(out)],check=True)
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['finished_at']=datetime.now(timezone.utc).isoformat()
        save()
        subprocess.run([sys.executable,str(out/'summarize.py'),str(out)],check=True)

if __name__ == '__main__':
    main()

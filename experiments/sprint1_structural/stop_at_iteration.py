"""Stop an identified existing trainer after a completed iteration, without changing schedules."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        # Field 22 is the process start time; parse after the parenthesized comm.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--iteration',type=int,required=True)
    args=p.parse_args()
    token=identity(args.pid)
    if token is None:raise RuntimeError('Trainer is not alive')
    cmd=Path(f'/proc/{args.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'train.py' not in cmd or str(args.run.resolve()) not in cmd:
        raise RuntimeError('PID does not identify the requested trainer')
    dest=args.run.parent/'iteration_stop.json'
    record=dict(pid=args.pid,start_identity=token,target=args.iteration,
                original_schedule_budget=10000,status='watching',command=cmd)
    def save():
        temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(dest)
    save()
    while identity(args.pid)==token:
        live=json.loads((args.run/'live.json').read_text())
        completed=live.get('completed_iterations',0)
        if completed>=args.iteration:
            os.kill(args.pid,signal.SIGTERM)
            record.update(status='signal_sent',observed_iteration=completed,sent_at=time.time());save()
            break
        time.sleep(.1)
    while identity(args.pid)==token:
        # A zombie has exited even if its parent has not reaped it yet.
        stat=Path(f'/proc/{args.pid}/stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':break
        time.sleep(.5)
    live=json.loads((args.run/'live.json').read_text())
    record.update(status='exited',final_status=live.get('status'),
                  final_iteration=live.get('completed_iterations'),finished_at=time.time())
    save()
    if live.get('status')!='completed' or live.get('completed_iterations')!=args.iteration:
        raise RuntimeError(f'Unexpected endpoint: {live}')


if __name__=='__main__':main()

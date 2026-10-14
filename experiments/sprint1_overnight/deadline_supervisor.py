"""Absolute-deadline process-group supervisor for bounded overnight experiments."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--deadline',required=True)
 parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('command',nargs=argparse.REMAINDER)
 args=parser.parse_args()
 stamp=datetime.fromisoformat(args.deadline)
 if stamp.tzinfo is None:parser.error('timezone required')
 seconds=stamp.timestamp()-time.time()
 if seconds<=0:parser.error('deadline passed')
 command=args.command[1:] if args.command[:1]==['--'] else args.command
 if not command:parser.error('command required')
 args.out.mkdir(parents=True,exist_ok=False)
 state=dict(status='starting',supervisor_pid=os.getpid(),deadline=args.deadline,
            command=command,started_at=datetime.now(timezone.utc).isoformat())
 def save():
  temporary=args.out/'supervisor.json.tmp'
  temporary.write_text(json.dumps(state,indent=2))
  os.replace(temporary,args.out/'supervisor.json')
 save()
 with (args.out/'process.log').open('w') as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state.update(status='running',pid=child.pid);save()
  try:
   code=child.wait(timeout=max(.01,stamp.timestamp()-time.time()-5))
   state.update(status='completed' if code==0 else 'failed',returncode=code)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=max(.01,stamp.timestamp()-time.time()-1))
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL);child.wait()
   # Stop remaining children even if the leader exited on SIGTERM.
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   state.update(status='deadline_enforced',returncode=child.returncode)
  finally:
   state['finished_at']=datetime.now(timezone.utc).isoformat();save()

if __name__=='__main__':main()

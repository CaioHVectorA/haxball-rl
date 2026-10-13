"""Local read-only dashboard, live run state and server-sent telemetry events."""
import argparse
from datetime import datetime, timezone
from functools import partial
import http.server
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from src.telemetry import atomic_json

BASE_DIR=Path(__file__).resolve().parent
ACTIVE=BASE_DIR/'runs'/'active.json'


def read_json(path, default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,ValueError):return default


def read_records(path):
    try:lines=path.read_text().splitlines()
    except FileNotFoundError:return []
    result=[]
    for line in lines:
        try:result.append(json.loads(line))
        except ValueError:pass
    return result


def run_directory():
    entry=read_json(ACTIVE,{})
    return Path(entry['path']) if entry.get('path') else None


def live_state(run):
    if run is None:return dict(status='waiting',phase='waiting',server_time=time.time())
    state=read_json(run/'live.json',dict(status='starting',phase='starting'))
    state['run_path']=str(run)
    state['server_time']=time.time()
    if state.get('updated_at'):
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['updated_at'])).total_seconds()
        state['heartbeat_age_s']=age
        try:
            os.kill(state['pid'],0)
            command=Path(f'/proc/{state["pid"]}/cmdline').read_bytes().decode().replace('\0',' ')
            alive='src/train.py' in command
        except (ProcessLookupError,PermissionError,FileNotFoundError,KeyError):alive=False
        if state.get('status')=='running' and not alive:state['status']='interrupted'
        elif state.get('status')=='running' and age>20:state['status']='stale'
    return state


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(BASE_DIR),**kwargs)

    def log_message(self,*args):pass

    def json(self,data):
        self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store');self.end_headers()
        self.wfile.write(json.dumps(data,allow_nan=False).encode())

    def do_GET(self):
        path=urlparse(self.path).path
        run=run_directory()
        if path=='/api/state':return self.json(live_state(run))
        if path=='/api/history':return self.json(dict(training=read_records(run/'training.jsonl') if run else [],evaluations=read_records(run/'evaluations.jsonl') if run else []))
        if path=='/api/events':
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.send_header('Cache-Control','no-cache');self.send_header('Connection','keep-alive')
            self.end_headers();previous=None
            train_count=eval_count=0
            try:
                while True:
                    current=run_directory()
                    if current!=previous:train_count=eval_count=0;previous=current
                    training=read_records(current/'training.jsonl') if current else []
                    evaluation=read_records(current/'evaluations.jsonl') if current else []
                    state=live_state(current)
                    data=dict(live=state,training=training[train_count:],evaluations=evaluation[eval_count:])
                    train_count=len(training);eval_count=len(evaluation)
                    self.wfile.write(('data: '+json.dumps(data,allow_nan=False)+'\n\n').encode());self.wfile.flush()
                    time.sleep(1)
            except (BrokenPipeError,ConnectionResetError):return
        if path in ('/','/dashboard.html'):
            self.path='/dashboard.html';return super().do_GET()
        # Serve only dashboard assets, not checkpoints or workspace files.
        if path in ('/dashboard.js','/dashboard.css'):return super().do_GET()
        self.send_error(404)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8085)
    args=ap.parse_args()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Dashboard: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()

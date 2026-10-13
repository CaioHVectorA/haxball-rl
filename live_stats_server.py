import http.server
import socketserver
import json
import re
import subprocess
from pathlib import Path

PORT = 8085
BASE_DIR = Path(__file__).parent
LOG_PATH = Path("/home/usuario/.gemini/antigravity/brain/9843ad18-37de-4e8b-a25e-ee4d811d6f71/.system_generated/tasks/task-2031.log")
POC_FILE = BASE_DIR / "poc_results.json"
poc_process = None

class DynamicHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        global poc_process
        if self.path.startswith("/api/run-poc"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                params = json.loads(body)
            except Exception:
                params = {}
                
            iters = int(params.get("iters", 150))
            envs = int(params.get("envs", 256))
            warm_start = params.get("warm_start", "models/model_final.pt")
            
            # Check if already running
            if poc_process is not None and poc_process.poll() is None:
                self.send_response(409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "POC já está em execução"}).encode("utf-8"))
                return
                
            cmd = [
                "/home/usuario/.local/bin/uv", "run", "src/poc_reward.py",
                "--iters", str(iters),
                "--envs", str(envs),
                "--warm-start", str(warm_start),
                "--out", str(POC_FILE)
            ]
            
            poc_process = subprocess.Popen(cmd, cwd=str(BASE_DIR))
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "STARTED", "pid": poc_process.pid}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/poc"):
            data = {}
            if POC_FILE.exists():
                try:
                    data = json.loads(POC_FILE.read_text())
                except Exception:
                    pass
            is_running = poc_process is not None and poc_process.poll() is None
            data["is_running"] = is_running
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif self.path.startswith("/api/stats"):
            data = []
            if LOG_PATH.exists():
                for line in LOG_PATH.read_text().splitlines():
                    m = re.match(r"it\s+(\d+)\s+\|\s+([\d\.]+)k steps/s\s+\|\s+net vs chase\s+([+-]?\d+)\s+\(best\s+([+-]?\d+)\)", line)
                    if m:
                        data.append({
                            "it": int(m.group(1)),
                            "sps": float(m.group(2)),
                            "net": int(m.group(3)),
                            "best": int(m.group(4))
                        })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            super().do_GET()

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("0.0.0.0", PORT), DynamicHandler) as httpd:
    httpd.serve_forever()

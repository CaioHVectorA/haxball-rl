import re
import json
from pathlib import Path

log_path = Path("/home/usuario/.gemini/antigravity/brain/9843ad18-37de-4e8b-a25e-ee4d811d6f71/.system_generated/tasks/task-1038.log")
lines = log_path.read_text().splitlines() if log_path.exists() else []

history = []
for line in lines:
    m = re.match(r"it\s+(\d+)\s+\|\s+([\d\.]+)k steps/s\s+\|\s+net vs chase\s+([+-]?\d+)\s+\(best\s+([+-]?\d+)\)", line)
    if m:
        history.append({
            "it": int(m.group(1)),
            "sps": float(m.group(2)),
            "net": int(m.group(3)),
            "best": int(m.group(4))
        })

print(f"Total entries: {len(history)}")
if history:
    print(f"Latest: it {history[-1]['it']} | Net vs chase: {history[-1]['net']}")

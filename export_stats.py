import re
import json
from pathlib import Path

log_path = Path("/home/usuario/.gemini/antigravity/brain/9843ad18-37de-4e8b-a25e-ee4d811d6f71/.system_generated/tasks/task-1038.log")
data = []
if log_path.exists():
    for line in log_path.read_text().splitlines():
        m = re.match(r"it\s+(\d+)\s+\|\s+([\d\.]+)k steps/s\s+\|\s+net vs chase\s+([+-]?\d+)\s+\(best\s+([+-]?\d+)\)", line)
        if m:
            data.append({
                "it": int(m.group(1)),
                "sps": float(m.group(2)),
                "net": int(m.group(3)),
                "best": int(m.group(4))
            })

Path("training_stats.json").write_text(json.dumps(data, indent=2))
print(f"Exported {len(data)} points to training_stats.json")

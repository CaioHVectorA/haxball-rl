# Sprint 1: iteration 10000

`model.pt` is the best internally selected policy; `latest.pt` is the endpoint.
`resume.pt` includes Adam, schedules, references, opponent pool and RNGs.
The original `models/model.pt` is preserved. Records and intermediate checkpoints:
`experiments/sprint1_structural/gpu_continuation_10000` (ZIP parts with SHA-256 manifest).

## Web test

Pre-exported `policy.json` uses the best model; `policy_latest.json` uses the endpoint.
From the repository root in PowerShell, install the existing room dependencies:

```powershell
npm --prefix headless-bot ci
$env:POLICY_PATH = (Resolve-Path 'models/sprint1_10000/policy.json').Path
node headless-bot/bot.js
```

Supply HEADLESS_TOKEN only in the process environment, using a fresh Haxball token.
The bot prints the room link to open in the web client. No token is stored here.
Export compatibility was checked; human gameplay has not been validated.

## Warm start (a new training run, fresh optimizer and schedules)

```powershell
python src/train.py --sprint1 --warm-start models/sprint1_10000/latest.pt --device cuda --out runs/new_finetune
```

Use `model.pt` instead to start from the internally selected best policy.
For exact training continuation, use `--resume`, which preserves the saved recipe.
The 10000-iteration resume has already exhausted its original budget; extending it
requires an explicit new budget decision rather than silently changing schedules.

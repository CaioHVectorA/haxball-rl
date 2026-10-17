#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(git branch --show-current)" != "gpt" ]]; then
  echo "Expected branch gpt; no branch was changed." >&2
  exit 1
fi
# Secret files and installed dependencies remain excluded by .gitignore.
# Archived runs/checkpoints are included in artifacts/, not force-added wholesale.
git add -A
git diff --cached --check
if ! git diff --cached --quiet; then
  git commit -m "Publish complete experiment snapshot and behavior replays"
fi
git push origin HEAD:refs/heads/gpt

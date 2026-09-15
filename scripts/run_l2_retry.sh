#!/usr/bin/env bash
# Retry the supplementary L2 sweep after the data-volume study frees the GPU.
# The first attempt died with BrokenPipeError spawning DataLoader workers;
# num_workers is now 0 for this sweep.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "waiting for run_remaining.sh (data-volume) to finish"
while pgrep -f "run_remaining.sh" > /dev/null; do sleep 60; done
log "running supplementary L2 sweep"
uv run python scripts/l2_supplementary.py
log "supplementary L2 exit $?"

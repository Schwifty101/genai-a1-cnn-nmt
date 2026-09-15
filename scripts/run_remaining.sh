#!/usr/bin/env bash
# Runs every remaining experiment after the five full Q1 model runs finish.
# Sequential on purpose: concurrent jobs contend for the single MPS device.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for the five full model runs to finish"
while pgrep -f "run_q1_full_models.sh" > /dev/null; do sleep 60; done
log "full model runs done; continuing"

# 1. Q2 vanilla RNN training + evaluation (small, do it first for a fast deliverable)
if [ ! -f runs/q2/main/result.json ]; then
  log "Q2 training"
  uv run python -m src.q2_rnn.train --config configs/q2_base.yaml --run-name main \
    --split-csv results/q2_split.csv --vocab-dir results/vocab
  log "Q2 training done (exit $?)"
fi
if [ ! -f results/q2_metrics.json ]; then
  log "Q2 evaluation"
  uv run python -m src.q2_rnn.evaluate --run runs/q2/main \
    --split-csv results/q2_split.csv --vocab-dir results/vocab
  log "Q2 evaluation done (exit $?)"
fi

# 2. Supplementary L2 sweep over a range above float32 resolution.
#    The graded grid used {0,1e-5,1e-4}, which is a silent no-op at lr=1e-4.
if [ ! -f results/q1_l2_supplementary.csv ]; then
  log "supplementary L2 sweep {0, 1e-4, 1e-3, 1e-2}"
  uv run python scripts/l2_supplementary.py
  log "supplementary L2 done (exit $?)"
fi

# 3. Data-volume study at 224px on the winning config
if [ ! -f results/q1_datavolume.csv ]; then
  log "data-volume study 25/50/75/100%"
  uv run python -m src.q1_cnn.gridsearch datavolume \
    --config configs/q1_base.yaml --best results/q1_best_config.json \
    --split-manifest results/q1_split_manifest.csv \
    --out-root runs/q1/datavolume --fractions 0.25 0.5 0.75 1.0 \
    --results-csv results/q1_datavolume.csv
  log "data-volume done (exit $?)"
fi

log "ALL REMAINING EXPERIMENTS COMPLETE"

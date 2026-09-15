#!/usr/bin/env bash
# Q1 test-set evaluation + error analysis. Waits for every other experiment
# to finish first so it sees the data-volume runs too.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for run_remaining.sh to finish"
while pgrep -f "run_remaining.sh" > /dev/null; do sleep 60; done
log "continuing"

log "Q1 test-set evaluation across all five models"
uv run python -m src.q1_cnn.evaluate --all \
  --runs-root runs/q1 --split-manifest results/q1_split_manifest.csv \
  --out-csv results/q1_comparison.csv
log "evaluation exit $?"

log "Q1 error analysis on the primary model"
uv run python -m src.q1_cnn.error_analysis --run runs/q1/pneumonet_full \
  --split-manifest results/q1_split_manifest.csv --top-n 12 --out-dir results
log "error analysis exit $?"

log "Q1 hyperparameter table export"
uv run python -c "
import json, pandas as pd
from pathlib import Path
from src.q1_cnn.gridsearch import summarize_search
from src.q1_cnn.train import RunConfig
from src.q1_cnn.evaluate import dataframe_to_latex
rows = pd.read_csv('results/q1_gridsearch.csv')
best = RunConfig(**json.loads(Path('results/q1_best_config.json').read_text()))
t = summarize_search(rows, best)
t.to_csv('results/q1_hyperparam_table.csv', index=False)
dataframe_to_latex(t, Path('report/tables/q1_hyperparams.tex'),
    'Grid search: hyperparameter, range searched, and optimal value.', 'tab:q1hyper')
print(t.to_string(index=False))
"
log "ALL Q1 EVALUATION COMPLETE"

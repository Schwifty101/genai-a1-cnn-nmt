#!/usr/bin/env bash
# Train all five Q1 models at 224px on the grid-search winning configuration.
# Idempotent: a model whose result.json already exists is skipped.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1

BEST=results/q1_best_config.json
read -r BS LR EP PA DO L2 L1 NORM AUG < <(
  uv run python -c "
import json;c=json.load(open('$BEST'))
print(c['batch_size'],c['lr'],c['epochs'],c['patience'],c['dropout'],
      c['l2_lambda'],c['l1_lambda'],c['normalization'],c['augmentation'])"
)
echo "winning config: bs=$BS lr=$LR epochs=$EP patience=$PA dropout=$DO l2=$L2 l1=$L1 norm=$NORM aug=$AUG"
echo "IMAGE SIZE OVERRIDDEN TO 224 (grid proxy ran at 128)"

# Deliberate, disclosed override of the grid's epochs/patience selection.
# Stage D chose 60/10 (val macro-F1 0.9761, 49 epochs) over 30/5 (0.9757,
# 17 epochs): +0.0004 macro-F1 for ~3x the compute. We run 30/5 under an
# explicit compute budget. Both numbers are reported; the search result is
# not misrepresented.
EP=30
PA=5
echo "EPOCHS/PATIENCE OVERRIDDEN TO ${EP}/${PA} (grid selected 60/10; +0.0004 macro-F1 for 3x compute)"

for M in pneumonet vgg16_frozen vgg16_finetune resnet50_frozen resnet50_finetune; do
  OUT="runs/q1/${M}_full"
  if [ -f "$OUT/result.json" ]; then echo "[skip] $M already done"; continue; fi
  echo "=== $(date '+%H:%M:%S') training $M ==="
  uv run python -m src.q1_cnn.train --config configs/q1_base.yaml \
    --model "$M" --run-name "${M}_full" \
    --set image_size=224 --set batch_size="$BS" --set lr="$LR" \
    --set epochs="$EP" --set patience="$PA" --set dropout="$DO" \
    --set l2_lambda="$L2" --set l1_lambda="$L1" \
    --set normalization="$NORM" --set augmentation="$AUG"
  echo "=== $(date '+%H:%M:%S') finished $M (exit $?) ==="
done
echo "ALL FULL RUNS COMPLETE $(date '+%H:%M:%S')"

"""Supplementary L2 sweep over a range that is actually representable in float32.

The graded grid searched l2_lambda in {0, 1e-5, 1e-4}. At lr=1e-4 the AdamW
decoupled update p <- p - lr*lambda*p is a relative change of 1e-8 or less,
below float32 epsilon (~1.19e-7), so every one of those values is a silent
no-op and the stage returned bit-identical results. This sweep re-runs the
same stage over {0, 1e-4, 1e-3, 1e-2} so the reported L2 row is informative.
"""
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.common.device import pick_device
from src.q1_cnn.gridsearch import run_stage
from src.q1_cnn.train import RunConfig

best = json.loads(Path("results/q1_best_config.json").read_text())
base = replace(RunConfig(**best), image_size=128, epochs=12)
split_df = pd.read_csv("results/q1_split_manifest.csv")

stage = {"name": "C2_l2_supplementary", "grid": {"l2_lambda": [0.0, 1e-4, 1e-3, 1e-2]}}
best_params, rows = run_stage(stage, base, split_df, Path("runs/q1/grid"), pick_device())

df = pd.DataFrame(rows)
df.to_csv("results/q1_l2_supplementary.csv", index=False)
print(df.to_string(index=False))
print("best:", best_params)

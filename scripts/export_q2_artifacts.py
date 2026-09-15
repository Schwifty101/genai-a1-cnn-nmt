"""Export Q2 run artifacts into results/ under the names the report expects.

q2_rnn.evaluate writes into runs/q2/<run>/result.json and samples.json.
The report contract wants results/q2_metrics.json, results/q2_samples.csv,
results/q2_error_taxonomy.json and results/figs/q2_curves.png. This copies
rather than re-computes, so the numbers are identical by construction.
"""
import json, shutil
from pathlib import Path

import pandas as pd

run = Path("runs/q2/main")
res = Path("results"); (res / "figs").mkdir(parents=True, exist_ok=True)

d = json.loads((run / "result.json").read_text())
tm = d["test_metrics"]

metrics = {
    "best_val_loss": d["best_val_loss"],
    "best_val_ppl": d["best_val_ppl"],
    "test_bleu": tm["bleu"],
    "n_test": tm["n"],
    "epochs_run": len(d["history"]["train_loss"]),
    "params_total": d["params_total"],
    "train_seconds": d["train_seconds"],
}
(res / "q2_metrics.json").write_text(json.dumps(metrics, indent=2))

taxonomy = {k: tm[k] for k in
            ("n", "unk_rate", "repetition_rate", "empty_rate",
             "mean_len_ratio", "bleu_by_src_len_bucket")}
(res / "q2_error_taxonomy.json").write_text(json.dumps(taxonomy, indent=2))

samples = json.loads((run / "samples.json").read_text())
pd.DataFrame(samples).to_csv(res / "q2_samples.csv", index=False)

if (run / "curves.png").exists():
    shutil.copy(run / "curves.png", res / "figs" / "q2_curves.png")

print(json.dumps(metrics, indent=2))
print(f"samples exported: {len(samples)}")

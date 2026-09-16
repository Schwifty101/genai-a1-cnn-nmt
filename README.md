# GenAI A1 — Chest X-Ray Classification & English→Urdu Translation

## 1. What this is

This repository implements two independent deep-learning studies for the GenAI Assignment 1 coursework: **Q1**, a custom convolutional neural network (`PneumoNet`) that classifies chest X-rays into `COVID19` / `NORMAL` / `PNEUMONIA` and is benchmarked against fine-tuned and frozen VGG16 and ResNet50 baselines, and **Q2**, a vanilla RNN (GRU) encoder–decoder that translates English sentences into Urdu. The full written report, with methodology, results, and error analysis for both studies, is at [`report/main.pdf`](report/main.pdf).

## 2. The seed: 42

Every stochastic step in this repository — dataset shuffling, train/val/test splits, weight initialization, dropout, data-loader ordering, vocabulary construction — is seeded from a single constant, `SEED = 42` (`src/common/seed.py`).

**Reproducibility caveat:** splits, vocabularies, and hyperparameter selections reproduce **exactly** on any machine. Metric *values* (accuracy, F1, BLEU, loss) reproduce to roughly **±0.3%** on Apple Silicon, because Apple's MPS backend kernels are not bitwise-deterministic across runs even with a fixed seed. On CUDA or CPU, results should be closer to exact.

## 3. Headline results

### Q1 — chest X-ray classification (test set)

| Model | Params | Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|---:|
| **pneumonet** (custom, ours) | 1,239,779 | 0.9797 | 0.9794 | 0.9982 |
| vgg16_finetune | 134,272,835 | 0.9765 | 0.9759 | 0.9980 |
| resnet50_finetune | 23,514,179 | 0.9734 | 0.9753 | 0.9949 |
| resnet50_frozen | 23,514,179 | 0.9405 | 0.9407 | 0.9925 |
| vgg16_frozen | 134,272,835 | 0.9249 | 0.9160 | 0.9836 |

Source: [`results/q1_comparison.csv`](results/q1_comparison.csv).

The custom 1.24M-parameter CNN beats every pre-trained baseline, including fine-tuned VGG16 at roughly **108×** the parameter count (134.3M / 1.24M ≈ 108).

### Q2 — English→Urdu translation (test set)

| Metric | Value |
|---|---:|
| BLEU (844 test pairs) | 2.45 |
| Best validation loss | 4.2207 |
| Best validation perplexity | 68.08 |
| Model parameters | 4,770,425 |

Source: [`results/q2_metrics.json`](results/q2_metrics.json).

## 4. Setup

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev]"
```

Requires Python 3.13, `torch==2.14.0`, `torchvision==0.29.0`. Training was run on Apple Silicon (M1 Pro) via the MPS backend; the code also runs on CPU/CUDA.

## 5. Kaggle credentials

Both datasets are downloaded automatically from Kaggle on first use. The Kaggle API needs credentials in **one** of:

- `~/.kaggle/access_token`
- `~/.kaggle/kaggle.json`

**Never commit credentials to this repository.** Neither path lives under the project directory, and `git ls-files` is verified clean of any credential file before every commit (see Verification, below).

Datasets used:

- Q1: [`prashant268/chest-xray-covid19-pneumonia`](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
- Q2: [`muhammadnoman76/translation-dataset`](https://www.kaggle.com/datasets/muhammadnoman76/translation-dataset)

Both `download_dataset()` steps (`src/q1_cnn/data_prep.py`, `src/q2_rnn/data_prep.py`) check whether the data is already present under `data/q1` / `data/q2` and skip the network call entirely if so — safe to re-run.

## 6. Reproducing everything

Run in order (all commands assume the repo root and `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Q1: download, integrity-check, dedupe, and split (a few minutes)
uv run python -m src.q1_cnn.data_prep
uv run python -m src.q1_cnn.splits

# Q2: download, clean, and split (a few minutes)
uv run python -m src.q2_rnn.data_prep --config configs/q2_base.yaml
uv run python -m src.q2_rnn.splits --config configs/q2_base.yaml

# Q1 staged grid search: 30 configs at 128px proxy resolution (~8 hours)
uv run python -m src.q1_cnn.gridsearch search --config configs/q1_base.yaml \
  --split-manifest results/q1_split_manifest.csv \
  --out-root runs/q1/grid --results-csv results/q1_gridsearch.csv

# Five full Q1 models at 224px on the winning config (~72 minutes)
bash scripts/run_q1_full_models.sh

# Q2 training (~6 minutes)
uv run python -m src.q2_rnn.train --config configs/q2_base.yaml --run-name main \
  --split-csv results/q2_split.csv --vocab-dir results/vocab

# Data-volume study, Q2 evaluation, supplementary L2 sweep
bash scripts/run_remaining.sh

# Q1 test-set evaluation, error analysis, hyperparameter table export
bash scripts/run_q1_eval.sh
```

`scripts/run_q1_full_models.sh`, `scripts/run_remaining.sh`, and `scripts/run_q1_eval.sh` are all **idempotent**: each step checks whether its output (typically a `result.json`) already exists and skips it if so. An interrupted pipeline can simply be re-invoked and it resumes rather than restarts from scratch.

## 7. Grid search

The Q1 hyperparameter search is a **named deliverable**, and it is deliberately not an exhaustive cartesian product over all nine hyperparameters — that space is too large to search jointly at 224px under a reasonable compute budget. Instead it is a **staged coordinate descent**: each stage sweeps a small grid at a cheap proxy resolution (128px, capped at 12 epochs), selects the winner by validation macro-F1, and hands that winning value to the next stage as its fixed starting point.

| Stage | Sweeps | Configs |
|---|---|---:|
| A — preprocessing | normalization × augmentation | 9 |
| B — optimization | learning rate × batch size | 9 |
| C1 — dropout | dropout | 3 |
| C2 — L2 | L2 weight decay | 3 |
| C3 — L1 | L1 penalty | 2 |
| D — schedule | epochs × patience | 4 |
| **Total** | | **30** |

Command:

```bash
uv run python -m src.q1_cnn.gridsearch search --config configs/q1_base.yaml \
  --split-manifest results/q1_split_manifest.csv \
  --out-root runs/q1/grid --results-csv results/q1_gridsearch.csv
```

Results land in [`results/q1_gridsearch.csv`](results/q1_gridsearch.csv) (all 30 proxy runs), [`results/q1_best_config.json`](results/q1_best_config.json) (the winning config), and [`results/q1_hyperparam_table.csv`](results/q1_hyperparam_table.csv) (one row per hyperparameter: range searched vs. optimal value).

**Winning configuration:** `batch_size=16`, `lr=1e-4`, `dropout=0.3`, `L2=0.0`, `L1=0.0`, `normalization=imagenet`, `augmentation=none`, `epochs/patience=60/10`.

**Disclosed deviation:** the five full-size runs (`scripts/run_q1_full_models.sh`) use `epochs/patience=30/5`, not the search's `60/10` selection, under a stated compute budget. The measured difference in Stage D was **+0.0004 macro-F1** for roughly **3×** the compute (0.9761 val macro-F1 at 49 epochs for 60/10, vs. 0.9757 at 17 epochs for 30/5). This trade-off is disclosed in the script itself and in the report; it is not a silent substitution.

## 8. Streamlit demo

```bash
uv run streamlit run app.py
```

The app shows the assignment's required outputs for both studies (grid-search results, hyperparameter table, comparison table, data-volume curves, training curves, error analysis) alongside **live inference**: upload a chest X-ray to classify it with the trained `PneumoNet`, or type an English sentence to translate it with the trained Q2 RNN.

## 9. Repository map

| Path | Contents |
|---|---|
| `src/common` | Shared seed, device selection, metrics, plotting, and training-loop utilities used by both studies |
| `src/q1_cnn` | Q1 data prep, splits, augmentation, datasets, models (PneumoNet + baselines), training, grid search, evaluation, error analysis |
| `src/q2_rnn` | Q2 data prep, splits, tokenizer/vocab, dataset, RNN encoder–decoder model, training, evaluation |
| `configs` | Base YAML configs (`q1_base.yaml`, `q2_base.yaml`) read by training and grid-search entry points |
| `scripts` | Unattended, idempotent runner scripts for full model training, remaining experiments, evaluation, and report building |
| `results` | All numeric artifacts (CSVs, JSONs, figures, vocab) that the report and Streamlit app read from |
| `report` | LaTeX source and the built `main.pdf` (IEEE two-column report) |
| `tests` | Unit tests for both pipelines and shared utilities |
| `docs` | Planning documents, specs, and other project documentation |

## 10. Tests

```bash
uv run pytest tests/
```

185 tests pass as of this writing (verify locally — this count is not guaranteed to stay current as the codebase evolves).

## 11. Known limitations

- The Q2 corpus is 9,103 pairs of predominantly Biblical, King-James-register English, not the ~24k general-domain corpus the assignment describes. BLEU here is not comparable to published general-domain NMT results.
- The L2 range searched (`{0, 1e-5, 1e-4}`) is below float32 resolution at `lr=1e-4`, so L2 regularization had no measurable effect in this search. See the report's Section IV for the supplementary sweep that confirms this.
- The grid search is staged (coordinate descent), not exhaustive — see Section 7 above.
- Vanilla-RNN translation quality degrades sharply with sentence length (BLEU 8.94 at 6–10 source tokens, down to 1.75 at 16+ tokens), which is the expected consequence of the architecture's fixed-context-vector bottleneck.

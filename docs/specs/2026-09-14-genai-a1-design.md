# GenAI A1 — Design Spec

Date: 2026-09-14
Author: Soban (orchestrated with Claude Code)

Two independent deep-learning studies delivered as one repo and one IEEE two-column report:

- **Q1** — Chest X-ray classification with a custom CNN as primary model, pre-trained networks as comparison baselines only.
- **Q2** — English→Urdu neural machine translation with a vanilla RNN encoder–decoder (no LSTM/GRU/attention/Transformer).

## 0. Global Constraints

These bind every task. Violating any one is a spec failure.

- **SEED = 42**, set for `random`, `numpy`, `torch`, `torch.mps`, DataLoader `generator` and `worker_init_fn`. Declared once in `src/common/seed.py::set_seed`.
- **Splits are 80/10/10**, stratified where labels exist, produced once and persisted to CSV manifests. Every downstream script reads the manifest; no script re-splits.
- **Framework is PyTorch** (2.14.0) with MPS. Device selection lives only in `src/common/device.py`.
- **Q1 primary model is the custom CNN.** VGG16 / ResNet50 appear only as comparison baselines and are never presented as the primary result.
- **Q2 uses `nn.RNN` only.** No `nn.LSTM`, `nn.GRU`, attention, or Transformer components anywhere in `src/q2_rnn/`.
- **Augmentation applies to the training split only.** Val and test transforms are resize + normalize.
- **No fabricated numbers.** Every table and figure in the report is generated from a file in `results/`. If a run did not happen, it does not appear.
- **Python 3.13** via `uv`. Dependencies pinned in `pyproject.toml`.
- **MPS determinism caveat** must be stated in the README and report: splits, vocabularies and hyperparameter selections reproduce exactly; metric values reproduce to roughly +/-0.3% because MPS kernels are not bitwise-deterministic.

## 1. Repository Layout

```
genai-a1/
  pyproject.toml            uv / python 3.13, pinned deps
  README.md                 seed, setup, exact repro commands, grid-search usage
  configs/q1_base.yaml      Q1 defaults
  configs/q2_base.yaml      Q2 defaults
  src/common/
    seed.py                 set_seed(42), worker_init_fn, generator
    device.py               pick_device() -> mps | cpu
    metrics.py              accuracy, per-class + macro P/R/F1, one-vs-rest AUC, confusion matrix
    plots.py                training curves, confusion-matrix heatmap, ROC, montages
    trainer.py              generic loop: loss, optimizer, checkpointing, early stopping, history JSON
  src/q1_cnn/
    data_prep.py            download, integrity scan, dedup, manifest CSV
    splits.py               stratified 80/10/10, per-class counts, leakage assertion
    datasets.py             Dataset + train/eval transform pipelines
    augment.py              five augmentations + before/after figure
    models.py               PneumoNet, vgg16/resnet50 frozen + finetuned builders
    train.py                CLI training entry point
    gridsearch.py           staged coordinate search, 30 configs
    datasize.py             25/50/75/100% study
    evaluate.py             test metrics, confusion matrix, ROC, sample preds, comparative table
    error_analysis.py       misclassified montage + statistics
  src/q2_rnn/
    data_prep.py            xlsx load, normalization, dedup, length filter
    splits.py               80/10/10 + overlap assertion
    tokenizer.py            word-level vocab with specials
    dataset.py              encode, pad, mask, collate
    models.py               vanilla RNN encoder / decoder / seq2seq
    train.py                masked CE, Adam, grad clipping, checkpoints
    evaluate.py             greedy decode, perplexity, BLEU, samples, error taxonomy
  scripts/run_q1_all.sh     full Q1 pipeline
  scripts/run_q2_all.sh     full Q2 pipeline
  results/                  committed metrics JSON/CSV + figures
  report/main.tex           IEEEtran two-column, sections I-VII
```

## 2. Q1 — Dataset and Preprocessing

**Dataset:** `prashant268/chest-xray-covid19-pneumonia`, three classes (COVID19, NORMAL, PNEUMONIA), layout `Data/{train,test}/{CLASS}/`. The official train and test directories are **pooled** and re-split, because the assignment mandates a reproducible 80/10/10 split.

**Pipeline, in order:**

1. **Integrity** — `PIL.Image.verify()` then reopen and load. Corrupt or truncated files are dropped and counted.
2. **Duplicates** — exact MD5 first, then a near-duplicate pass. Both counts reported. Dedup runs **before** splitting, which is what makes the leakage guarantee real.

   The near-duplicate rule is **contrast-normalized 32x32 thumbnail correlation >= 0.99**, not perceptual-hash Hamming distance. This was changed after measurement, and the measurement belongs in the report: across this dataset the *median* nearest-neighbour correlation is 0.926, because every chest radiograph shares the same gross anatomy and framing. A dHash-8 rule at Hamming <= 3 therefore flagged 1,570 images (25% of the data) as near-duplicates, of which a 40-pair audit found **zero** genuinely near-identical pairs, an average flagged-pair correlation of 0.842 against a random-pair baseline of 0.605, and 10 of 40 merges spanning two different diagnostic classes. The correlation rule at 0.99 sits in the genuine-duplicate tail (p99 = 0.970, p99.5 = 0.987) and removes 14 near-duplicates plus 34 exact duplicates, 0.75% of the dataset. `dhash` is still computed and stored in the manifest as a cheap blocking key and for reporting.
3. **Channels** — force grayscale, then replicate to 3 channels. Chest X-rays are single-channel; ImageNet backbones require 3; giving the custom CNN the same 3-channel input is what makes "identical conditions" literally true.
4. **Resize** — 224x224 bilinear for all full runs. Grid-search proxy runs use 128x128 for speed.
5. **Normalization** — a swept variable: ImageNet mean/std, plain [0,1], or dataset mean/std.

**Split:** stratified 80/10/10, seed 42, persisted as `results/q1_split_manifest.csv`. Deliverables: per-class count table and a printed hash-intersection assertion proving the three splits are disjoint.

**Class imbalance:** the three classes are strongly skewed. Addressed with inverse-frequency weighted cross-entropy, with `WeightedRandomSampler` available as a grid alternative, and **macro-averaged F1 as the model-selection metric** throughout.

**Augmentation — training split only** (five techniques, exceeding the required four):

| Technique | Setting |
|---|---|
| Rotation | +/- 10 degrees |
| Horizontal flip | p = 0.5 |
| Zoom | RandomResizedCrop, scale 0.85-1.0 |
| Shift | RandomAffine translate +/- 10% |
| Brightness | ColorJitter brightness +/- 20% |

A before/after sample grid is written to `results/figs/q1_augmentation.png`. The report notes that horizontal flip is anatomically questionable for chest radiographs (situs/cardiac position); it is included because the assignment requires flipping, and its actual cost is measured by the augmentation-policy grid variable rather than asserted.

## 3. Q1 — Methodology

**Primary model — PneumoNet** (custom, roughly 1.5M parameters):

Four blocks of `[Conv3x3 -> BatchNorm -> ReLU] x2 -> MaxPool` at widths 32 / 64 / 128 / 256, then global average pooling -> Dropout -> FC-256 -> Dropout -> FC-3.

Design rationale to state in the report: batch norm stabilizes training on a small medical dataset; stacked 3x3 convolutions buy a 5x5 effective receptive field more cheaply than a single 5x5; global average pooling replaces flatten-and-dense, removing the parameter explosion that drives overfitting on ~6k images; dropout is the explicitly tunable regularizer that the grid search sweeps.

The layer table (output shapes, per-layer parameter counts) is generated with `torchinfo` and exported to LaTeX. It is never hand-typed.

**Comparison baselines** — VGG16 and ResNet50, each in two settings:

- *Frozen*: backbone in eval mode with gradients disabled, only a fresh classifier head trains.
- *Fine-tuned*: final block unfrozen at a 10x lower learning rate.

Identical conditions across all five models: same split manifest, same 224x224 input, same augmentation policy, same optimizer family, same early-stopping rule, same selection metric.

## 4. Q1 — Experimental Setup

**Training pipeline:** weighted cross-entropy, AdamW, `ReduceLROnPlateau`, checkpoint on best validation macro-F1, early stopping on validation macro-F1. Per-epoch history persisted to JSON and plotted as training/validation loss and accuracy curves, with convergence discussed against those curves.

**Grid search — staged coordinate descent, 30 proxy configs** at 128px / 12 epochs, followed by one full 224px run on the winning configuration:

| Stage | Swept | Configs |
|---|---|---|
| A | normalization {imagenet, 0-1, dataset} x augmentation policy {none, light, heavy} | 9 |
| B | learning rate {1e-4, 3e-4, 1e-3} x batch size {16, 32, 64} | 9 |
| C1 | dropout {0.2, 0.3, 0.5} | 3 |
| C2 | L2 lambda {0, 1e-5, 1e-4} | 3 |
| C3 | L1 lambda {0, 1e-5} | 2 |
| D | epochs {30, 60} x early-stopping patience {5, 10} | 4 |

Each stage fixes the previous stages' winners. Every hyperparameter named in the assignment gets a **hyperparameter / range / optimal value** row. The report states plainly that the search is greedy-staged rather than exhaustive cartesian, and gives the compute reason. Results are written to `results/q1_gridsearch.csv`.

**Data-volume study:** stratified 25 / 50 / 75 / 100% subsamples of the training split (seed 42), winning configuration, evaluated on the untouched test split. Plot: test accuracy and macro-F1 against training-set size.

## 5. Q1 — Results

Required outputs, all generated:

- Confusion matrix (counts and row-normalized)
- Overall accuracy
- Per-class and macro precision / recall / F1
- One-vs-rest AUC-ROC, three curves plus macro average
- Sample-prediction grid with true label, predicted label, confidence
- One comparative table across all five models: parameters, epoch time, accuracy, macro-F1, macro-AUC

## 6. Q1 — Error Analysis

At least twelve misclassified test images, selected as the highest-confidence errors, rendered as a labeled montage. Discussion covers failure patterns, overfitting versus underfitting read off the training curves, generalization, the measured effect of regularization taken from grid stages C1-C3, tuning observations, and future work.

## 7. Q2 — Dataset and Preprocessing

**Dataset:** `muhammadnoman76/translation-dataset`, a single `english_to_urdu_dataset.xlsx` of roughly 24k parallel pairs.

**Pipeline:** load with openpyxl; drop null and empty rows; Unicode **NFC** normalization plus Arabic-to-Urdu character folding (ي→ی, ك→ک, ه→ہ); optional diacritic stripping, documented either way; whitespace collapse; punctuation separated by spaces; English lowercased; exact duplicate pairs removed; length filtered to the 95th percentile with a source/target length-ratio guard. Length-distribution histograms before and after filtering.

**Splits:** 80/10/10, seed 42, persisted, with a hash-based assertion that no pair appears in two splits.

**Tokenization:** word-level on normalized whitespace, `min_freq = 2`. Specials fixed at `<pad>=0`, `<sos>=1`, `<eos>=2`, `<unk>=3`. Both vocabulary sizes and the resulting OOV rate are reported.

## 8. Q2 — Methodology

**Encoder:** `Embedding(V_en, 256)` -> `nn.RNN(256, 512, nonlinearity='tanh', batch_first=True)`.
**Decoder:** `Embedding(V_ur, 256)` -> `nn.RNN(256, 512, batch_first=True)` initialized from the encoder's final hidden state -> `Linear(512, V_ur)`.

Layer table with output shapes and parameter counts generated with `torchinfo`.

## 9. Q2 — Sequence Encoding, Padding, Batching

Tokenized sentences are converted to integer sequences and right-padded. Boolean masks are generated for padded positions. The encoder uses `pack_padded_sequence` so padding never contaminates the final hidden state. The collate function emits three tensors:

- `enc_in` — padded source
- `dec_in` — `[<sos>] + target[:-1]`
- `dec_target` — `target + [<eos>]`

## 10. Q2 — Training

Masked cross-entropy with `ignore_index=0`, Adam, **gradient clipping at 1.0** (vanilla RNNs explode without it), fixed teacher-forcing ratio of 0.5, checkpoint on best validation loss, early stopping. Training and validation loss curves are plotted and convergence discussed.

## 11. Q2 — Results and Error Analysis

The assignment's Q2 task list stops at training, but the mandated report structure requires sections V and VI for both studies. Q2 therefore reports validation and test loss and perplexity, greedy-decoded sample translations, corpus BLEU via sacrebleu, and a failure taxonomy: repetition loops, `<unk>` proliferation, and quality collapse beyond roughly 15 tokens. This degradation is the expected vanilla-RNN outcome given the no-LSTM/GRU/attention constraint, and it is the honest lead-in to future work rather than a result to apologize for.

## 12. Report

`report/main.tex`, IEEEtran two-column, target 6 pages and hard cap at 6. Sections I-VII in the mandated sequence, with Q1 and Q2 as parallel subsections inside each section so the two studies read side by side. All tables and figures are `\input` or `\includegraphics` of generated files. Built with tectonic.

## 13. Delivery

Private GitHub repository created with `gh`, pushed to `main`. README documents the seed, environment setup, and the exact command sequence to reproduce every number, including the grid-search scripts.

## 14. Accepted Risks

- VGG16 fine-tuning at 224px with batch 32 may exceed 16 GB unified memory. Fallback: batch 16 plus gradient accumulation, documented in the report rather than hidden.
- MPS is not bitwise-deterministic. Documented as above.
- Total wall clock is estimated at 8-12 hours, dominated by unattended background training.

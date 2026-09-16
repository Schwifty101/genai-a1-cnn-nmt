# Streamlit viewer

A read-only viewer and live-inference demo for the two completed studies
(Q1 chest X-ray CNN, Q2 English→Urdu NMT). It only reads finished artifacts
under `results/` and `runs/` and loads the two trained checkpoints for
inference — it does not retrain or modify anything.

## Run

```bash
uv run streamlit run app.py
```

This opens the app in your browser (default `http://localhost:8501`).

## What it shows

- **Q1 tab**: dataset/cleaning report, train/val/test split counts,
  augmentation examples, PneumoNet's layer table and parameter count,
  training curves, grid-search hyperparameter table and data-volume study,
  the model comparison table plus confusion matrices / ROC / sample
  predictions, error-analysis figure and stats, and a live file-uploader
  that runs an uploaded chest X-ray through the trained PneumoNet
  checkpoint (`runs/q1/pneumonet_full/best.pt`).
- **Q2 tab**: dataset/cleaning report and length-distribution figure,
  split sizes, tokenization/vocab stats, the encoder–decoder layer table
  and parameter count, training curves with best val loss/perplexity,
  test BLEU, error taxonomy, example translations, and a live text box
  that greedy-decodes an English sentence through the trained vanilla RNN
  checkpoint (`runs/q2/main/best.pt`).

## Notes

- Model and vocabulary loads are wrapped in `@st.cache_resource`, and
  artifact reads in `@st.cache_data`, so checkpoints are loaded once per
  server process, not on every interaction.
- Both live-inference code paths run on CPU and are wrapped in
  try/except; a missing artifact shows a warning banner instead of
  crashing the app.

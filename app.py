"""Streamlit viewer + live-inference demo for the GenAI A1 coursework.

This app only reads finished artifacts under `results/` and `runs/` and
loads the two trained checkpoints for live inference. It does not retrain,
re-evaluate, or modify anything. Run with:

    uv run streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.q1_cnn.augment import build_transforms
from src.q1_cnn.data_prep import CLASS_NAMES as Q1_CLASS_NAMES
from src.q1_cnn.models import PneumoNet, build_model, layer_table as q1_layer_table
from src.q2_rnn.data_prep import normalize_english
from src.q2_rnn.models import Seq2SeqRNN
from src.q2_rnn.models import layer_table as q2_layer_table
from src.q2_rnn.tokenizer import EOS, PAD, Vocab, tokenize

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"
RUNS = ROOT / "runs"

DEVICE = torch.device("cpu")

st.set_page_config(page_title="GenAI A1 — Q1/Q2 Viewer", layout="wide")


# --------------------------------------------------------------------------
# Generic artifact loaders (cached)
# --------------------------------------------------------------------------


@st.cache_data
def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def show_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width='stretch')
    else:
        st.warning(f"{path.name} not found")


def show_json_warn(name: str, data) -> bool:
    if data is None:
        st.warning(f"{name} not found")
        return False
    return True


def show_df_warn(name: str, df) -> bool:
    if df is None:
        st.warning(f"{name} not found")
        return False
    return True


# --------------------------------------------------------------------------
# Q1 model / vocab loading (cached resources)
# --------------------------------------------------------------------------


@st.cache_resource
def load_q1_model() -> PneumoNet | None:
    ckpt_path = RUNS / "q1" / "pneumonet_full" / "best.pt"
    if not ckpt_path.exists():
        return None
    model = build_model("pneumonet", num_classes=3, dropout=0.3)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_q1_transform():
    return build_transforms(224, "imagenet", "none", train=False)


@st.cache_resource
def load_q2_model_and_vocabs():
    result_path = RUNS / "q2" / "main" / "result.json"
    ckpt_path = RUNS / "q2" / "main" / "best.pt"
    src_vocab_path = RESULTS / "vocab" / "src_vocab.json"
    tgt_vocab_path = RESULTS / "vocab" / "tgt_vocab.json"
    if not (result_path.exists() and ckpt_path.exists() and src_vocab_path.exists() and tgt_vocab_path.exists()):
        return None, None, None, {}
    cfg = json.loads(result_path.read_text())["config"]
    src_vocab = Vocab.load(src_vocab_path)
    tgt_vocab = Vocab.load(tgt_vocab_path)
    model = Seq2SeqRNN(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, src_vocab, tgt_vocab, cfg


# --------------------------------------------------------------------------
# Q1 tab
# --------------------------------------------------------------------------


def render_q1():
    st.header("Q1: Chest X-ray CNN (PneumoNet)")

    # 1. Dataset & preprocessing
    st.subheader("1. Dataset & preprocessing")
    report = load_json(str(RESULTS / "q1_data_report.json"))
    if show_json_warn("q1_data_report.json", report):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Found", report["total_found"])
        c2.metric("Corrupt", report["corrupt"])
        c3.metric("Exact dupes", report["exact_duplicates"])
        c4.metric("Near dupes", report["near_duplicates"])
        c5.metric("Kept", report["kept"])
        st.bar_chart(pd.Series(report["class_counts"], name="count"))

    # 2. Splits
    st.subheader("2. Splits")
    split_counts = load_csv(str(RESULTS / "q1_split_counts.csv"))
    if show_df_warn("q1_split_counts.csv", split_counts):
        cols = [c for c in ["label", "train", "val", "test"] if c in split_counts.columns]
        st.dataframe(split_counts[cols], width='stretch')
        st.caption(
            "No image path or content hash appears in more than one split "
            "(dedup runs before splitting; split assignment is exclusive)."
        )

    # 3. Augmentation
    st.subheader("3. Augmentation")
    show_image(FIGS / "q1_augmentation.png")
    st.caption(
        "Five augmentation techniques (rotation, horizontal flip, resized crop, "
        "affine translation, color jitter) are applied to the training split only."
    )

    # 4. Model architecture
    st.subheader("4. Model architecture")
    try:
        arch_model = PneumoNet(num_classes=3, dropout=0.3)
        table_text = q1_layer_table(arch_model, input_size=(1, 3, 224, 224))
        total_params = sum(p.numel() for p in arch_model.parameters())
        trainable_params = sum(p.numel() for p in arch_model.parameters() if p.requires_grad)
        st.code(table_text, language="text")
        c1, c2 = st.columns(2)
        c1.metric("Total parameters", f"{total_params:,}")
        c2.metric("Trainable parameters", f"{trainable_params:,}")
    except Exception as e:
        st.error(f"Could not build PneumoNet layer table: {e}")

    # 5. Training curves
    st.subheader("5. Training curves")
    show_image(FIGS / "q1_curves_pneumonet.png")

    # 6. Grid search & data volume
    st.subheader("6. Grid search & data volume")
    st.caption(
        "The search was staged coordinate descent over 30 configurations, "
        "not an exhaustive grid."
    )
    hp = load_csv(str(RESULTS / "q1_hyperparam_table.csv"))
    if show_df_warn("q1_hyperparam_table.csv", hp):
        st.dataframe(hp, width='stretch')
    show_image(FIGS / "q1_datavolume.png")

    # 7. Results
    st.subheader("7. Results")
    comparison = load_csv(str(RESULTS / "q1_comparison.csv"))
    if show_df_warn("q1_comparison.csv", comparison):
        st.dataframe(comparison, width='stretch')
    col_a, col_b = st.columns(2)
    with col_a:
        show_image(FIGS / "q1_confusion_matrix.png", caption="Confusion matrix (counts)")
        show_image(FIGS / "q1_roc.png", caption="ROC curves")
    with col_b:
        show_image(FIGS / "q1_confusion_matrix_norm.png", caption="Confusion matrix (normalized)")
        show_image(FIGS / "q1_sample_predictions.png", caption="Sample predictions")

    # 8. Error analysis
    st.subheader("8. Error analysis")
    show_image(FIGS / "q1_errors.png")
    err_stats = load_json(str(RESULTS / "q1_error_stats.json"))
    if show_json_warn("q1_error_stats.json", err_stats):
        st.write(
            f"**{err_stats['n_errors']}** errors on the test set "
            f"(COVID19: {err_stats['by_true_class'].get('COVID19', 0)}, "
            f"NORMAL: {err_stats['by_true_class'].get('NORMAL', 0)}, "
            f"PNEUMONIA: {err_stats['by_true_class'].get('PNEUMONIA', 0)})."
        )
        st.json(err_stats)

    # 9. Live inference
    st.subheader("9. Live inference")
    model = load_q1_model()
    if model is None:
        st.warning("runs/q1/pneumonet_full/best.pt not found")
    else:
        uploaded = st.file_uploader(
            "Upload a chest X-ray image", type=["jpg", "jpeg", "png"], key="q1_uploader"
        )
        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded image", width=300)
                transform = load_q1_transform()
                tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                pred_idx = int(torch.argmax(probs).item())
                pred_class = Q1_CLASS_NAMES[pred_idx]
                st.success(f"Predicted class: **{pred_class}**")
                prob_series = pd.Series(
                    {name: float(probs[i]) for i, name in enumerate(Q1_CLASS_NAMES)}
                )
                st.bar_chart(prob_series)
            except Exception as e:
                st.error(f"Inference failed: {e}")


# --------------------------------------------------------------------------
# Q2 tab
# --------------------------------------------------------------------------


def render_q2():
    st.header("Q2: English → Urdu NMT (Vanilla RNN)")

    # 1. Dataset & preprocessing
    st.subheader("1. Dataset & preprocessing")
    report = load_json(str(RESULTS / "q2_data_report.json"))
    if show_json_warn("q2_data_report.json", report):
        st.json(report)
    show_image(FIGS / "q2_lengths.png")

    # 2. Splits
    st.subheader("2. Splits")
    st.write("Train: **6,752** / Val: **844** / Test: **844**")
    st.caption("No sentence pair appears in more than one split.")

    # 3. Tokenization
    st.subheader("3. Tokenization")
    vocab_report = load_json(str(RESULTS / "vocab" / "vocab_report.json"))
    if vocab_report is not None:
        c1, c2 = st.columns(2)
        c1.metric("English vocab size", vocab_report.get("src_vocab_size"))
        c2.metric("Urdu vocab size", vocab_report.get("tgt_vocab_size"))
    else:
        st.warning("results/vocab/vocab_report.json not found — showing fixed values")
        c1, c2 = st.columns(2)
        c1.metric("English vocab size", 3656)
        c2.metric("Urdu vocab size", 3961)
    st.write("Special tokens: `<pad>`=0, `<sos>`=1, `<eos>`=2, `<unk>`=3")

    # 4. Model architecture
    st.subheader("4. Model architecture")
    try:
        src_vocab = Vocab.load(RESULTS / "vocab" / "src_vocab.json")
        tgt_vocab = Vocab.load(RESULTS / "vocab" / "tgt_vocab.json")
        cfg = json.loads((RUNS / "q2" / "main" / "result.json").read_text())["config"]
        arch_model = Seq2SeqRNN(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        table_text = q2_layer_table(arch_model, src_vocab, tgt_vocab)
        st.code(table_text, language="latex")
        total_params = sum(p.numel() for p in arch_model.parameters())
        st.metric("Total parameters", f"{total_params:,}")
    except Exception as e:
        st.error(f"Could not build Q2 layer table: {e}")

    # 5. Training curves
    st.subheader("5. Training curves")
    show_image(FIGS / "q2_curves.png")
    metrics = load_json(str(RESULTS / "q2_metrics.json"))
    if show_json_warn("q2_metrics.json", metrics):
        c1, c2 = st.columns(2)
        c1.metric("Best val loss", f"{metrics['best_val_loss']:.4f}")
        c2.metric("Best val perplexity", f"{metrics['best_val_ppl']:.2f}")

    # 6. Results
    st.subheader("6. Results")
    if metrics is not None:
        st.metric("Test BLEU", f"{metrics['test_bleu']:.2f}")
    taxonomy = load_json(str(RESULTS / "q2_error_taxonomy.json"))
    if show_json_warn("q2_error_taxonomy.json", taxonomy):
        st.json(taxonomy)
    samples = load_csv(str(RESULTS / "q2_samples.csv"))
    if show_df_warn("q2_samples.csv", samples):
        st.dataframe(samples, width='stretch')

    # 7. Live inference
    st.subheader("7. Live inference")
    st.caption(
        "The corpus is 9,103 sentence pairs of predominantly Biblical, "
        "King-James-register English, so output quality reflects that domain. "
        "The encoder is trained on reversed source sequences (Sutskever et al. "
        "2014), which raised test BLEU from 2.45 to 4.64. BLEU by source "
        "length: 1-5 → 4.34, 6-10 → 3.72, 11-15 → 3.32, 16+ → 4.13. Even at "
        "its best this vanilla RNN produces fluent but largely unfaithful "
        "output — the constraint, not the implementation, is the limit."
    )
    model, q2_src_vocab, q2_tgt_vocab, q2_cfg = load_q2_model_and_vocabs()
    if model is None:
        st.warning("runs/q2/main/best.pt (or its vocab/result files) not found")
    else:
        sentence = st.text_input("English sentence", key="q2_input")
        if st.button("Translate", key="q2_translate_btn") and sentence.strip():
            try:
                normalized = normalize_english(sentence)
                tokens = tokenize(normalized)
                ids = q2_src_vocab.encode(tokens)
                # Must mirror training: the stored config says whether the
                # encoder was trained on reversed source sequences.
                if bool(q2_cfg.get("reverse_source", False)):
                    ids = ids[::-1]
                enc_in = torch.tensor([ids], dtype=torch.long)
                enc_len = torch.tensor([len(ids)], dtype=torch.long)
                with torch.no_grad():
                    pred_ids = model.greedy_decode(enc_in, enc_len, max_len=50)
                out_ids = []
                for tok_id in pred_ids[0].tolist():
                    if tok_id in (EOS, PAD):
                        break
                    out_ids.append(tok_id)
                urdu_text = " ".join(q2_tgt_vocab.decode(out_ids, strip_specials=False))
                st.success(f"Urdu translation: **{urdu_text}**")
            except Exception as e:
                st.error(f"Translation failed: {e}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

st.title("GenAI A1 — Coursework Viewer")

tab1, tab2 = st.tabs(["Q1: Chest X-ray CNN", "Q2: English→Urdu NMT"])
with tab1:
    render_q1()
with tab2:
    render_q2()

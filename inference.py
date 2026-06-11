"""
inference.py – Inference wrapper for the Streamlit ECG interface
================================================================
Wraps your existing part1–part7 code so app.py stays clean.

Supports:
    - Loading trained checkpoints (CNN or CNN-BiLSTM)
    - Preprocessing raw signals from CSV or WFDB
    - Running forward pass → probabilities
    - Grad-CAM on the last CNN stage
    - Attention weights from BiLSTM model
"""

import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ── re-use your existing modules ─────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from part1_load_preprocess import preprocess_signal, SIGNAL_LENGTH, NUM_LEADS
from part2_label_mapping   import SUPERCLASSES
from part5_baseline_cnn    import build_baseline_cnn
from part6_cnn_bilstm      import build_cnn_bilstm
from part7_train           import load_best_model

# ── constants ─────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("./checkpoints")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEAD_NAMES     = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
CLASS_COLORS   = {
    "NORM": "#4C9BE8", "MI": "#E8694C",
    "STTC": "#6DBE6D", "CD": "#B36DBE", "HYP": "#E8C24C",
}
THRESHOLD = 0.5   # default decision threshold


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached by Streamlit via @st.cache_resource)
# ══════════════════════════════════════════════════════════════════
def load_model(model_name: str):
    """
    Load a trained model from checkpoints/.
    Call this once at startup — Streamlit caches it.
    """
    model = load_best_model(model_name, DEVICE)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════
# SIGNAL LOADING
# ══════════════════════════════════════════════════════════════════
def load_from_csv(file_obj) -> np.ndarray:
    """
    Load ECG from an uploaded CSV file.
    Expected: 12 columns (leads) × up to 5000 rows (time steps).
    Returns: preprocessed np.ndarray shape (5000, 12) float32
    """
    df = pd.read_csv(file_obj, header=None)
    if df.shape[1] != 12:
        raise ValueError(
            f"CSV must have exactly 12 columns (one per lead). "
            f"Got {df.shape[1]} columns."
        )
    signal = df.values.astype(np.float32)

    # Pad or trim to SIGNAL_LENGTH
    if signal.shape[0] < SIGNAL_LENGTH:
        pad = np.zeros((SIGNAL_LENGTH - signal.shape[0], NUM_LEADS), dtype=np.float32)
        signal = np.vstack([signal, pad])
    else:
        signal = signal[:SIGNAL_LENGTH, :]

    return preprocess_signal(signal)


def load_from_wfdb(hea_bytes: bytes, dat_bytes: bytes) -> np.ndarray:
    """
    Load ECG from WFDB .hea + .dat bytes uploaded by the user.
    Writes temp files, reads with wfdb, returns preprocessed signal.
    """
    import wfdb, tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "record.hea").write_bytes(hea_bytes)
        (tmp / "record.dat").write_bytes(dat_bytes)
        record = wfdb.rdrecord(str(tmp / "record"))
        signal = record.p_signal.astype(np.float32)
        if signal.shape[0] < SIGNAL_LENGTH:
            pad = np.zeros((SIGNAL_LENGTH - signal.shape[0], NUM_LEADS), dtype=np.float32)
            signal = np.vstack([signal, pad])
        else:
            signal = signal[:SIGNAL_LENGTH, :]
        return preprocess_signal(signal)
    finally:
        shutil.rmtree(tmp)


# ══════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(model, signal: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """
    Run forward pass and return predictions.

    Args:
        model   : loaded PyTorch model
        signal  : (5000, 12) preprocessed signal
        threshold: decision threshold for binary prediction

    Returns dict with:
        probs      : (5,) float — sigmoid probabilities
        preds      : (5,) int  — binary predictions
        logits     : (5,) float
        tensor     : (1, 12, 5000) torch tensor (for Grad-CAM)
    """
    # (5000, 12) → (1, 12, 5000)
    x = torch.from_numpy(signal.T.copy()).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs  = torch.sigmoid(logits).cpu().squeeze().numpy()
    preds  = (probs >= threshold).astype(int)

    return {
        "probs" : probs,
        "preds" : preds,
        "logits": logits.cpu().squeeze().numpy(),
        "tensor": x,
    }


# ══════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════
def compute_gradcam(model, tensor: torch.Tensor, class_idx: int,
                    model_name: str) -> np.ndarray:

    # cuDNN RNN backward requires train() mode — switch temporarily
    was_training = model.training
    model.train()  # ← KEY FIX: switch to train mode for backward pass

    # Disable BatchNorm/Dropout stochasticity while in train mode
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.Dropout, torch.nn.Dropout2d)):
            module.eval()

    tensor = tensor.clone().requires_grad_(True)

    try:
        with torch.enable_grad():
            if model_name == "baseline_cnn":
                x = model.stem(tensor)
                x = model.stage1(x)
                x = model.stage2(x)
                x = model.stage3(x)
                feat = model.stage4(x)
            else:
                feat = model.get_cnn_features(tensor)

            feat.retain_grad()

            if model_name == "baseline_cnn":
                logits = model.head(feat)
            else:
                x = feat.permute(0, 2, 1)
                x, _ = model.bilstm(x)
                x = model.var_dropout(x)
                x = model.attention(x)
                x_avg = x.mean(dim=1)
                x_max = x.max(dim=1).values
                logits = model.head(torch.cat([x_avg, x_max], dim=1))

            model.zero_grad()
            logits[0, class_idx].backward()

    finally:
        # Always restore original mode
        if not was_training:
            model.eval()

    grads   = feat.grad
    weights = grads.mean(dim=-1, keepdim=True)
    cam     = (weights * feat).sum(dim=1)
    cam     = F.relu(cam).squeeze().detach().cpu().numpy()

    cam_up = np.interp(
        np.linspace(0, len(cam) - 1, SIGNAL_LENGTH),
        np.arange(len(cam)),
        cam,
    )
    if cam_up.max() > 0:
        cam_up = cam_up / cam_up.max()

    return cam_up

# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════
def plot_ecg(signal: np.ndarray, cam: np.ndarray = None,
             fs: int = 500, title: str = "12-Lead ECG") -> plt.Figure:
    """
    Plot all 12 ECG leads. If cam is provided, overlay as a heatmap
    background tinting each region by saliency.

    Args:
        signal : (5000, 12) — time × leads
        cam    : (5000,) optional Grad-CAM weights
        fs     : sampling rate (for x-axis in seconds)
    """
    t     = np.arange(SIGNAL_LENGTH) / fs
    ncols = 2
    nrows = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10),
                             sharex=True, facecolor="#0e1117")
    fig.suptitle(title, fontsize=14, color="white", fontweight="bold")

    for i, ax in enumerate(axes.flatten()):
        ax.set_facecolor("#0e1117")
        lead_sig = signal[:, i]
        ax.plot(t, lead_sig, color="#00c8ff", linewidth=0.7, alpha=0.9)

        if cam is not None:
            # Colour background by saliency
            rgba = cm.hot(cam)
            for j in range(0, SIGNAL_LENGTH - 1, 10):
                ax.axvspan(t[j], t[min(j + 10, SIGNAL_LENGTH - 1)],
                           color=rgba[j][:3], alpha=float(cam[j]) * 0.5,
                           linewidth=0)

        ax.set_ylabel(LEAD_NAMES[i], color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.grid(True, color="#222", linewidth=0.4)

    axes[-1, 0].set_xlabel("Time (s)", color="white", fontsize=9)
    axes[-1, 1].set_xlabel("Time (s)", color="white", fontsize=9)
    fig.tight_layout()
    return fig


def plot_probabilities(probs: np.ndarray, preds: np.ndarray,
                       threshold: float = THRESHOLD) -> plt.Figure:
    """
    Horizontal bar chart of per-class probabilities.
    Bars are coloured by class; threshold line shown.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    colors = [CLASS_COLORS[sc] for sc in SUPERCLASSES]
    y_pos  = np.arange(len(SUPERCLASSES))

    bars = ax.barh(y_pos, probs, color=colors, alpha=0.85, height=0.55)

    # Threshold line
    ax.axvline(threshold, color="white", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Threshold ({threshold:.2f})")

    # Labels on bars
    for i, (bar, prob, pred) in enumerate(zip(bars, probs, preds)):
        label = f"{prob:.2f}  {'✓' if pred else ''}"
        ax.text(min(prob + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left",
                color="white", fontsize=10, fontweight="bold" if pred else "normal")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(SUPERCLASSES, color="white", fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Probability", color="white")
    ax.set_title("Predicted probabilities per class", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(fontsize=8, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#333")
    fig.tight_layout()
    return fig


def plot_attention(attn_weights: np.ndarray) -> plt.Figure:
    """
    Plot temporal attention weight matrix as a heatmap.
    attn_weights: (T, T) numpy array
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    im = ax.imshow(attn_weights, aspect="auto",
                   cmap="magma", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046,
                 label="Attention weight").ax.yaxis.label.set_color("white")
    ax.set_title("Temporal self-attention map (BiLSTM)", color="white", fontsize=11)
    ax.set_xlabel("Key timestep", color="white")
    ax.set_ylabel("Query timestep", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# REPORT EXPORT
# ══════════════════════════════════════════════════════════════════
def build_report(probs: np.ndarray, preds: np.ndarray,
                 model_name: str, threshold: float) -> dict:
    """Build a JSON-serialisable report dict for download."""
    return {
        "model"    : model_name,
        "threshold": round(float(threshold), 3),
        "classes"  : {
            sc: {
                "probability": round(float(probs[i]), 4),
                "prediction" : int(preds[i]),
            }
            for i, sc in enumerate(SUPERCLASSES)
        },
        "positive_classes": [
            sc for i, sc in enumerate(SUPERCLASSES) if preds[i]
        ],
    }

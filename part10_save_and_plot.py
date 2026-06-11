"""
Part 10 – Model Saving and Performance Plots
=============================================
Depends on: part1–part9

Provides:
    1. Robust model export utilities (checkpoint, ONNX, TorchScript)
    2. Training curve plots (loss, F1, LR schedule)
    3. Model comparison dashboard (baseline CNN vs CNN+BiLSTM)
    4. Per-class F1 progression across epochs
    5. Final summary report card (publication-ready figure)
    6. Full project file inventory printout

Run standalone (after training both models):
    python part10_save_and_plot.py --models baseline_cnn cnn_bilstm

Run for a single model:
    python part10_save_and_plot.py --models cnn_bilstm
"""

import argparse
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch
import torch.nn as nn

from part2_label_mapping import SUPERCLASSES
from part5_baseline_cnn  import build_baseline_cnn
from part6_cnn_bilstm    import build_cnn_bilstm
from part7_train         import load_best_model, CHECKPOINT_DIR, LOG_DIR
from part8_evaluate      import evaluate, CLASS_COLORS

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
PLOTS_DIR  = Path("./final_plots")
EXPORT_DIR = Path("./model_exports")
REPORT_DIR = Path("./reports")

NUM_CLASSES = len(SUPERCLASSES)


# ══════════════════════════════════════════════
# SECTION 1 — MODEL EXPORT UTILITIES
# ══════════════════════════════════════════════

def export_checkpoint_metadata(model_name: str) -> Dict:
    """
    Load the best checkpoint and extract + save its metadata as JSON.
    Useful for reproducibility — records exact epoch, val F1, config used.

    Returns:
        metadata dict
    """
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    ckpt      = torch.load(ckpt_path, map_location="cpu")

    metadata = {
        "model_name"   : model_name,
        "saved_epoch"  : int(ckpt["epoch"]),
        "val_macro_f1" : float(ckpt["val_f1"]),
        "val_class_f1s": {
            sc: float(f1)
            for sc, f1 in zip(SUPERCLASSES, ckpt.get("val_class_f1s", [0]*5))
        },
        "config"       : ckpt.get("config", {}),
        "exported_at"  : datetime.now().isoformat(),
    }

    out = EXPORT_DIR / f"{model_name}_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[EXPORT] Metadata → {out}")
    return metadata


def export_torchscript(model_name: str, device: torch.device) -> Path:
    """
    Export model to TorchScript (.pt) via tracing.
    TorchScript models run without the Python source code —
    suitable for C++ inference or serving.

    Returns:
        path to exported .pt file
    """
    model   = load_best_model(model_name, device)
    model.eval()

    # Trace with a dummy input
    dummy   = torch.randn(1, 12, 5000, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    out = EXPORT_DIR / f"{model_name}_torchscript.pt"
    traced.save(str(out))
    print(f"[EXPORT] TorchScript → {out}")
    return out


def export_onnx(model_name: str, device: torch.device) -> Path:
    """
    Export model to ONNX format.
    ONNX enables inference with ONNX Runtime, TensorRT, or CoreML.

    Returns:
        path to exported .onnx file
    """
    model = load_best_model(model_name, device)
    model.eval()

    dummy  = torch.randn(1, 12, 5000, device=device)
    out    = EXPORT_DIR / f"{model_name}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        export_params    = True,
        opset_version    = 14,
        input_names      = ["ecg_signal"],
        output_names     = ["class_logits"],
        dynamic_axes     = {
            "ecg_signal"  : {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
        do_constant_folding = True,
    )
    print(f"[EXPORT] ONNX       → {out}")
    return out


def copy_best_checkpoint(model_name: str) -> Path:
    """Copy best checkpoint to the export directory for archiving."""
    src = CHECKPOINT_DIR / f"{model_name}_best.pt"
    dst = EXPORT_DIR / f"{model_name}_best.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[EXPORT] Checkpoint → {dst}")
    return dst


# ══════════════════════════════════════════════
# SECTION 2 — TRAINING CURVE PLOTS
# ══════════════════════════════════════════════

def load_training_log(model_name: str) -> pd.DataFrame:
    """Load the CSV training log produced by Part 7."""
    path = LOG_DIR / f"{model_name}_training_log.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")
    return pd.read_csv(path)


def plot_training_curves(log: pd.DataFrame, model_name: str, save_path: Path) -> None:
    """
    4-panel training dashboard:
        1. Train + Val Loss
        2. Train + Val Macro F1
        3. Val F1 per superclass
        4. Learning rate schedule
    """
    epochs = log["epoch"].values
    fig    = plt.figure(figsize=(16, 11))
    fig.suptitle(f"Training Dashboard — {model_name.replace('_', ' ').title()}",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    # ── Panel 1: Loss ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, log["train_loss"], label="Train", color="#2196F3", lw=2)
    ax1.plot(epochs, log["val_loss"],   label="Val",   color="#F44336", lw=2,
             linestyle="--")
    best_ep = log.loc[log["val_loss"].idxmin(), "epoch"]
    ax1.axvline(best_ep, color="gray", ls=":", lw=1.2, label=f"Best epoch={best_ep}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCEWithLogits Loss")
    ax1.set_title("Loss Curves", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ── Panel 2: Macro F1 ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, log["train_f1"], label="Train F1", color="#4CAF50", lw=2)
    ax2.plot(epochs, log["val_f1"],   label="Val F1",   color="#FF9800", lw=2,
             linestyle="--")
    best_f1_ep = log.loc[log["val_f1"].idxmax(), "epoch"]
    best_f1    = log["val_f1"].max()
    ax2.axvline(best_f1_ep, color="gray", ls=":", lw=1.2,
                label=f"Best F1={best_f1:.4f} @ ep{best_f1_ep}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Macro F1 Score", fontweight="bold")
    ax2.set_ylim(0, 1); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # ── Panel 3: Per-class Val F1 ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for sc in SUPERCLASSES:
        col = f"val_f1_{sc}"
        if col in log.columns:
            ax3.plot(epochs, log[col], label=sc,
                     color=CLASS_COLORS[sc], lw=1.8)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("F1 Score")
    ax3.set_title("Per-Class Validation F1", fontweight="bold")
    ax3.set_ylim(0, 1); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # ── Panel 4: Learning Rate ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(epochs, log["lr"], color="#9C27B0", lw=2)
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Learning Rate (log scale)")
    ax4.set_title("OneCycleLR Schedule", fontweight="bold")
    ax4.grid(True, alpha=0.3, which="both")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Training curves → {save_path.name}")


# ══════════════════════════════════════════════
# SECTION 3 — MODEL COMPARISON DASHBOARD
# ══════════════════════════════════════════════

def plot_model_comparison(
    metrics_dict: Dict[str, Dict],
    save_path   : Path,
) -> None:
    """
    Side-by-side comparison of all available models across F1, AUROC, AP.
    Works with 1 or 2 models.

    Args:
        metrics_dict : {model_name: metrics_from_part8}
    """
    model_names = list(metrics_dict.keys())
    n_models    = len(model_names)
    x           = np.arange(NUM_CLASSES)
    bar_w       = 0.35 / n_models

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Comparison — Test Set", fontsize=14, fontweight="bold")

    metric_keys   = ["class_f1s",   "class_aurocs", "class_aps"]
    metric_labels = ["F1 Score",    "AUROC",        "Average Precision"]
    macro_keys    = ["macro_f1",    "macro_auroc",  "macro_ap"]

    palettes = {
        "baseline_cnn": "#5C85D6",
        "cnn_bilstm"  : "#E85C5C",
    }
    default_colors = ["#5C85D6", "#E85C5C", "#5CD6A0"]

    for col, (ax, mkey, mlabel, mcrkey) in enumerate(
        zip(axes, metric_keys, metric_labels, macro_keys)
    ):
        for mi, (mname, metrics) in enumerate(metrics_dict.items()):
            offset = (mi - (n_models - 1) / 2) * bar_w
            color  = palettes.get(mname, default_colors[mi % len(default_colors)])
            vals   = metrics[mkey]

            bars = ax.bar(x + offset, vals, bar_w,
                          label=f"{mname.replace('_',' ').title()} "
                                f"(macro={metrics[mcrkey]:.3f})",
                          color=color, alpha=0.85)

            # Value labels on bars
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(SUPERCLASSES, fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(mlabel, fontsize=11)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Comparison   → {save_path.name}")


# ══════════════════════════════════════════════
# SECTION 4 — REPORT CARD
# ══════════════════════════════════════════════

def plot_report_card(
    metrics_dict: Dict[str, Dict],
    save_path   : Path,
) -> None:
    """
    Publication-ready summary figure: radar chart + metric table.
    One radar overlay per model.
    """
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe

    n_metrics = NUM_CLASSES
    angles    = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles   += angles[:1]   # close the polygon

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("ECG Superclass Classification — Performance Report Card",
                 fontsize=14, fontweight="bold", y=0.99)

    # ── Left: Radar chart ────────────────────────────────────────────────
    ax_radar = fig.add_subplot(121, polar=True)
    ax_radar.set_facecolor("#F0F0F8")

    palettes = {"baseline_cnn": "#4472C4", "cnn_bilstm": "#C44444"}
    default_colors = ["#4472C4", "#C44444"]

    for mi, (mname, metrics) in enumerate(metrics_dict.items()):
        vals   = list(metrics["class_f1s"]) + [metrics["class_f1s"][0]]
        color  = palettes.get(mname, default_colors[mi % 2])
        label  = f"{mname.replace('_',' ').title()} F1 (macro={metrics['macro_f1']:.3f})"
        ax_radar.plot(angles, vals, "o-", lw=2, color=color, label=label, markersize=5)
        ax_radar.fill(angles, vals, alpha=0.12, color=color)

    ax_radar.set_thetagrids(np.degrees(angles[:-1]), SUPERCLASSES, fontsize=11)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax_radar.set_title("F1 Score Radar", fontsize=12, fontweight="bold", pad=18)
    ax_radar.grid(True, alpha=0.4)

    # ── Right: Summary table ──────────────────────────────────────────────
    ax_table = fig.add_subplot(122)
    ax_table.axis("off")

    # Build table data
    col_labels = ["Metric"] + [m.replace("_", "\n") for m in metrics_dict.keys()]
    rows       = []

    for sc in SUPERCLASSES:
        i = SUPERCLASSES.index(sc)
        row = [f"F1  {sc}"]
        for metrics in metrics_dict.values():
            row.append(f"{metrics['class_f1s'][i]:.4f}")
        rows.append(row)

    for sc in SUPERCLASSES:
        i = SUPERCLASSES.index(sc)
        row = [f"AUC {sc}"]
        for metrics in metrics_dict.values():
            row.append(f"{metrics['class_aurocs'][i]:.4f}")
        rows.append(row)

    # Macro rows
    for label, key in [("Macro F1", "macro_f1"), ("Macro AUROC", "macro_auroc"),
                        ("Macro AP", "macro_ap")]:
        row = [label]
        for metrics in metrics_dict.values():
            row.append(f"{metrics[key]:.4f}")
        rows.append(row)

    table = ax_table.table(
        cellText    = rows,
        colLabels   = col_labels,
        loc         = "center",
        cellLoc     = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.1, 1.65)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight macro rows (last 3)
    for i in range(len(rows) - 3, len(rows)):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor("#EBF5FB")
            table[i + 1, j].set_text_props(fontweight="bold")

    # Alternating row colours
    for i in range(len(rows) - 3):
        bg = "#FFFFFF" if i % 2 == 0 else "#F4F6F7"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)

    ax_table.set_title("Detailed Metric Table", fontsize=12,
                        fontweight="bold", pad=10)

    # Timestamp watermark
    fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             ha="right", va="bottom", fontsize=7, color="gray", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PLOT] Report card  → {save_path.name}")


# ══════════════════════════════════════════════
# SECTION 5 — FILE INVENTORY
# ══════════════════════════════════════════════

def print_project_inventory() -> None:
    """Print all generated files across the project."""
    dirs = {
        "Data Cache"      : Path("./data_cache"),
        "Checkpoints"     : CHECKPOINT_DIR,
        "Logs"            : LOG_DIR,
        "Eval Plots"      : Path("./evaluation_plots"),
        "XAI Plots"       : Path("./xai_plots"),
        "Final Plots"     : PLOTS_DIR,
        "Model Exports"   : EXPORT_DIR,
        "Reports"         : REPORT_DIR,
    }

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                  PROJECT FILE INVENTORY                 ║")
    print("╠══════════════════════════════════════════════════════════╣")

    total_size = 0
    for label, d in dirs.items():
        if not d.exists():
            print(f"║  {label:<18} — (not yet created)               ║")
            continue
        files = sorted(d.glob("**/*"))
        files = [f for f in files if f.is_file()]
        size  = sum(f.stat().st_size for f in files)
        total_size += size
        size_mb = size / (1024 ** 2)
        print(f"╠══════════════════════════════════════════════════════════╣")
        print(f"║  📁 {label:<18} ({len(files)} files, {size_mb:.1f} MB)          ║")
        for f in files[:8]:     # show max 8 per dir
            fsize = f.stat().st_size / 1024
            print(f"║    ├─ {f.name:<44} {fsize:>6.1f} KB ║")
        if len(files) > 8:
            print(f"║    └─ … and {len(files)-8} more files                         ║")

    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Total project output size: {total_size/(1024**2):.1f} MB                    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main(model_names: List[str]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_dict = {}

    for model_name in model_names:
        print(f"\n{'═'*60}")
        print(f"  Processing: {model_name}")
        print(f"{'═'*60}")

        # ── 1. Export checkpoint metadata ────────────────────────────────
        try:
            meta = export_checkpoint_metadata(model_name)
            print(f"[INFO] Best epoch={meta['saved_epoch']}  "
                  f"val F1={meta['val_macro_f1']:.4f}")
        except FileNotFoundError:
            print(f"[WARN] No checkpoint found for {model_name} — skipping export.")
            continue

        # ── 2. Export TorchScript ────────────────────────────────────────
        try:
            export_torchscript(model_name, device)
        except Exception as e:
            print(f"[WARN] TorchScript export failed: {e}")

        # ── 3. Export ONNX ───────────────────────────────────────────────
        try:
            export_onnx(model_name, device)
        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}")

        # ── 4. Copy best checkpoint ───────────────────────────────────────
        copy_best_checkpoint(model_name)

        # ── 5. Training curves ────────────────────────────────────────────
        try:
            log = load_training_log(model_name)
            plot_training_curves(log, model_name,
                                 PLOTS_DIR / f"{model_name}_training_curves.png")
        except FileNotFoundError as e:
            print(f"[WARN] Training log not found: {e}")

        # ── 6. Run evaluation + collect metrics ───────────────────────────
        print(f"\n[INFO] Running test set evaluation for {model_name}…")
        try:
            metrics = evaluate(model_name)
            metrics_dict[model_name] = metrics
        except Exception as e:
            print(f"[WARN] Evaluation failed: {e}")

    # ── 7. Multi-model comparison ─────────────────────────────────────────
    if len(metrics_dict) >= 1:
        plot_model_comparison(
            metrics_dict,
            PLOTS_DIR / "model_comparison.png",
        )

    # ── 8. Report card ────────────────────────────────────────────────────
    if len(metrics_dict) >= 1:
        plot_report_card(
            metrics_dict,
            REPORT_DIR / "performance_report_card.png",
        )

    # ── 9. Save combined metrics CSV ──────────────────────────────────────
    all_rows = []
    for mname, metrics in metrics_dict.items():
        for i, sc in enumerate(SUPERCLASSES):
            all_rows.append({
                "model" : mname,
                "class" : sc,
                "f1"    : round(float(metrics["class_f1s"][i]),    4),
                "auroc" : round(float(metrics["class_aurocs"][i]),  4),
                "ap"    : round(float(metrics["class_aps"][i]),     4),
            })
        all_rows.append({
            "model": mname, "class": "MACRO",
            "f1"   : round(float(metrics["macro_f1"]),    4),
            "auroc": round(float(metrics["macro_auroc"]), 4),
            "ap"   : round(float(metrics["macro_ap"]),    4),
        })

    if all_rows:
        csv_out = REPORT_DIR / "all_models_metrics.csv"
        pd.DataFrame(all_rows).to_csv(csv_out, index=False)
        print(f"\n[INFO] Combined metrics CSV → {csv_out}")

    # ── 10. File inventory ────────────────────────────────────────────────
    print_project_inventory()
    print("[INFO] Part 10 complete. Project is ready for publication.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["cnn_bilstm"],
        choices=["baseline_cnn", "cnn_bilstm"],
        help="One or both model names to process",
    )
    args = parser.parse_args()
    main(args.models)

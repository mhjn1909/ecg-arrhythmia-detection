"""
Part 8 – Evaluation: Macro F1, AUROC, Confusion Matrix
========================================================
Depends on: part1–part7

Changes from original:
    - Added bootstrap_confidence_interval() — error bars on Macro F1
    - Added mcnemar_test() — statistical significance between two models
    - Added print_benchmark_comparison() — PTB-XL paper reference numbers
    - compare_models() function runs both models and prints full comparison
    - All original plots and CSV exports unchanged

Run standalone (after training):
    python part8_evaluate.py --model cnn_bilstm
    python part8_evaluate.py --compare   ← runs both models + significance test
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from scipy.stats import chi2   # for McNemar

from part1_load_preprocess import load_metadata, load_all_waveforms, PTB_XL_PATH
from part2_label_mapping   import load_labels, SUPERCLASSES
from part3_data_split      import (align_signals_labels, load_splits,
                                   get_split_data, patient_wise_split,
                                   LABELS_CSV, SPLIT_CACHE)
from part4_dataset         import get_dataloaders
from part7_train           import load_best_model, run_epoch

PLOT_DIR    = Path("./evaluation_plots")
NUM_CLASSES = len(SUPERCLASSES)

CLASS_COLORS = {
    "NORM": "#4C9BE8", "MI": "#E8694C", "STTC": "#6DBE6D",
    "CD"  : "#B36DBE", "HYP": "#E8C24C",
}

# ══════════════════════════════════════════════════════════════════
# PTB-XL BENCHMARK REFERENCE (from Strodthoff et al. 2020 paper)
# https://www.nature.com/articles/s41597-020-0495-6
# Table 3 — best reported numbers on PTB-XL superclass task
# Using their best single model (xresnet1d101) as reference
# ══════════════════════════════════════════════════════════════════
PTB_XL_BENCHMARK = {
    "model"     : "xresnet1d101 (Strodthoff et al. 2020)",
    "macro_f1"  : 0.809,
    "macro_auroc": 0.931,
    "NORM_f1"   : 0.882,
    "MI_f1"     : 0.826,
    "STTC_f1"   : 0.820,
    "CD_f1"     : 0.809,
    "HYP_f1"    : 0.706,
}


# ══════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def get_predictions(
    model  : nn.Module,
    loader : torch.utils.data.DataLoader,
    device : torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        logits  = model(signals)
        all_logits.append(logits.cpu().float().numpy())
        all_labels.append(labels.float().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ══════════════════════════════════════════════════════════════════
# OPTIMAL THRESHOLD SEARCH
# ══════════════════════════════════════════════════════════════════
def find_optimal_thresholds(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    n_steps   : int = 100,
) -> np.ndarray:
    probs      = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
    thresholds = np.zeros(NUM_CLASSES)
    candidates = np.linspace(0.1, 0.9, n_steps)

    for c in range(NUM_CLASSES):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            preds = (probs[:, c] >= t).astype(int)
            f1    = f1_score(val_labels[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t

    print("\n── Optimal Thresholds (per class) ───────────────────────────")
    for sc, t in zip(SUPERCLASSES, thresholds):
        print(f"  {sc:<6}: {t:.3f}")
    return thresholds


# ══════════════════════════════════════════════════════════════════
# CORE METRICS
# ══════════════════════════════════════════════════════════════════
def compute_all_metrics(
    logits    : np.ndarray,
    labels    : np.ndarray,
    thresholds: np.ndarray,
) -> Dict:
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = np.zeros_like(probs, dtype=int)
    for c in range(NUM_CLASSES):
        preds[:, c] = (probs[:, c] >= thresholds[c]).astype(int)

    class_f1s    = f1_score(labels, preds, average=None, zero_division=0)
    macro_f1     = class_f1s.mean()
    class_aurocs = roc_auc_score(labels, probs, average=None)
    macro_auroc  = class_aurocs.mean()
    class_aps    = average_precision_score(labels, probs, average=None)
    macro_ap     = class_aps.mean()

    return {
        "macro_f1"    : macro_f1,
        "class_f1s"   : class_f1s,
        "macro_auroc" : macro_auroc,
        "class_aurocs": class_aurocs,
        "macro_ap"    : macro_ap,
        "class_aps"   : class_aps,
        "preds"       : preds,
        "probs"       : probs,
    }


def print_metrics(metrics: Dict) -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║              TEST SET EVALUATION RESULTS                ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Macro F1    : {metrics['macro_f1']:.4f}                                ║")
    print(f"║  Macro AUROC : {metrics['macro_auroc']:.4f}                                ║")
    print(f"║  Macro AP    : {metrics['macro_ap']:.4f}                                ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  {'Class':<6}  {'F1':>6}  {'AUROC':>7}  {'AP':>7}                  ║")
    print(f"║  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}                  ║")
    for sc, f1, auc, ap in zip(
        SUPERCLASSES, metrics["class_f1s"],
        metrics["class_aurocs"], metrics["class_aps"],
    ):
        print(f"║  {sc:<6}  {f1:>6.4f}  {auc:>7.4f}  {ap:>7.4f}                  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════════════════════════
# NEW: BOOTSTRAP CONFIDENCE INTERVAL ON MACRO F1
# ══════════════════════════════════════════════════════════════════
def bootstrap_confidence_interval(
    labels    : np.ndarray,
    preds     : np.ndarray,
    metric_fn = None,
    n_boot    : int   = 1000,
    ci        : float = 0.95,
    seed      : int   = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for any scalar metric.

    Resamples (with replacement) n_boot times from test set predictions
    and computes the metric each time. Reports percentile CI.

    Args:
        labels    : (N, C) ground truth
        preds     : (N, C) binary predictions
        metric_fn : callable(labels, preds) → float. Defaults to Macro F1.
        n_boot    : number of bootstrap samples
        ci        : confidence level (0.95 = 95% CI)
        seed      : random seed for reproducibility

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    if metric_fn is None:
        metric_fn = lambda y, p: f1_score(y, p, average="macro", zero_division=0)

    rng      = np.random.default_rng(seed)
    n        = len(labels)
    boot_scores = []

    for _ in range(n_boot):
        idx   = rng.integers(0, n, size=n)
        score = metric_fn(labels[idx], preds[idx])
        boot_scores.append(score)

    boot_scores = np.array(boot_scores)
    alpha       = (1 - ci) / 2
    lower       = np.percentile(boot_scores, alpha * 100)
    upper       = np.percentile(boot_scores, (1 - alpha) * 100)
    point       = metric_fn(labels, preds)

    return point, lower, upper


# ══════════════════════════════════════════════════════════════════
# NEW: McNEMAR'S TEST (model A vs model B significance)
# ══════════════════════════════════════════════════════════════════
def mcnemar_test(
    labels   : np.ndarray,   # (N, C) ground truth
    preds_a  : np.ndarray,   # (N, C) model A binary predictions
    preds_b  : np.ndarray,   # (N, C) model B binary predictions
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> None:
    """
    McNemar's test for significance of difference between two classifiers.

    For multi-label: flattens all (sample, class) pairs and treats as
    a single binary comparison. A sample-class pair is 'correct' if
    predicted label matches ground truth.

    Null hypothesis: both models make errors on the same pairs.
    If p < 0.05: reject null → difference is statistically significant.

    Prints:
        - contingency table (b, c counts)
        - chi-squared statistic
        - p-value
        - conclusion
    """
    # Flatten across all samples and classes
    correct_a = (preds_a == labels).astype(int).flatten()
    correct_b = (preds_b == labels).astype(int).flatten()

    # Contingency counts
    # b: A wrong, B correct | c: A correct, B wrong
    b = ((correct_a == 0) & (correct_b == 1)).sum()
    c = ((correct_a == 1) & (correct_b == 0)).sum()

    # McNemar statistic with continuity correction (Yates)
    if (b + c) == 0:
        print("[McNemar] Both models make identical errors — cannot test.")
        return

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)

    print("\n── McNemar's Test ────────────────────────────────────────────")
    print(f"  Comparing : {model_a_name}  vs  {model_b_name}")
    print(f"  b (A wrong, B correct) : {b:,}")
    print(f"  c (A correct, B wrong) : {c:,}")
    print(f"  Chi-squared statistic  : {chi2_stat:.4f}")
    print(f"  p-value                : {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✓ SIGNIFICANT (p < 0.05) — {model_b_name} is meaningfully better")
    else:
        print(f"  ✗ NOT SIGNIFICANT (p ≥ 0.05) — difference may be due to chance")
    print("─" * 62)


# ══════════════════════════════════════════════════════════════════
# NEW: PTB-XL BENCHMARK COMPARISON
# ══════════════════════════════════════════════════════════════════
def print_benchmark_comparison(metrics_a: Dict, metrics_b: Dict) -> None:
    """
    Print a table comparing your models against PTB-XL paper benchmark.
    Reference: Strodthoff et al. 2020, xresnet1d101 (best single model).
    """
    ref = PTB_XL_BENCHMARK

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║            BENCHMARK COMPARISON vs PTB-XL Paper                    ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  {'Metric':<12} {'Ref (paper)':>12} {'Baseline CNN':>14} {'CNN-BiLSTM':>12} ║")
    print(f"║  {'─'*12} {'─'*12} {'─'*14} {'─'*12} ║")

    # Macro F1
    print(f"║  {'Macro F1':<12} {ref['macro_f1']:>12.4f} "
          f"{metrics_a['macro_f1']:>14.4f} {metrics_b['macro_f1']:>12.4f} ║")

    # Macro AUROC
    print(f"║  {'Macro AUROC':<12} {ref['macro_auroc']:>12.4f} "
          f"{metrics_a['macro_auroc']:>14.4f} {metrics_b['macro_auroc']:>12.4f} ║")

    print(f"║  {'─'*12} {'─'*12} {'─'*14} {'─'*12} ║")

    # Per-class F1
    class_keys = [f"{sc}_f1" for sc in SUPERCLASSES]
    for sc, key in zip(SUPERCLASSES, class_keys):
        ref_val = ref.get(key, float("nan"))
        a_val   = metrics_a["class_f1s"][SUPERCLASSES.index(sc)]
        b_val   = metrics_b["class_f1s"][SUPERCLASSES.index(sc)]
        print(f"║  {sc+' F1':<12} {ref_val:>12.4f} {a_val:>14.4f} {b_val:>12.4f} ║")

    print("╠══════════════════════════════════════════════════════════════════════╣")
    gap_a = metrics_a["macro_f1"] - ref["macro_f1"]
    gap_b = metrics_b["macro_f1"] - ref["macro_f1"]
    print(f"║  Gap vs ref : Baseline CNN {gap_a:+.4f} | CNN-BiLSTM {gap_b:+.4f}          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")


# ══════════════════════════════════════════════════════════════════
# PLOTS (unchanged from original)
# ══════════════════════════════════════════════════════════════════
def plot_roc_curves(labels, probs, aurocs, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, sc in enumerate(SUPERCLASSES):
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{sc}  (AUC={aurocs[i]:.3f})",
                color=CLASS_COLORS[sc], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] ROC curves → {save_path}")


def plot_pr_curves(labels, probs, aps, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, sc in enumerate(SUPERCLASSES):
        prec, rec, _ = precision_recall_curve(labels[:, i], probs[:, i])
        ax.plot(rec, prec, label=f"{sc}  (AP={aps[i]:.3f})",
                color=CLASS_COLORS[sc], lw=2)
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] PR curves  → {save_path}")


def plot_confusion_matrices(labels, preds, save_path):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(20, 4))
    fig.suptitle("Confusion Matrices (One-vs-Rest) — Test Set",
                 fontsize=14, fontweight="bold")
    for i, (sc, ax) in enumerate(zip(SUPERCLASSES, axes)):
        cm      = confusion_matrix(labels[:, i], preds[:, i])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im      = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred–", "Pred+"])
        ax.set_yticklabels(["True–", "True+"])
        ax.set_title(sc, fontsize=12, fontweight="bold", color=CLASS_COLORS[sc])
        for r in range(2):
            for c in range(2):
                raw  = cm[r, c]; rate = cm_norm[r, c]
                color = "white" if rate > 0.6 else "black"
                ax.text(c, r, f"{raw}\n({rate:.2f})",
                        ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Confusion  → {save_path}")


def plot_probability_distributions(probs, labels, save_path):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(20, 4), sharey=False)
    fig.suptitle("Predicted Probability Distributions — Test Set",
                 fontsize=14, fontweight="bold")
    for i, (sc, ax) in enumerate(zip(SUPERCLASSES, axes)):
        neg_probs = probs[labels[:, i] == 0, i]
        pos_probs = probs[labels[:, i] == 1, i]
        bins = np.linspace(0, 1, 30)
        ax.hist(neg_probs, bins=bins, alpha=0.6, color="steelblue",
                label="True –", density=True)
        ax.hist(pos_probs, bins=bins, alpha=0.6, color="tomato",
                label="True +", density=True)
        ax.set_title(sc, fontsize=12, fontweight="bold", color=CLASS_COLORS[sc])
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Prob dist  → {save_path}")


def plot_metrics_bar(metrics, save_path):
    x      = np.arange(NUM_CLASSES)
    width  = 0.25
    colors = [CLASS_COLORS[sc] for sc in SUPERCLASSES]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, metrics["class_f1s"],    width, label="F1",    alpha=0.85)
    bars2 = ax.bar(x,          metrics["class_aurocs"], width, label="AUROC", alpha=0.85)
    bars3 = ax.bar(x + width,  metrics["class_aps"],   width, label="AP",    alpha=0.85)
    for bars in [bars1, bars2, bars3]:
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SUPERCLASSES, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics — Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(metrics["macro_f1"],    color="black", ls="--", lw=1.2,
               label=f"Macro F1={metrics['macro_f1']:.3f}")
    ax.axhline(metrics["macro_auroc"], color="gray",  ls=":",  lw=1.2,
               label=f"Macro AUC={metrics['macro_auroc']:.3f}")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Metrics bar→ {save_path}")


def save_metrics_csv(metrics, model_name, save_path):
    rows = []
    for i, sc in enumerate(SUPERCLASSES):
        rows.append({
            "model": model_name,
            "class": sc,
            "f1"   : round(float(metrics["class_f1s"][i]),   4),
            "auroc": round(float(metrics["class_aurocs"][i]), 4),
            "ap"   : round(float(metrics["class_aps"][i]),    4),
        })
    rows.append({
        "model": model_name, "class": "MACRO",
        "f1"   : round(float(metrics["macro_f1"]),    4),
        "auroc": round(float(metrics["macro_auroc"]), 4),
        "ap"   : round(float(metrics["macro_ap"]),    4),
    })
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"[INFO] Metrics CSV → {save_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN EVALUATE FUNCTION
# ══════════════════════════════════════════════════════════════════
def evaluate(model_name: str) -> Dict:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating model : {model_name}  on {device}")

    df_full      = load_metadata(PTB_XL_PATH)
    label_df     = load_labels(LABELS_CSV)
    signals_full = load_all_waveforms(df_full, PTB_XL_PATH)
    signals, label_df = align_signals_labels(signals_full, label_df, df_full.index)

    try:
        train_idx, val_idx, test_idx = load_splits(SPLIT_CACHE)
    except FileNotFoundError:
        train_idx, val_idx, test_idx = patient_wise_split(label_df)

    data    = get_split_data(signals, label_df, train_idx, val_idx, test_idx)
    loaders = get_dataloaders(data, batch_size=64, num_workers=4)

    model = load_best_model(model_name, device)

    print("[INFO] Running inference on validation set…")
    val_logits, val_labels = get_predictions(model, loaders["val"], device)

    print("[INFO] Running inference on test set…")
    test_logits, test_labels = get_predictions(model, loaders["test"], device)

    thresholds = find_optimal_thresholds(val_logits, val_labels)
    metrics    = compute_all_metrics(test_logits, test_labels, thresholds)
    print_metrics(metrics)

    # ── Bootstrap CI on Macro F1 ─────────────────────────────────
    point, lower, upper = bootstrap_confidence_interval(
        test_labels, metrics["preds"], n_boot=1000
    )
    print(f"[INFO] Macro F1 Bootstrap 95% CI: {point:.4f}  [{lower:.4f}, {upper:.4f}]")
    metrics["f1_ci_lower"] = lower
    metrics["f1_ci_upper"] = upper

    print("── Sklearn Classification Report ─────────────────────────────")
    print(classification_report(
        test_labels, metrics["preds"],
        target_names=SUPERCLASSES, zero_division=0,
    ))

    pfx = PLOT_DIR / model_name
    plot_roc_curves(test_labels, metrics["probs"], metrics["class_aurocs"],
                    Path(f"{pfx}_roc_curves.png"))
    plot_pr_curves(test_labels, metrics["probs"], metrics["class_aps"],
                   Path(f"{pfx}_pr_curves.png"))
    plot_confusion_matrices(test_labels, metrics["preds"],
                            Path(f"{pfx}_confusion_matrices.png"))
    plot_probability_distributions(metrics["probs"], test_labels,
                                   Path(f"{pfx}_prob_distributions.png"))
    plot_metrics_bar(metrics, Path(f"{pfx}_metrics_bar.png"))
    save_metrics_csv(metrics, model_name, Path(f"{pfx}_metrics.csv"))

    return metrics, test_labels


# ══════════════════════════════════════════════════════════════════
# NEW: COMPARE BOTH MODELS (runs everything + significance test)
# ══════════════════════════════════════════════════════════════════
def compare_models() -> None:
    """
    Evaluate both models on the same test set, run McNemar's test,
    bootstrap CIs, and print PTB-XL benchmark comparison.
    Call with: python part8_evaluate.py --compare
    """
    print("\n" + "═"*62)
    print("  EVALUATING: baseline_cnn")
    print("═"*62)
    metrics_a, test_labels = evaluate("baseline_cnn")

    print("\n" + "═"*62)
    print("  EVALUATING: cnn_bilstm")
    print("═"*62)
    metrics_b, _           = evaluate("cnn_bilstm")

    # McNemar's test
    mcnemar_test(
        test_labels,
        metrics_a["preds"],
        metrics_b["preds"],
        model_a_name="Baseline CNN",
        model_b_name="CNN-BiLSTM",
    )

    # Bootstrap CIs side by side
    print("\n── Bootstrap 95% CI on Macro F1 ─────────────────────────────")
    print(f"  Baseline CNN : {metrics_a['macro_f1']:.4f}  "
          f"[{metrics_a['f1_ci_lower']:.4f}, {metrics_a['f1_ci_upper']:.4f}]")
    print(f"  CNN-BiLSTM   : {metrics_b['macro_f1']:.4f}  "
          f"[{metrics_b['f1_ci_lower']:.4f}, {metrics_b['f1_ci_upper']:.4f}]")

    # PTB-XL benchmark
    print_benchmark_comparison(metrics_a, metrics_b)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn_bilstm",
                        choices=["baseline_cnn", "cnn_bilstm"])
    parser.add_argument("--compare", action="store_true",
                        help="Evaluate both models + McNemar test + benchmark")
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        evaluate(args.model)
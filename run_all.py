"""
run_all.py – Master Pipeline Script
=====================================
Runs the complete ECG abnormality classification pipeline end-to-end.

Changes from original:
    - weight_decay updated to 3e-4 (matches new part7_train.py default)
    - focal_gamma and focal_alpha added to train config dicts (required by new FocalLoss)
    - Phase 7 evaluation now also calls mcnemar_test + bootstrap CI + benchmark comparison
    - Minor: eval_only path prints cleaner skip message

Usage:
    python run_all.py                          # full pipeline, both models
    python run_all.py --skip_baseline          # skip baseline CNN training
    python run_all.py --skip_xai               # skip XAI (faster)
    python run_all.py --epochs 50 --batch 32
    python run_all.py --eval_only              # skip training, just evaluate
"""

import os
import sys
import time
import argparse
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

import torch


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH      = 32
DEFAULT_LR         = 3e-4
DEFAULT_PATIENCE   = 10
DEFAULT_WORKERS    = 0
DEFAULT_XAI_CASES  = 3
DEFAULT_SEED       = 42

SUPERCLASSES   = ["NORM", "MI", "STTC", "CD", "HYP"]
SEPARATOR      = "═" * 70
THIN_SEP       = "─" * 70


# ─────────────────────────────────────────────
# PRETTY PRINT HELPERS
# ─────────────────────────────────────────────
def banner(text: str, char: str = "═") -> None:
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def section(text: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{THIN_SEP}")
    print(f"  [{ts}]  {text}")
    print(THIN_SEP)


def ok(text: str)   -> None: print(f"  ✅  {text}")
def info(text: str) -> None: print(f"  ℹ️   {text}")
def warn(text: str) -> None: print(f"  ⚠️   {text}")
def fail(text: str) -> None: print(f"  ❌  {text}")


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ─────────────────────────────────────────────
# PHASE TIMER
# ─────────────────────────────────────────────
class PhaseTimer:
    def __init__(self):
        self.times: Dict[str, float] = {}
        self._start: Optional[float] = None
        self._phase: Optional[str]   = None

    def start(self, phase: str) -> None:
        self._phase = phase
        self._start = time.time()

    def stop(self) -> float:
        elapsed = time.time() - self._start
        self.times[self._phase] = elapsed
        return elapsed

    def summary(self) -> None:
        print(f"\n{'─'*40}")
        print("  Phase Timing Summary")
        print(f"{'─'*40}")
        total = 0.0
        for phase, t in self.times.items():
            print(f"  {phase:<35} {format_time(t):>10}")
            total += t
        print(f"{'─'*40}")
        print(f"  {'TOTAL':<35} {format_time(total):>10}")
        print(f"{'─'*40}\n")


# ─────────────────────────────────────────────
# FULL EVALUATION PRINTER
# ─────────────────────────────────────────────
def print_full_evaluation(
    model_name: str,
    metrics   : Dict,
    thresholds: np.ndarray,
    val_metrics: Optional[Dict] = None,
) -> None:
    probs = metrics["probs"]
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"""
╔{'═'*68}╗
║{'ECG SUPERCLASS CLASSIFICATION — FULL EVALUATION REPORT':^68}║
║{'Model: ' + model_name.replace('_',' ').upper():^68}║
║{'Generated: ' + now:^68}║
╠{'═'*68}╣""")

    print(f"║{'':68}║")
    print(f"║  {'PRIMARY METRICS (Test Set)':^66}║")
    print(f"║{'':68}║")
    print(f"║  {'Metric':<30} {'Value':>10} {'Benchmark':>12} {'Status':>10}  ║")
    print(f"║  {'─'*30} {'─'*10} {'─'*12} {'─'*10}  ║")

    def _status(val, good=0.80, ok_=0.70):
        if val >= good: return "✅ Good"
        if val >= ok_:  return "⚠️  Fair"
        return "❌ Low"

    rows = [
        ("Macro F1 Score",      metrics["macro_f1"],    0.80),
        ("Macro AUROC",         metrics["macro_auroc"], 0.85),
        ("Macro Avg Precision", metrics["macro_ap"],    0.80),
    ]
    for name, val, thresh in rows:
        status = _status(val, thresh, thresh - 0.10)
        print(f"║  {name:<30} {val:>10.4f} {thresh:>12.2f} {status:>12}  ║")

    print(f"║{'':68}║")
    print(f"║  {'PER-CLASS METRICS':^66}║")
    print(f"║{'':68}║")
    print(f"║  {'Class':<8} {'F1':>7} {'AUROC':>7} {'AP':>7} "
          f"{'Threshold':>10}  ║")
    print(f"║  {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*10}  ║")

    preds = metrics["preds"]
    for i, sc in enumerate(SUPERCLASSES):
        f1  = metrics["class_f1s"][i]
        auc = metrics["class_aurocs"][i]
        ap  = metrics["class_aps"][i]
        thr = thresholds[i]
        print(f"║  {sc:<8} {f1:>7.4f} {auc:>7.4f} {ap:>7.4f} {thr:>10.3f}  ║")

    print(f"║{'':68}║")
    print(f"║  {'PREDICTION STATISTICS':^66}║")
    print(f"║{'':68}║")
    for i, sc in enumerate(SUPERCLASSES):
        p = probs[:, i]
        print(f"║  {sc:<6}  mean={p.mean():.3f}  std={p.std():.3f}  "
              f"min={p.min():.3f}  max={p.max():.3f}  "
              f"median={np.median(p):.3f}             ║")

    print(f"║{'':68}║")
    print(f"║  {'MULTI-LABEL PREDICTION STATS':^66}║")
    print(f"║{'':68}║")
    n_pred_labels = preds.sum(axis=1)
    for n in range(6):
        cnt = (n_pred_labels == n).sum()
        pct = cnt / len(preds) * 100
        bar = "█" * int(pct / 2)
        print(f"║  {n} label(s) predicted: {cnt:>5} ({pct:5.1f}%)  {bar:<25}       ║")

    print(f"║{'':68}║")
    print(f"╚{'═'*68}╝\n")


def print_comparison_table(all_metrics: Dict[str, Dict]) -> None:
    if len(all_metrics) < 2:
        return

    banner("MODEL COMPARISON TABLE", "═")

    header = f"  {'Metric':<22}"
    for mname in all_metrics:
        header += f"  {mname.replace('_',' ').upper():>18}"
    print(header)
    print("  " + "─" * (22 + 20 * len(all_metrics)))

    for label, key in [
        ("Macro F1",    "macro_f1"),
        ("Macro AUROC", "macro_auroc"),
        ("Macro AP",    "macro_ap"),
    ]:
        row  = f"  {label:<22}"
        vals = [m[key] for m in all_metrics.values()]
        best = max(vals)
        for v in vals:
            marker = " ◀ best" if v == best and len(vals) > 1 else ""
            row += f"  {v:>11.4f}{marker:<7}"
        print(row)

    print("  " + "─" * (22 + 20 * len(all_metrics)))

    print(f"\n  {'Per-class F1':<22}" +
          "".join(f"  {'  '+m.replace('_',' '):>18}" for m in all_metrics))
    for i, sc in enumerate(SUPERCLASSES):
        row  = f"  {sc:<22}"
        vals = [m["class_f1s"][i] for m in all_metrics.values()]
        best = max(vals)
        for v in vals:
            marker = " ◀" if v == best and len(vals) > 1 else "  "
            row += f"  {v:>16.4f}{marker}"
        print(row)

    print(f"\n  {'Per-class AUROC':<22}" +
          "".join(f"  {'  '+m.replace('_',' '):>18}" for m in all_metrics))
    for i, sc in enumerate(SUPERCLASSES):
        row  = f"  {sc:<22}"
        vals = [m["class_aurocs"][i] for m in all_metrics.values()]
        best = max(vals)
        for v in vals:
            marker = " ◀" if v == best and len(vals) > 1 else "  "
            row += f"  {v:>16.4f}{marker}"
        print(row)

    print()


def print_timing_and_system_info(timer: PhaseTimer, cfg: dict) -> None:
    banner("SYSTEM & RUN CONFIGURATION", "═")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_info = ""
    if torch.cuda.is_available():
        cuda_info = (f"\n  GPU              : {torch.cuda.get_device_name(0)}"
                     f"\n  VRAM             : "
                     f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print(f"""
  Device           : {device.upper()}{cuda_info}
  PyTorch version  : {torch.__version__}
  Python version   : {sys.version.split()[0]}

  Epochs           : {cfg['epochs']}
  Batch size       : {cfg['batch']}
  Learning rate    : {cfg['lr']}
  Weight decay     : 3e-4  (fixed — increased for regularisation)
  Focal gamma      : 2.0
  Focal alpha      : 0.25
  Patience         : {cfg['patience']}
  Seed             : {cfg['seed']}
  Workers          : {cfg['workers']}
  Skip baseline    : {cfg['skip_baseline']}
  Skip XAI         : {cfg['skip_xai']}
  Eval only        : {cfg['eval_only']}
""")
    timer.summary()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(cfg: dict) -> None:
    pipeline_start = time.time()
    timer          = PhaseTimer()
    all_metrics    : Dict[str, Dict]       = {}
    all_thresholds : Dict[str, np.ndarray] = {}

    banner("ECG SUPERCLASS CLASSIFICATION — FULL PIPELINE", "═")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models   : {'baseline_cnn + cnn_bilstm' if not cfg['skip_baseline'] else 'cnn_bilstm only'}")
    print(f"  Eval only: {cfg['eval_only']}")

    # ══════════════════════════════════════════
    # PHASE 1 — DATA LOADING
    # ══════════════════════════════════════════
    section("PHASE 1 — Data Loading & Preprocessing")
    timer.start("Phase 1: Data Loading")
    try:
        from part1_load_preprocess import (
            load_metadata, load_all_waveforms, PTB_XL_PATH, sanity_check
        )
        df_full      = load_metadata(PTB_XL_PATH)
        signals_full = load_all_waveforms(df_full, PTB_XL_PATH)
        sanity_check(signals_full, df_full)
        ok(f"Loaded {len(df_full):,} records | "
           f"Signals shape: {signals_full.shape} | "
           f"dtype: {signals_full.dtype}")
    except Exception as e:
        fail(f"Phase 1 failed: {e}"); sys.exit(1)
    elapsed = timer.stop()
    info(f"Phase 1 completed in {format_time(elapsed)}")

    # ══════════════════════════════════════════
    # PHASE 2 — LABEL MAPPING
    # ══════════════════════════════════════════
    section("PHASE 2 — SCP Code → Diagnostic Superclass Mapping")
    timer.start("Phase 2: Label Mapping")
    try:
        from part2_label_mapping import (
            load_scp_statements, build_label_matrix,
            print_label_statistics, compute_class_weights,
            save_labels, SUPERCLASSES
        )
        from part3_data_split import LABELS_CSV
        label_path = Path(LABELS_CSV)

        if label_path.exists():
            from part2_label_mapping import load_labels
            label_df = load_labels(LABELS_CSV)
            info("Loaded label matrix from cache.")
        else:
            scp_statements = load_scp_statements(PTB_XL_PATH)
            label_df       = build_label_matrix(df_full, scp_statements)
            save_labels(label_df)

        print_label_statistics(label_df)
        pos_weights = compute_class_weights(label_df)
        ok(f"Label matrix shape: {label_df[SUPERCLASSES].shape} | "
           f"Records with labels: {len(label_df):,}")
    except Exception as e:
        fail(f"Phase 2 failed: {e}"); sys.exit(1)
    elapsed = timer.stop()
    info(f"Phase 2 completed in {format_time(elapsed)}")

    # ══════════════════════════════════════════
    # PHASE 3 — PATIENT-WISE SPLIT
    # ══════════════════════════════════════════
    section("PHASE 3 — Patient-Wise Train / Val / Test Split")
    timer.start("Phase 3: Data Split")
    try:
        from part3_data_split import (
            align_signals_labels, patient_wise_split,
            load_splits, save_splits, get_split_data,
            print_split_statistics, SPLIT_CACHE
        )
        signals, label_df = align_signals_labels(
            signals_full, label_df, df_full.index
        )
        try:
            train_idx, val_idx, test_idx = load_splits(SPLIT_CACHE)
            info("Loaded split indices from cache.")
        except FileNotFoundError:
            train_idx, val_idx, test_idx = patient_wise_split(label_df)
            save_splits(train_idx, val_idx, test_idx)

        print_split_statistics(label_df, train_idx, val_idx, test_idx)
        data = get_split_data(signals, label_df, train_idx, val_idx, test_idx)
        ok(f"Train: {len(train_idx):,}  |  Val: {len(val_idx):,}  |  "
           f"Test: {len(test_idx):,}")
        ok("Patient-wise integrity verified — zero data leakage")
    except Exception as e:
        fail(f"Phase 3 failed: {e}"); sys.exit(1)
    elapsed = timer.stop()
    info(f"Phase 3 completed in {format_time(elapsed)}")

    # ══════════════════════════════════════════
    # PHASE 4 — DATALOADER VERIFY
    # ══════════════════════════════════════════
    section("PHASE 4 — Dataset & DataLoader Verification")
    timer.start("Phase 4: DataLoader")
    try:
        from part4_dataset import get_dataloaders, print_dataloader_info
        loaders = get_dataloaders(
            data, batch_size=cfg["batch"], num_workers=cfg["workers"],
        )
        print_dataloader_info(loaders)
        ok("DataLoaders verified — correct shapes and augmentation confirmed")
    except Exception as e:
        fail(f"Phase 4 failed: {e}"); sys.exit(1)
    elapsed = timer.stop()
    info(f"Phase 4 completed in {format_time(elapsed)}")

    # ══════════════════════════════════════════
    # PHASE 5 — MODEL SMOKE TEST
    # ══════════════════════════════════════════
    section("PHASE 5 — Model Architecture Smoke Test")
    timer.start("Phase 5: Model Build")
    try:
        from part5_baseline_cnn import build_baseline_cnn, count_parameters
        from part6_cnn_bilstm   import build_cnn_bilstm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy  = torch.randn(2, 12, 5000, device=device)

        m_base = build_baseline_cnn(device)
        out_b  = m_base(dummy)
        ok(f"BaselineCNN   — params: {count_parameters(m_base):>10,} | "
           f"output: {tuple(out_b.shape)}")
        del m_base

        m_bilstm = build_cnn_bilstm(device)
        out_bl   = m_bilstm(dummy)
        ok(f"CNN+BiLSTM    — params: {count_parameters(m_bilstm):>10,} | "
           f"output: {tuple(out_bl.shape)}")
        del m_bilstm

    except Exception as e:
        fail(f"Phase 5 failed: {e}"); sys.exit(1)
    elapsed = timer.stop()
    info(f"Phase 5 completed in {format_time(elapsed)}")

    # ══════════════════════════════════════════
    # PHASE 6 — TRAINING
    # ══════════════════════════════════════════
    if not cfg["eval_only"]:
        from part7_train import train

        # ── FIXED: weight_decay 3e-4, added focal_gamma + focal_alpha ────
        train_cfg_base = {
            "model"        : "baseline_cnn",
            "epochs"       : cfg["epochs"],
            "batch_size"   : cfg["batch"],
            "lr"           : cfg["lr"],
            "weight_decay" : 3e-4,    # ← was 1e-4, now matches part7 default
            "grad_clip"    : 1.0,
            "patience"     : cfg["patience"],
            "num_workers"  : cfg["workers"],
            "seed"         : cfg["seed"],
            "focal_gamma"  : 2.0,     # ← NEW: required by FocalLoss
            "focal_alpha"  : 0.25,    # ← NEW: required by FocalLoss
        }
        # ─────────────────────────────────────────────────────────────────

        if not cfg["skip_baseline"]:
            section("PHASE 6A — Training: Baseline CNN")
            timer.start("Phase 6A: Train BaselineCNN")
            try:
                train(train_cfg_base)
                ok("Baseline CNN training complete")
            except Exception as e:
                warn(f"Baseline CNN training failed: {e}")
            elapsed = timer.stop()
            info(f"Phase 6A completed in {format_time(elapsed)}")

        section("PHASE 6B — Training: CNN + BiLSTM")
        timer.start("Phase 6B: Train CNN+BiLSTM")
        try:
            train_cfg_bilstm = {**train_cfg_base, "model": "cnn_bilstm"}
            train(train_cfg_bilstm)
            ok("CNN+BiLSTM training complete")
        except Exception as e:
            fail(f"CNN+BiLSTM training failed: {e}"); sys.exit(1)
        elapsed = timer.stop()
        info(f"Phase 6B completed in {format_time(elapsed)}")

    else:
        info("Skipping training (--eval_only flag set)")

    # ══════════════════════════════════════════
    # PHASE 7 — EVALUATION
    # ══════════════════════════════════════════
    section("PHASE 7 — Full Test Set Evaluation")
    timer.start("Phase 7: Evaluation")

    from part8_evaluate import (
        get_predictions, find_optimal_thresholds,
        compute_all_metrics, print_metrics,
        bootstrap_confidence_interval,   # ← NEW
        mcnemar_test,                    # ← NEW
        print_benchmark_comparison,      # ← NEW
    )
    from part7_train import load_best_model
    from part4_dataset import ECGDataset
    from sklearn.metrics import classification_report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_to_eval = []
    if not cfg["skip_baseline"]:
        models_to_eval.append("baseline_cnn")
    models_to_eval.append("cnn_bilstm")

    for model_name in models_to_eval:
        ckpt = Path(f"./checkpoints/{model_name}_best.pt")
        if not ckpt.exists():
            warn(f"No checkpoint for {model_name} — skipping evaluation.")
            continue

        banner(f"EVALUATING: {model_name.replace('_',' ').upper()}", "─")

        model = load_best_model(model_name, device)

        val_dataset = ECGDataset(
            data["val"]["signals"], data["val"]["labels"], augment=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=False,
            num_workers=cfg["workers"],
        )
        val_logits, val_labels = get_predictions(model, val_loader, device)
        thresholds = find_optimal_thresholds(val_logits, val_labels)

        test_dataset = ECGDataset(
            data["test"]["signals"], data["test"]["labels"], augment=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False,
            num_workers=cfg["workers"],
        )
        test_logits, test_labels = get_predictions(model, test_loader, device)
        metrics = compute_all_metrics(test_logits, test_labels, thresholds)

        metrics["true_labels"] = test_labels
        metrics["n_positive"]  = test_labels.sum(axis=0).astype(int)
        metrics["n_negative"]  = (1 - test_labels).sum(axis=0).astype(int)

        # ── Bootstrap CI ─────────────────────────────────────────────────
        point, lower, upper = bootstrap_confidence_interval(
            test_labels, metrics["preds"], n_boot=1000
        )
        metrics["f1_ci_lower"] = lower
        metrics["f1_ci_upper"] = upper
        ok(f"Macro F1: {point:.4f}  95% CI: [{lower:.4f}, {upper:.4f}]")
        # ─────────────────────────────────────────────────────────────────

        all_metrics[model_name]    = metrics
        all_thresholds[model_name] = thresholds

        print_full_evaluation(model_name, metrics, thresholds)
        print_metrics(metrics)

        print("── Sklearn Classification Report ────────────────────────────")
        print(classification_report(
            test_labels, metrics["preds"],
            target_names=SUPERCLASSES, zero_division=0,
        ))

    # ── McNemar test + benchmark (only when both models evaluated) ────────
    if "baseline_cnn" in all_metrics and "cnn_bilstm" in all_metrics:
        mcnemar_test(
            all_metrics["cnn_bilstm"]["true_labels"],
            all_metrics["baseline_cnn"]["preds"],
            all_metrics["cnn_bilstm"]["preds"],
            model_a_name="Baseline CNN",
            model_b_name="CNN-BiLSTM",
        )
        print_benchmark_comparison(
            all_metrics["baseline_cnn"],
            all_metrics["cnn_bilstm"],
        )
    # ─────────────────────────────────────────────────────────────────────

    elapsed = timer.stop()
    info(f"Phase 7 completed in {format_time(elapsed)}")

    # ══════════════════════════════════════════
    # PHASE 8 — EVALUATION PLOTS
    # ══════════════════════════════════════════
    section("PHASE 8 — Generating All Evaluation Plots")
    timer.start("Phase 8: Eval Plots")
    try:
        from part8_evaluate import (
            plot_roc_curves, plot_pr_curves,
            plot_confusion_matrices, plot_probability_distributions,
            plot_metrics_bar, save_metrics_csv, PLOT_DIR,
        )
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        for model_name, metrics in all_metrics.items():
            pfx = PLOT_DIR / model_name
            tl  = metrics["true_labels"]

            plot_roc_curves(tl, metrics["probs"], metrics["class_aurocs"],
                            Path(f"{pfx}_roc_curves.png"))
            plot_pr_curves(tl, metrics["probs"], metrics["class_aps"],
                           Path(f"{pfx}_pr_curves.png"))
            plot_confusion_matrices(tl, metrics["preds"],
                                    Path(f"{pfx}_confusion_matrices.png"))
            plot_probability_distributions(metrics["probs"], tl,
                                           Path(f"{pfx}_prob_distributions.png"))
            plot_metrics_bar(metrics, Path(f"{pfx}_metrics_bar.png"))
            save_metrics_csv(metrics, model_name, Path(f"{pfx}_metrics.csv"))
            ok(f"Plots saved for {model_name}")
    except Exception as e:
        warn(f"Phase 8 plots failed: {e}")
    elapsed = timer.stop()
    info(f"Phase 8 completed in {format_time(elapsed)}")

    # ══════════════════════════════════════════
    # PHASE 9 — XAI
    # ══════════════════════════════════════════
    if not cfg["skip_xai"] and "cnn_bilstm" in all_metrics:
        section("PHASE 9 — Explainable AI (Grad-CAM, Saliency, Attention)")
        timer.start("Phase 9: XAI")
        try:
            from part9_xai import run_xai
            run_xai("cnn_bilstm", n_cases=cfg["xai_cases"])
            ok(f"XAI plots generated for {cfg['xai_cases']} cases")
        except Exception as e:
            warn(f"Phase 9 XAI failed: {e}")
        elapsed = timer.stop()
        info(f"Phase 9 completed in {format_time(elapsed)}")
    else:
        if cfg["skip_xai"]:
            info("Skipping XAI (--skip_xai flag set)")

    # ══════════════════════════════════════════
    # PHASE 10 — EXPORT & FINAL PLOTS
    # ══════════════════════════════════════════
    section("PHASE 10 — Model Export & Final Report")
    timer.start("Phase 10: Export & Report")
    try:
        from part10_save_and_plot import (
            export_checkpoint_metadata, export_torchscript,
            export_onnx, copy_best_checkpoint,
            plot_training_curves, load_training_log,
            plot_model_comparison, plot_report_card,
            PLOTS_DIR, EXPORT_DIR, REPORT_DIR,
        )
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model_name in all_metrics:
            export_checkpoint_metadata(model_name)
            try:
                export_torchscript(model_name, device)
            except Exception as e:
                warn(f"TorchScript export failed: {e}")
            try:
                export_onnx(model_name, device)
            except Exception as e:
                warn(f"ONNX export failed: {e}")
            copy_best_checkpoint(model_name)

            try:
                log = load_training_log(model_name)
                plot_training_curves(
                    log, model_name,
                    PLOTS_DIR / f"{model_name}_training_curves.png"
                )
            except FileNotFoundError:
                warn(f"No training log for {model_name}")

        if all_metrics:
            plot_model_comparison(all_metrics,
                                  PLOTS_DIR / "model_comparison.png")
            plot_report_card(all_metrics,
                             REPORT_DIR / "performance_report_card.png")

            rows = []
            for mname, metrics in all_metrics.items():
                for i, sc in enumerate(SUPERCLASSES):
                    rows.append({
                        "model": mname, "class": sc,
                        "f1"   : round(float(metrics["class_f1s"][i]),   4),
                        "auroc": round(float(metrics["class_aurocs"][i]),4),
                        "ap"   : round(float(metrics["class_aps"][i]),   4),
                    })
                rows.append({
                    "model": mname, "class": "MACRO",
                    "f1"   : round(float(metrics["macro_f1"]),    4),
                    "auroc": round(float(metrics["macro_auroc"]), 4),
                    "ap"   : round(float(metrics["macro_ap"]),    4),
                })
            csv_out = REPORT_DIR / "all_models_metrics.csv"
            pd.DataFrame(rows).to_csv(csv_out, index=False)
            ok(f"Combined metrics CSV → {csv_out}")

        ok("All exports and plots complete")
    except Exception as e:
        warn(f"Phase 10 partially failed: {e}")
    elapsed = timer.stop()
    info(f"Phase 10 completed in {format_time(elapsed)}")

    # ══════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════
    banner("PIPELINE COMPLETE — FINAL SUMMARY", "═")
    print_comparison_table(all_metrics)
    print_timing_and_system_info(timer, cfg)

    total_time = time.time() - pipeline_start
    print(f"\n  🏁  Total wall-clock time : {format_time(total_time)}")
    print(f"  📁  All outputs in        : {Path('.').resolve()}")
    print(f"  📊  Report card           : ./reports/performance_report_card.png")
    print(f"  🧠  XAI plots             : ./xai_plots/")
    print(f"\n{'═'*70}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ECG Classification — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python run_all.py                          # full pipeline
          python run_all.py --skip_baseline          # skip baseline CNN
          python run_all.py --skip_xai               # faster, no XAI plots
          python run_all.py --eval_only              # evaluate existing models
          python run_all.py --epochs 30 --batch 16   # custom hyperparams
        """)
    )
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",         type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--patience",      type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--workers",       type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--xai_cases",     type=int,   default=DEFAULT_XAI_CASES)
    parser.add_argument("--seed",          type=int,   default=DEFAULT_SEED)
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_xai",      action="store_true")
    parser.add_argument("--eval_only",     action="store_true")

    args = parser.parse_args()
    cfg  = vars(args)
    run_pipeline(cfg)
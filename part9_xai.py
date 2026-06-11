"""
Part 9 – Explainable AI: Grad-CAM, Saliency, Irregularity Detection, Clinical Report
======================================================================================
Depends on: part1–part8

New features added:
    1. Irregularity Detection
       - Automatic thresholding: flag regions where saliency > mean + 2*std
       - Extracts start/end timestamps in milliseconds (500Hz → 2ms/sample)
       - Per-lead and averaged across all leads

    2. Severity Scoring
       - Score = predicted_probability × mean_saliency_in_flagged_region
       - Bands: Low (<0.2) / Moderate (0.2–0.4) / High (0.4–0.6) / Critical (>0.6)

    3. Clinical Text Report (doctor-readable .txt)
       - Record ID, predicted diagnoses with %, certainty bands
       - Per-region: leads affected, timestamp ms, severity score
       - Overall clinical impression paragraph
       - Saved to xai_plots/reports/

    4. Per-lead plot with irregularity bands
       - 12 individual lead plots + averaged saliency timeline
       - Flagged regions highlighted with severity colour

    Bug fixes carried forward:
       - OOB index fix (test_idx filtered to labeled subset)
       - BatchNorm/LSTM hybrid mode (set_xai_mode)

Run standalone:
    python part9_xai.py --model cnn_bilstm --n_cases 3
"""

import argparse
import textwrap
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from part1_load_preprocess import load_metadata, load_all_waveforms, PTB_XL_PATH
from part2_label_mapping   import load_labels, SUPERCLASSES
from part3_data_split      import (align_signals_labels, load_splits,
                                   get_split_data, patient_wise_split,
                                   LABELS_CSV, SPLIT_CACHE)
from part4_dataset         import ECGDataset
from part5_baseline_cnn    import BaselineCNN
from part6_cnn_bilstm      import CNNBiLSTM
from part7_train           import load_best_model

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
XAI_DIR       = Path("./xai_plots")
REPORT_DIR    = Path("./xai_plots/reports")
SAMPLING_RATE = 500
MS_PER_SAMPLE = 1000 / SAMPLING_RATE   # 2.0 ms per sample
LEAD_NAMES    = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
CLASS_COLORS  = {
    "NORM": "#4C9BE8", "MI": "#E8694C", "STTC": "#6DBE6D",
    "CD"  : "#B36DBE", "HYP": "#E8C24C",
}
LEAD_II_IDX = 1

# Severity bands: (min_score, label, hex_color)
SEVERITY_BANDS = [
    (0.6, "Critical",  "#FF0000"),
    (0.4, "High",      "#FF6600"),
    (0.2, "Moderate",  "#FFAA00"),
    (0.0, "Low",       "#FFDD00"),
]

CLASS_CLINICAL = {
    "NORM": "Normal sinus rhythm — no diagnostic abnormality detected",
    "MI"  : "Myocardial Infarction — evidence of ischaemic cardiac injury",
    "STTC": "ST-segment/T-wave Changes — repolarisation abnormality",
    "CD"  : "Conduction Disturbance — abnormal electrical conduction pathway",
    "HYP" : "Hypertrophy — increased cardiac muscle mass",
}


def certainty_band(prob: float) -> str:
    if prob >= 0.85: return "Very High Confidence"
    if prob >= 0.70: return "High Confidence"
    if prob >= 0.50: return "Moderate Confidence"
    if prob >= 0.35: return "Low Confidence"
    return "Uncertain"


def severity_from_score(score: float) -> Tuple[str, str]:
    for thresh, label, color in SEVERITY_BANDS:
        if score >= thresh:
            return label, color
    return "Low", "#FFDD00"


# ══════════════════════════════════════════════════════════════════
# XAI MODE SETTER (BatchNorm=eval, LSTM=train)
# ══════════════════════════════════════════════════════════════════
def set_xai_mode(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            module.train()


# ══════════════════════════════════════════════════════════════════
# R-PEAK DETECTION
# ══════════════════════════════════════════════════════════════════
def detect_r_peaks(signal, fs=500, min_height=0.3, min_dist=150):
    squared   = signal ** 2
    win       = max(1, int(0.025 * fs))
    kernel    = np.ones(win) / win
    smoothed  = np.convolve(squared, kernel, mode="same")
    threshold = min_height * smoothed.max()
    above     = smoothed > threshold
    r_peaks, last_peak = [], -min_dist
    i = 0
    while i < len(smoothed):
        if above[i]:
            j = i
            while j < len(smoothed) and above[j]:
                j += 1
            local_max = i + np.argmax(smoothed[i:j])
            if local_max - last_peak >= min_dist:
                r_peaks.append(local_max)
                last_peak = local_max
            i = j
        else:
            i += 1
    return np.array(r_peaks, dtype=int)


def r_peaks_to_attention_cols(r_peaks, original_length=5000, compressed_length=157):
    scale = compressed_length / original_length
    return np.clip(np.round(r_peaks * scale).astype(int), 0, compressed_length - 1)


# ══════════════════════════════════════════════════════════════════
# IRREGULARITY DETECTION
# ══════════════════════════════════════════════════════════════════
def detect_irregularities(
    saliency       : np.ndarray,   # (12, 5000)
    pred_prob      : float,
    n_std          : float = 2.0,
    min_duration_ms: float = 20.0,
) -> List[Dict]:
    """
    Flag time regions where averaged saliency > mean + n_std * std.

    Returns list of region dicts with timestamps (ms), affected leads,
    mean saliency, severity score and band.
    """
    avg_sal   = saliency.mean(axis=0)          # (5000,)
    mean_s    = avg_sal.mean()
    std_s     = avg_sal.std()
    threshold = mean_s + n_std * std_s
    above     = avg_sal > threshold

    min_samples = max(1, int(min_duration_ms / MS_PER_SAMPLE))
    regions = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= min_samples:
                region_sal    = saliency[:, i:j]              # (12, len)
                mean_sal_reg  = float(region_sal.mean())
                lead_means    = region_sal.mean(axis=1)        # (12,)
                global_means  = saliency.mean(axis=1)          # (12,)
                aff_idx       = np.where(lead_means > global_means * 1.2)[0]
                aff_leads     = ([LEAD_NAMES[li] for li in aff_idx]
                                 if len(aff_idx) > 0
                                 else [LEAD_NAMES[int(np.argmax(lead_means))]])
                score         = float(pred_prob * mean_sal_reg)
                band, color   = severity_from_score(score)

                regions.append({
                    "start_ms"      : round(i * MS_PER_SAMPLE, 1),
                    "end_ms"        : round(j * MS_PER_SAMPLE, 1),
                    "duration_ms"   : round((j - i) * MS_PER_SAMPLE, 1),
                    "start_sample"  : i,
                    "end_sample"    : j,
                    "affected_leads": aff_leads,
                    "severity_score": round(score, 4),
                    "severity_band" : band,
                    "severity_color": color,
                    "mean_saliency" : round(mean_sal_reg, 4),
                    "threshold_used": round(float(threshold), 4),
                })
            i = j
        else:
            i += 1
    return regions


# ══════════════════════════════════════════════════════════════════
# CLINICAL TEXT REPORT
# ══════════════════════════════════════════════════════════════════
def generate_clinical_report(
    record_id               : int,
    case_idx                : int,
    true_labels             : np.ndarray,
    pred_probs              : np.ndarray,
    irregularities_per_class: Dict[str, List[Dict]],
    saliency                : np.ndarray,
    save_path               : Path,
) -> str:
    """Generate doctor-readable clinical report and save as .txt"""
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines += [
        "=" * 70,
        "       ECG DIAGNOSTIC ANALYSIS REPORT",
        "       AI-Assisted Interpretation — For Clinical Review Only",
        "=" * 70,
        f"  Report Generated : {now}",
        f"  Record ID        : PTB-XL #{record_id}",
        f"  Analysis Case    : {case_idx}",
        f"  Model            : CNN+BiLSTM (Macro AUROC = 0.908)",
        f"  Signal           : 12-lead ECG, 10s @ 500 Hz (2 ms/sample)",
        "=" * 70, "",
    ]

    # Section 1: Predicted Diagnoses
    lines += ["SECTION 1: PREDICTED DIAGNOSES", "-" * 40]
    positive_classes = []
    for i, sc in enumerate(SUPERCLASSES):
        prob     = float(pred_probs[i])
        true_val = int(true_labels[i])
        cert     = certainty_band(prob)
        marker   = "► DETECTED" if prob >= 0.35 else "  Not detected"
        gt_str   = "POSITIVE" if true_val else "negative"
        lines.append(
            f"  {sc:<6}  {prob*100:>5.1f}%  [{cert:<22}]  "
            f"{marker}  [Ground truth: {gt_str}]"
        )
        if prob >= 0.35:
            positive_classes.append((sc, prob))
    lines += [""]

    # Section 2: Clinical Descriptions
    lines += ["SECTION 2: CLINICAL CLASS DESCRIPTIONS", "-" * 40]
    if positive_classes:
        for sc, prob in positive_classes:
            lines.append(f"  {sc} ({prob*100:.1f}%): {CLASS_CLINICAL[sc]}")
    else:
        lines.append("  No diagnostic abnormality detected above confidence threshold.")
    lines += [""]

    # Section 3: Irregularity Regions
    lines += [
        "SECTION 3: DETECTED IRREGULARITY REGIONS",
        "-" * 40,
        "  Method: Saliency threshold = mean + 2×SD across all leads.",
        "  Severity = predicted_probability × mean_saliency_in_region.",
        "  Timestamps reported in milliseconds (2 ms/sample at 500 Hz).",
        "",
    ]
    total_regions = 0
    for sc, regions in irregularities_per_class.items():
        if not regions:
            continue
        prob = float(pred_probs[SUPERCLASSES.index(sc)])
        lines += [f"  ── Class: {sc}  (P = {prob*100:.1f}%) ──"]
        for ri, reg in enumerate(regions):
            total_regions += 1
            lines += [
                f"    Region {ri+1}:",
                f"      Timestamp     : {reg['start_ms']:.0f} ms "
                f"— {reg['end_ms']:.0f} ms  "
                f"(duration: {reg['duration_ms']:.0f} ms)",
                f"      Leads affected: {', '.join(reg['affected_leads'])}",
                f"      Mean saliency : {reg['mean_saliency']:.4f}  "
                f"(threshold: {reg['threshold_used']:.4f})",
                f"      Severity      : {reg['severity_score']:.4f}  "
                f"[{reg['severity_band']}]",
            ]
        lines += [""]
    if total_regions == 0:
        lines += ["  No significant irregularity regions detected above threshold.", ""]

    # Section 4: Lead-wise Saliency
    lines += ["SECTION 4: LEAD-WISE SALIENCY RANKING", "-" * 40]
    lead_imp      = saliency.mean(axis=1)
    lead_imp_norm = lead_imp / (lead_imp.max() + 1e-8)
    sorted_leads  = np.argsort(lead_imp_norm)[::-1]
    for rank, li in enumerate(sorted_leads[:6]):
        bar = "█" * int(lead_imp_norm[li] * 20)
        lines.append(f"  {rank+1}. {LEAD_NAMES[li]:<4}  {lead_imp_norm[li]:.3f}  {bar}")
    lines += [""]

    # Section 5: Clinical Impression
    lines += ["SECTION 5: OVERALL CLINICAL IMPRESSION", "-" * 40]

    if not positive_classes:
        impression = (
            "The AI model did not detect any significant diagnostic abnormality "
            "in this 12-lead ECG recording with confidence above the clinical "
            "threshold. The signal characteristics are consistent with normal "
            "sinus rhythm. Clinical correlation is recommended."
        )
    else:
        class_list = " and ".join(
            [f"{sc} ({p*100:.0f}%)" for sc, p in positive_classes]
        )
        top_leads = ", ".join([LEAD_NAMES[li] for li in sorted_leads[:3]])

        if total_regions == 0:
            region_desc = (
                "No discrete irregularity regions were identified by "
                "saliency analysis above the statistical threshold."
            )
        elif total_regions <= 2:
            region_desc = (
                f"Saliency analysis identified {total_regions} region(s) "
                f"of interest with elevated model attention, suggesting "
                f"localised ECG abnormality."
            )
        else:
            region_desc = (
                f"Saliency analysis identified {total_regions} distinct "
                f"regions of elevated model attention, suggesting diffuse "
                f"ECG abnormality across the recording."
            )

        critical = [r for regs in irregularities_per_class.values()
                    for r in regs if r["severity_band"] == "Critical"]
        crit_desc = (
            f" {len(critical)} region(s) reached Critical severity — "
            f"immediate clinical review is strongly advised."
            if critical else ""
        )

        impression = (
            f"The AI model has identified findings consistent with {class_list} "
            f"in this 12-lead ECG recording. The model assigned highest diagnostic "
            f"relevance to leads {top_leads}, which exhibited the greatest gradient "
            f"saliency during classification. {region_desc}{crit_desc} "
            f"These findings are generated by an automated deep learning system "
            f"and must be interpreted in the context of the patient's clinical "
            f"history, symptoms, and physical examination. Formal interpretation "
            f"by a qualified cardiologist is required before any clinical decision "
            f"is made."
        )

    for line in textwrap.wrap(impression, width=66):
        lines.append(f"  {line}")
    lines += [""]

    # Disclaimer
    lines += [
        "=" * 70,
        "  DISCLAIMER",
        "-" * 40,
        "  This report is generated by an AI research model trained on",
        "  the PTB-XL dataset. It is intended for research and educational",
        "  purposes only. It does NOT constitute a medical diagnosis.",
        "  All findings must be reviewed and confirmed by a licensed",
        "  medical professional before any clinical action is taken.",
        "=" * 70,
    ]

    report = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [REPORT] → {save_path.name}")
    return report


# ══════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════
class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._fwd_hook   = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook   = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, signal, class_idx, original_length=5000):
        set_xai_mode(self.model)
        with torch.enable_grad():
            signal = signal.requires_grad_(True)
            logits = self.model(signal)
            self.model.zero_grad()
            logits[0, class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=-1, keepdim=True)
        cam     = (weights * self.activations).sum(dim=1)
        cam     = torch.relu(cam).squeeze(0).cpu().numpy()
        cam     = np.interp(np.linspace(0, len(cam)-1, original_length),
                            np.arange(len(cam)), cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_gradcam_target_layer(model):
    if isinstance(model, BaselineCNN):   return model.stage4[-1]
    elif isinstance(model, CNNBiLSTM):   return model.cnn_body[-1]
    else: raise ValueError(f"Unsupported model: {type(model)}")


# ══════════════════════════════════════════════════════════════════
# SALIENCY
# ══════════════════════════════════════════════════════════════════
def compute_saliency(model, signal, class_idx):
    set_xai_mode(model)
    with torch.enable_grad():
        signal = signal.requires_grad_(True)
        logit  = model(signal)[0, class_idx]
        model.zero_grad()
        logit.backward()
    return signal.grad.abs().squeeze(0).cpu().numpy()


def aggregate_saliency(saliency):
    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-8 else x
    return _norm(saliency.max(axis=0)), _norm(saliency.mean(axis=1))


def compute_integrated_gradients(model, signal, class_idx, n_steps=50):
    set_xai_mode(model)
    baseline = torch.zeros_like(signal)
    ig = torch.zeros_like(signal)
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        with torch.enable_grad():
            interp = (baseline + alpha * (signal - baseline)).requires_grad_(True)
            logit  = model(interp)[0, class_idx]
            model.zero_grad()
            logit.backward()
            ig += interp.grad.detach()
    ig = ig * (signal - baseline) / n_steps
    return ig.abs().squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════
# FIND CASES
# ══════════════════════════════════════════════════════════════════
def find_abnormal_cases(model, dataset, device, n_cases=3, threshold=0.5):
    model.eval()
    cases = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sig, lab = dataset[idx]
            sig_t    = sig.unsqueeze(0).to(device)
            probs    = torch.sigmoid(model(sig_t)).squeeze(0).cpu().numpy()
            preds    = (probs >= threshold).astype(int)
            labels   = lab.numpy().astype(int)
            norm_idx = SUPERCLASSES.index("NORM")
            if labels[norm_idx] == 1 or labels.sum() == 0:
                continue
            abnormal = [i for i, sc in enumerate(SUPERCLASSES)
                        if sc != "NORM" and labels[i] == 1]
            correct  = [i for i in abnormal if preds[i] == 1]
            if not correct:
                continue
            cases.append({
                "dataset_idx"    : idx,
                "signal_tensor"  : sig.unsqueeze(0),
                "true_labels"    : labels,
                "pred_probs"     : probs,
                "correct_classes": correct,
            })
            if len(cases) >= n_cases:
                break
    if not cases:
        raise RuntimeError("No correctly-predicted abnormal cases found.")
    print(f"[INFO] Found {len(cases)} abnormal cases.")
    return cases


# ══════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════
def plot_gradcam_overlay(signal_np, cam, class_name, case_idx,
                         true_labels, pred_probs, save_path, beat_times=None):
    t   = np.arange(signal_np.shape[1]) / SAMPLING_RATE
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle(
        f"Grad-CAM — Class: {class_name}  |  Case {case_idx}\n"
        f"True: {[SUPERCLASSES[i] for i,v in enumerate(true_labels) if v]}  |  "
        f"Pred: {dict(zip(SUPERCLASSES, pred_probs.round(2)))}",
        fontsize=11, color="white", y=0.99
    )
    cmap = plt.get_cmap("inferno")
    norm_c = mcolors.Normalize(vmin=0, vmax=1)
    for i, lead in enumerate(LEAD_NAMES):
        ax = fig.add_subplot(6, 2, i+1)
        ax.set_facecolor("#111111")
        for j in range(len(t)-1):
            ax.axvspan(t[j], t[j+1], alpha=cam[j]*0.55, color=cmap(cam[j]), linewidth=0)
        ax.plot(t, signal_np[i], color="white", lw=0.7, alpha=0.95)
        if beat_times is not None:
            for bt in beat_times:
                ax.axvline(x=bt, color="#00ff88", lw=0.6, alpha=0.5, linestyle="--")
        ax.set_ylabel(lead, color="white", fontsize=9, labelpad=2)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        if i < 10: ax.set_xticklabels([])
        else: ax.set_xlabel("Time (s)", color="white", fontsize=8)
    sm   = cm.ScalarMappable(cmap=cmap, norm=norm_c)
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.015, pad=0.01)
    cbar.set_label("Grad-CAM Importance", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PLOT] Grad-CAM  → {save_path.name}")


def plot_saliency_overlay(signal_np, saliency, class_name, case_idx, save_path):
    _, per_lead = aggregate_saliency(saliency)
    t = np.arange(signal_np.shape[1]) / SAMPLING_RATE
    fig, axes = plt.subplots(1, 2, figsize=(18, 5),
                             gridspec_kw={"width_ratios": [4, 1]})
    fig.suptitle(f"Saliency Map — Class: {class_name}  |  Case {case_idx}",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    im = ax.imshow(saliency, aspect="auto", cmap="hot",
                   extent=[t[0], t[-1], len(LEAD_NAMES)-0.5, -0.5],
                   interpolation="bilinear")
    ax.set_yticks(range(len(LEAD_NAMES)))
    ax.set_yticklabels(LEAD_NAMES, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title("Input Gradient Saliency  (leads × time)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02, label="Normalised |∂logit/∂input|")
    for i in range(len(LEAD_NAMES)):
        sig_norm = signal_np[i] / (np.abs(signal_np[i]).max() + 1e-8) * 0.35
        ax.plot(t, i + sig_norm, color="cyan", lw=0.4, alpha=0.45)
    ax2 = axes[1]
    ax2.barh(range(len(LEAD_NAMES)), per_lead, color=plt.cm.hot(per_lead))
    ax2.set_yticks(range(len(LEAD_NAMES)))
    ax2.set_yticklabels(LEAD_NAMES, fontsize=9)
    ax2.set_xlabel("Mean Saliency", fontsize=10)
    ax2.set_title("Lead Importance", fontsize=11)
    ax2.invert_yaxis(); ax2.set_xlim(0, 1.05)
    ax2.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saliency  → {save_path.name}")


def plot_per_lead_with_irregularities(
    signal_np : np.ndarray,   # (12, 5000)
    saliency  : np.ndarray,   # (12, 5000)
    regions   : List[Dict],
    class_name: str,
    case_idx  : int,
    save_path : Path,
) -> None:
    """
    12 per-lead plots with ECG waveform, saliency heatmap background,
    and coloured severity bands at flagged irregularity regions.
    Bottom row: averaged saliency timeline with threshold line.
    """
    t        = np.arange(signal_np.shape[1]) / SAMPLING_RATE
    cmap_sal = plt.get_cmap("hot")

    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor("#0a0a0a")
    fig.suptitle(
        f"Per-Lead ECG + Saliency + Irregularity Regions — "
        f"Class: {class_name} | Case {case_idx}",
        fontsize=13, color="white", fontweight="bold", y=0.99
    )
    gs = fig.add_gridspec(7, 2, hspace=0.45, wspace=0.3)

    for i, lead in enumerate(LEAD_NAMES):
        row = i // 2; col = i % 2
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#111111")

        sal_lead = saliency[i]
        sal_norm = (sal_lead - sal_lead.min()) / (sal_lead.max() - sal_lead.min() + 1e-8)
        for j in range(len(t)-1):
            ax.axvspan(t[j], t[j+1], alpha=sal_norm[j]*0.5,
                       color=cmap_sal(sal_norm[j]), linewidth=0)

        ax.plot(t, signal_np[i], color="white", lw=0.8, alpha=0.9, zorder=3)

        for reg in regions:
            ts = reg["start_sample"] / SAMPLING_RATE
            te = reg["end_sample"]   / SAMPLING_RATE
            ax.axvspan(ts, te, alpha=0.22, color=reg["severity_color"], zorder=2)
            ax.axvline(ts, color=reg["severity_color"], lw=0.9, alpha=0.85, zorder=4)
            ax.axvline(te, color=reg["severity_color"], lw=0.9, alpha=0.85, zorder=4)

        ax.set_ylabel(lead, color="white", fontsize=9)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        if i < 10: ax.set_xticklabels([])
        else: ax.set_xlabel("Time (s)", color="white", fontsize=8)

    # Averaged saliency timeline
    ax_avg = fig.add_subplot(gs[6, :])
    ax_avg.set_facecolor("#111111")
    avg_sal   = saliency.mean(axis=0)
    mean_s    = avg_sal.mean()
    std_s     = avg_sal.std()
    thresh    = mean_s + 2 * std_s

    ax_avg.fill_between(t, avg_sal, alpha=0.6, color="#4C9BE8", label="Avg saliency")
    ax_avg.axhline(thresh, color="red",  lw=1.2, linestyle="--",
                   label=f"Threshold (mean+2σ = {thresh:.4f})")
    ax_avg.axhline(mean_s, color="gray", lw=0.8, linestyle=":",
                   label=f"Mean = {mean_s:.4f}")

    for reg in regions:
        ts = reg["start_sample"] / SAMPLING_RATE
        te = reg["end_sample"]   / SAMPLING_RATE
        ax_avg.axvspan(ts, te, alpha=0.3, color=reg["severity_color"])
        ax_avg.text((ts+te)/2, thresh * 1.02,
                    f"{reg['severity_band']}\n{reg['start_ms']:.0f}–{reg['end_ms']:.0f}ms",
                    ha="center", va="bottom", fontsize=6.5,
                    color=reg["severity_color"], fontweight="bold")

    ax_avg.set_xlim(t[0], t[-1])
    ax_avg.set_xlabel("Time (s)", color="white", fontsize=10)
    ax_avg.set_ylabel("Avg Saliency", color="white", fontsize=9)
    ax_avg.set_title(
        "Averaged Saliency Timeline (all 12 leads) — Flagged Regions Highlighted",
        color="white", fontsize=10
    )
    ax_avg.tick_params(colors="white", labelsize=8)
    ax_avg.legend(fontsize=8, loc="upper right",
                  facecolor="#222", labelcolor="white")
    for spine in ax_avg.spines.values():
        spine.set_edgecolor("#333")

    # Severity legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=f"{lb} (score≥{t:.1f})")
                      for t, lb, c in SEVERITY_BANDS]
    ax_avg.legend(handles=legend_patches + ax_avg.get_legend_handles_labels()[0],
                  fontsize=7, loc="upper left",
                  facecolor="#222", labelcolor="white", ncol=2)

    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Per-lead  → {save_path.name}")


def plot_per_class_panel(signal_np, all_cams, true_labels, pred_probs,
                         case_idx, save_path):
    t   = np.arange(signal_np.shape[1]) / SAMPLING_RATE
    sig = signal_np[1]
    fig, axes = plt.subplots(1, len(SUPERCLASSES), figsize=(20, 3.5), sharey=True)
    fig.suptitle(
        f"Per-Class Grad-CAM Comparison (Lead II) — Case {case_idx}\n"
        f"True: {[SUPERCLASSES[i] for i,v in enumerate(true_labels) if v]}",
        fontsize=12, fontweight="bold",
    )
    cmap = plt.get_cmap("inferno")
    for ax, sc in zip(axes, SUPERCLASSES):
        cam = all_cams.get(sc, np.zeros(len(t)))
        for j in range(len(t)-1):
            ax.axvspan(t[j], t[j+1], alpha=cam[j]*0.65,
                       color=cmap(cam[j]), linewidth=0)
        ax.plot(t, sig,
                color="white" if cam.mean() > 0.4 else "black", lw=0.8, zorder=2)
        ax.set_facecolor("#1a1a2e" if true_labels[SUPERCLASSES.index(sc)] else "#f5f5f5")
        label_str = (f"True={true_labels[SUPERCLASSES.index(sc)]}  "
                     f"P={pred_probs[SUPERCLASSES.index(sc)]:.2f}")
        ax.set_title(f"{sc}\n{label_str}", fontsize=9, fontweight="bold",
                     color=CLASS_COLORS[sc])
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_xlim(t[0], t[-1])
        for spine in ax.spines.values():
            spine.set_edgecolor(CLASS_COLORS[sc])
            spine.set_linewidth(2)
    axes[0].set_ylabel("Amplitude", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Panel     → {save_path.name}")


def plot_attention_map(attn_weights, class_name, case_idx, save_path,
                       r_peaks_compressed=None, beat_times=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_weights, cmap="viridis", aspect="auto",
                   interpolation="nearest")
    ax.set_title(f"Temporal Self-Attention — Class: {class_name}  |  Case {case_idx}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Key Time Step (compressed)", fontsize=11)
    ax.set_ylabel("Query Time Step (compressed)", fontsize=11)
    fig.colorbar(im, ax=ax, label="Attention Weight")
    if r_peaks_compressed is not None and len(r_peaks_compressed) > 0:
        T = attn_weights.shape[0]
        for bn, col in enumerate(r_peaks_compressed):
            if 0 <= col < T:
                ax.axvline(x=col, color="red", lw=1.2, linestyle="--", alpha=0.75)
                ax.axhline(y=col, color="red", lw=0.6, linestyle=":", alpha=0.4)
                ax.text(col, -2, f"B{bn+1}", ha="center", va="top",
                        fontsize=7, color="red", fontweight="bold", clip_on=False)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0], color="red", lw=1.5, linestyle="--",
                                   label="R-peak (beat position)")],
                  loc="upper left", fontsize=9, framealpha=0.7)
        if beat_times is not None:
            beat_str = ", ".join([f"B{i+1}={bt:.2f}s"
                                   for i, bt in enumerate(beat_times)])
            ax.set_xlabel(f"Key Time Step (compressed)\nBeat times: {beat_str}",
                          fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Attention → {save_path.name}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def run_xai(model_name: str, n_cases: int = 3) -> None:
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] XAI — model: {model_name}  device: {device}")

    df_full      = load_metadata(PTB_XL_PATH)
    label_df     = load_labels(LABELS_CSV)
    signals_full = load_all_waveforms(df_full, PTB_XL_PATH)
    signals, label_df = align_signals_labels(signals_full, label_df, df_full.index)

    try:
        _, _, test_idx = load_splits(SPLIT_CACHE)
    except FileNotFoundError:
        _, _, test_idx = patient_wise_split(label_df)

    # OOB fix — remap to labeled subset
    test_idx = np.array(test_idx)
    test_idx = test_idx[test_idx < len(signals)]
    if len(test_idx) == 0:
        raise RuntimeError("test_idx is empty after OOB filtering.")
    print(f"[INFO] test_idx after OOB filter: {len(test_idx)} records")

    test_signals    = signals[test_idx]
    test_labels_arr = label_df.iloc[test_idx][SUPERCLASSES].values.astype(np.float32)
    test_record_ids = label_df.iloc[test_idx].index.tolist()
    test_dataset    = ECGDataset(test_signals, test_labels_arr, augment=False)

    model = load_best_model(model_name, device)
    model.eval()

    target_layer = get_gradcam_target_layer(model)
    gradcam      = GradCAM1D(model, target_layer)
    cases        = find_abnormal_cases(model, test_dataset, device, n_cases=n_cases)

    for ci, case in enumerate(cases):
        sig_t      = case["signal_tensor"].to(device)
        sig_np     = sig_t.squeeze(0).cpu().numpy()
        true_lab   = case["true_labels"]
        pred_probs = case["pred_probs"]
        focus_cls  = case["correct_classes"][0]
        cls_name   = SUPERCLASSES[focus_cls]
        ds_idx     = case["dataset_idx"]
        record_id  = test_record_ids[ds_idx] if ds_idx < len(test_record_ids) else "unknown"

        print(f"\n── Case {ci} | Record: {record_id} | Focus: {cls_name} ─────")

        # R-peaks
        r_peaks_orig = detect_r_peaks(sig_np[LEAD_II_IDX], fs=SAMPLING_RATE)
        beat_times   = r_peaks_orig / SAMPLING_RATE
        comp_len     = 157 if model_name == "cnn_bilstm" else 79
        r_peaks_comp = r_peaks_to_attention_cols(r_peaks_orig, sig_np.shape[1], comp_len)
        print(f"  Detected {len(r_peaks_orig)} R-peaks at: {beat_times.round(2)} s")

        # 1. Grad-CAM
        all_cams = {sc: gradcam.compute(sig_t.clone(), c, 5000)
                    for c, sc in enumerate(SUPERCLASSES)}
        plot_gradcam_overlay(sig_np, all_cams[cls_name], cls_name, ci,
                             true_lab, pred_probs,
                             XAI_DIR / f"case{ci}_{cls_name}_gradcam.png",
                             beat_times=beat_times)

        # 2. Vanilla saliency (focus class)
        saliency = compute_saliency(model, sig_t.clone(), focus_cls)
        plot_saliency_overlay(sig_np, saliency, cls_name, ci,
                              XAI_DIR / f"case{ci}_{cls_name}_saliency.png")

        # 3. Integrated gradients
        ig = compute_integrated_gradients(model, sig_t.clone(), focus_cls, n_steps=30)
        plot_saliency_overlay(sig_np, ig, cls_name + " (IG)", ci,
                              XAI_DIR / f"case{ci}_{cls_name}_integrated_gradients.png")

        # 4. Per-class panel
        plot_per_class_panel(sig_np, all_cams, true_lab, pred_probs, ci,
                             XAI_DIR / f"case{ci}_all_classes_panel.png")

        # 5. Irregularity detection (per class)
        print(f"  Running irregularity detection…")
        irregularities_per_class = {}
        for c_idx, sc in enumerate(SUPERCLASSES):
            prob = float(pred_probs[c_idx])
            if prob < 0.15:
                irregularities_per_class[sc] = []
                continue
            sal_c = compute_saliency(model, sig_t.clone(), c_idx)
            regs  = detect_irregularities(sal_c, prob)
            irregularities_per_class[sc] = regs
            if regs:
                print(f"    [{sc}] {len(regs)} region(s) — "
                      f"severities: {[r['severity_band'] for r in regs]}")

        # 6. Per-lead plot with irregularity bands
        plot_per_lead_with_irregularities(
            sig_np, saliency,
            irregularities_per_class.get(cls_name, []),
            cls_name, ci,
            XAI_DIR / f"case{ci}_{cls_name}_per_lead_irregularities.png"
        )

        # 7. Clinical text report
        generate_clinical_report(
            record_id                = record_id,
            case_idx                 = ci,
            true_labels              = true_lab,
            pred_probs               = pred_probs,
            irregularities_per_class = irregularities_per_class,
            saliency                 = saliency,
            save_path                = REPORT_DIR / f"case{ci}_{cls_name}_report.txt",
        )

        # 8. Attention map
        if model_name == "cnn_bilstm" and isinstance(model, CNNBiLSTM):
            model.eval()
            with torch.no_grad():
                _ = model(sig_t)
            attn = model.get_attention_weights()
            if attn is not None:
                plot_attention_map(attn.squeeze(0).cpu().numpy(), cls_name, ci,
                                   XAI_DIR / f"case{ci}_{cls_name}_attention.png",
                                   r_peaks_compressed=r_peaks_comp,
                                   beat_times=beat_times)

    gradcam.remove_hooks()
    print(f"\n[INFO] XAI complete.")
    print(f"[INFO] Plots   → {XAI_DIR}/")
    print(f"[INFO] Reports → {REPORT_DIR}/")
    for f in sorted(XAI_DIR.glob("*.png")):
        print(f"  {f.name}")
    for f in sorted(REPORT_DIR.glob("*.txt")):
        print(f"  reports/{f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="cnn_bilstm",
                        choices=["baseline_cnn", "cnn_bilstm"])
    parser.add_argument("--n_cases", type=int, default=3)
    args = parser.parse_args()
    run_xai(args.model, args.n_cases)
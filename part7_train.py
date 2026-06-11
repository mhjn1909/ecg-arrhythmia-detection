"""
Part 7 – Training Loop with Scheduler and Early Stopping
=========================================================
Depends on: part1–part6

Changes from original:
    - FocalLoss replaces BCEWithLogitsLoss (fixes HYP class)
    - HYP class weight manually boosted (extra penalty for missing it)
    - ECG augmentation added: Gaussian noise, random lead masking, time shift
    - Weight decay increased from 1e-4 to 3e-4
    - Augmentation applied only during training, never val/test

Run standalone:
    python part7_train.py --model cnn_bilstm --epochs 50 --batch_size 32
"""

import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from sklearn.metrics import f1_score
from pathlib import Path
from typing import Dict, Tuple, Optional

from part1_load_preprocess import load_metadata, load_all_waveforms, PTB_XL_PATH
from part2_label_mapping   import load_labels, compute_class_weights, SUPERCLASSES
from part3_data_split      import (align_signals_labels, patient_wise_split,
                                   load_splits, get_split_data, SIGNALS_CACHE,
                                   LABELS_CSV, SPLIT_CACHE)
from part4_dataset         import get_dataloaders
from part5_baseline_cnn    import build_baseline_cnn
from part6_cnn_bilstm      import build_cnn_bilstm

CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR        = Path("./logs")

# ── Index of HYP in SUPERCLASSES list (NORM=0, MI=1, STTC=2, CD=3, HYP=4)
HYP_IDX = 4
# ── Extra multiplier on HYP pos_weight to further penalise missed HYP cases
HYP_BOOST = 2.0

DEFAULT_CONFIG = {
    "model"        : "cnn_bilstm",
    "epochs"       : 50,
    "batch_size"   : 32,
    "lr"           : 3e-4,
    "weight_decay" : 3e-4,   # increased from 1e-4 → stronger L2 regularisation
    "grad_clip"    : 1.0,
    "patience"     : 10,
    "num_workers"  : 0,
    "seed"         : 42,
    # Focal loss params
    "focal_gamma"  : 2.0,    # higher = more focus on hard/rare examples
    "focal_alpha"  : 0.25,   # down-weights easy negatives
}


# ══════════════════════════════════════════════════════════════════
# 1. FOCAL LOSS
# ══════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label classification.
    FL(p) = -alpha * (1-p)^gamma * log(p)

    Compared to BCEWithLogitsLoss:
        - gamma > 0 down-weights easy examples, focuses on hard ones
        - This naturally helps rare/ambiguous classes like HYP
        - pos_weight still applied to further boost minority classes
    """
    def __init__(
        self,
        pos_weight : torch.Tensor,
        gamma      : float = 2.0,
        alpha      : float = 0.25,
        reduction  : str   = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE with pos_weight for class imbalance
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight = self.pos_weight,
            reduction  = "none",
        )
        probs    = torch.sigmoid(logits)
        # p_t: probability of the TRUE class
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t: alpha for positives, (1-alpha) for negatives
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w  = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_w * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ══════════════════════════════════════════════════════════════════
# 2. ECG AUGMENTATION
# ══════════════════════════════════════════════════════════════════
class ECGAugmentation:
    """
    Augmentations applied only during training on raw signal tensors.
    Input shape: (batch, leads=12, time_steps)

    Three augmentations (each applied independently with a probability):
        1. Gaussian noise       – simulates electrode noise
        2. Random lead masking  – forces model to not rely on single lead
        3. Random time shift    – handles different recording start points
    """
    def __init__(
        self,
        noise_std      : float = 0.02,   # std of Gaussian noise relative to signal
        noise_prob     : float = 0.5,    # probability of applying noise
        lead_mask_prob : float = 0.3,    # probability of masking a random lead
        time_shift_max : int   = 50,     # max samples to shift (at 500Hz = 0.1s)
        time_shift_prob: float = 0.4,    # probability of applying time shift
    ) -> None:
        self.noise_std       = noise_std
        self.noise_prob      = noise_prob
        self.lead_mask_prob  = lead_mask_prob
        self.time_shift_max  = time_shift_max
        self.time_shift_prob = time_shift_prob

    def __call__(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signals: Tensor of shape (batch, 12, T)
        Returns:
            Augmented tensor of same shape
        """
        signals = signals.clone()

        # 1. Gaussian noise
        if torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(signals) * self.noise_std
            signals = signals + noise

        # 2. Random lead masking — zero out 1 random lead per sample in batch
        if torch.rand(1).item() < self.lead_mask_prob:
            batch_size, n_leads, _ = signals.shape
            for i in range(batch_size):
                lead_to_mask = torch.randint(0, n_leads, (1,)).item()
                signals[i, lead_to_mask, :] = 0.0

        # 3. Random time shift — circular shift along time axis
        if torch.rand(1).item() < self.time_shift_prob:
            shift = torch.randint(-self.time_shift_max,
                                   self.time_shift_max + 1, (1,)).item()
            signals = torch.roll(signals, shifts=shift, dims=2)

        return signals


# ══════════════════════════════════════════════════════════════════
# 3. UTILITIES (unchanged from original)
# ══════════════════════════════════════════════════════════════════
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "max") -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = -np.inf if mode == "max" else np.inf
        self.counter   = 0
        self.stop      = False

    def __call__(self, metric: float) -> bool:
        improved = (
            (self.mode == "max" and metric > self.best + self.min_delta) or
            (self.mode == "min" and metric < self.best - self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def compute_macro_f1(
    all_logits: np.ndarray,
    all_labels: np.ndarray,
    threshold : float = 0.5,
) -> Tuple[float, np.ndarray]:
    probs     = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
    preds     = (probs >= threshold).astype(int)
    class_f1s = f1_score(all_labels, preds, average=None, zero_division=0)
    macro_f1  = class_f1s.mean()
    return macro_f1, class_f1s


# ══════════════════════════════════════════════════════════════════
# 4. EPOCH RUNNER — augmentation injected here for train only
# ══════════════════════════════════════════════════════════════════
def run_epoch(
    model      : nn.Module,
    loader     : torch.utils.data.DataLoader,
    criterion  : nn.Module,
    optimiser  : Optional[torch.optim.Optimizer],
    scaler     : Optional[GradScaler],
    device     : torch.device,
    scheduler  : Optional[object] = None,
    grad_clip  : float = 1.0,
    is_train   : bool  = True,
    augment    : Optional[ECGAugmentation] = None,  # NEW
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(is_train)
    total_loss = 0.0
    all_logits = []
    all_labels = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            # ── Apply augmentation only during training ──────────
            if is_train and augment is not None:
                signals = augment(signals)
            # ─────────────────────────────────────────────────────

            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                logits = model(signals)
                loss   = criterion(logits, labels)

            if is_train:
                optimiser.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimiser.step()

                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu().float().numpy())
            all_labels.append(labels.detach().cpu().float().numpy())

    avg_loss   = total_loss / len(loader)
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return avg_loss, all_logits, all_labels


# ══════════════════════════════════════════════════════════════════
# 5. CSV LOGGER (unchanged)
# ══════════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path   = path
        self.file   = open(path, "w", newline="")
        self.writer = None

    def log(self, row: dict) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


# ══════════════════════════════════════════════════════════════════
# 6. MAIN TRAIN FUNCTION
# ══════════════════════════════════════════════════════════════════
def train(cfg: dict) -> nn.Module:
    set_seed(cfg["seed"])
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[INFO] Device     : {device}  |  AMP: {use_amp}")
    print(f"[INFO] Model      : {cfg['model']}")
    print(f"[INFO] Epochs     : {cfg['epochs']}  |  Batch: {cfg['batch_size']}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data…")
    df_full      = load_metadata(PTB_XL_PATH)
    label_df     = load_labels(LABELS_CSV)
    signals_full = load_all_waveforms(df_full, PTB_XL_PATH)
    signals, label_df = align_signals_labels(signals_full, label_df, df_full.index)

    try:
        train_idx, val_idx, test_idx = load_splits(SPLIT_CACHE)
    except FileNotFoundError:
        train_idx, val_idx, test_idx = patient_wise_split(label_df)

    data    = get_split_data(signals, label_df, train_idx, val_idx, test_idx)
    loaders = get_dataloaders(data, batch_size=cfg["batch_size"],
                              num_workers=0, pin_memory=False)

    print(f"[INFO] Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # ── Class weights with HYP boost ─────────────────────────────
    train_label_df          = label_df.loc[train_idx]
    pos_weights             = compute_class_weights(train_label_df)
    pos_weights[HYP_IDX]   *= HYP_BOOST   # boost HYP penalty
    pw_tensor               = torch.from_numpy(pos_weights).to(device)
    print(f"[INFO] pos_weights (after HYP boost): "
          f"{dict(zip(SUPERCLASSES, pos_weights.round(2)))}")
    # ─────────────────────────────────────────────────────────────

    if cfg["model"] == "baseline_cnn":
        model = build_baseline_cnn(device)
    else:
        model = build_cnn_bilstm(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters : {n_params:,}")

    # ── Focal Loss (replaces BCEWithLogitsLoss) ───────────────────
    criterion = FocalLoss(
        pos_weight = pw_tensor,
        gamma      = cfg["focal_gamma"],
        alpha      = cfg["focal_alpha"],
    )
    print(f"[INFO] Loss       : FocalLoss (gamma={cfg['focal_gamma']}, "
          f"alpha={cfg['focal_alpha']})")
    # ─────────────────────────────────────────────────────────────

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],   # now 3e-4
    )

    steps_per_epoch = len(loaders["train"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr           = cfg["lr"],
        steps_per_epoch  = steps_per_epoch,
        epochs           = cfg["epochs"],
        pct_start        = 0.1,
        anneal_strategy  = "cos",
        div_factor       = 25,
        final_div_factor = 1e4,
    )

    # ── Augmentation (train only) ─────────────────────────────────
    augment = ECGAugmentation(
        noise_std       = 0.02,
        noise_prob      = 0.5,
        lead_mask_prob  = 0.3,
        time_shift_max  = 50,
        time_shift_prob = 0.4,
    )
    print("[INFO] Augmentation: Gaussian noise + lead masking + time shift")
    # ─────────────────────────────────────────────────────────────

    scaler     = GradScaler('cuda') if use_amp else None
    early_stop = EarlyStopping(patience=cfg["patience"], mode="max")
    logger     = CSVLogger(LOG_DIR / f"{cfg['model']}_training_log.csv")

    best_val_f1 = -np.inf
    best_ckpt   = CHECKPOINT_DIR / f"{cfg['model']}_best.pt"
    last_ckpt   = CHECKPOINT_DIR / f"{cfg['model']}_last.pt"

    start_epoch = 1
    if last_ckpt.exists():
        print(f"[INFO] Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("val_f1", -np.inf)
        print(f"[INFO] Resumed at epoch {start_epoch} | "
              f"Best Val F1 so far: {best_val_f1:.4f}")

    print("\n── Training ─────────────────────────────────────────────────")
    header = (f"{'Ep':>4} | {'TrainLoss':>10} {'TrainF1':>8} | "
              f"{'ValLoss':>9} {'ValF1':>7} | {'LR':>10} | {'Time':>6}")
    print(header)
    print("─" * len(header))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_logits, train_labels = run_epoch(
            model, loaders["train"], criterion, optimiser,
            scaler, device, scheduler, cfg["grad_clip"],
            is_train=True, augment=augment,   # augment passed here
        )
        train_f1, _ = compute_macro_f1(train_logits, train_labels)

        val_loss, val_logits, val_labels = run_epoch(
            model, loaders["val"], criterion, None,
            None, device, is_train=False, augment=None,   # NO augment on val
        )
        val_f1, val_class_f1s = compute_macro_f1(val_logits, val_labels)

        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        # Print per-class F1 for HYP visibility
        hyp_f1 = val_class_f1s[HYP_IDX]
        print(f"{epoch:>4} | {train_loss:>10.4f} {train_f1:>8.4f} | "
              f"{val_loss:>9.4f} {val_f1:>7.4f} | "
              f"{current_lr:>10.2e} | {elapsed:>5.1f}s"
              f" | HYP_F1={hyp_f1:.3f}"
              + (" ◀ best" if val_f1 > best_val_f1 else ""))

        log_row = {
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "train_f1"  : round(train_f1,   6),
            "val_loss"  : round(val_loss,    6),
            "val_f1"    : round(val_f1,      6),
            "lr"        : round(current_lr,  8),
            "time_s"    : round(elapsed,     2),
        }
        for sc, f1 in zip(SUPERCLASSES, val_class_f1s):
            log_row[f"val_f1_{sc}"] = round(float(f1), 6)
        logger.log(log_row)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch"        : epoch,
                "model_state"  : model.state_dict(),
                "optim_state"  : optimiser.state_dict(),
                "val_f1"       : val_f1,
                "val_class_f1s": val_class_f1s,
                "config"       : cfg,
            }, best_ckpt)

        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "optim_state": optimiser.state_dict(),
            "sched_state": scheduler.state_dict(),
            "val_f1"     : val_f1,
            "config"     : cfg,
        }, last_ckpt)

        if early_stop(val_f1):
            print(f"\n[INFO] Early stopping at epoch {epoch} "
                  f"(best val F1 = {best_val_f1:.4f})")
            break

    logger.close()
    print(f"\n[INFO] Training complete. Best Val Macro F1 : {best_val_f1:.4f}")
    print(f"[INFO] Best checkpoint → {best_ckpt}")
    print(f"[INFO] Training log   → {LOG_DIR}/{cfg['model']}_training_log.csv")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_best_model(model_name: str, device: torch.device) -> nn.Module:
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_name == "baseline_cnn":
        model = build_baseline_cnn(device)
    else:
        model = build_cnn_bilstm(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[INFO] Loaded {model_name} from epoch {ckpt['epoch']} "
          f"(Val F1 = {ckpt['val_f1']:.4f})")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG classifier")
    parser.add_argument("--model",        default=DEFAULT_CONFIG["model"],
                        choices=["baseline_cnn", "cnn_bilstm"])
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--patience",     type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--grad_clip",    type=float, default=DEFAULT_CONFIG["grad_clip"])
    parser.add_argument("--num_workers",  type=int,   default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--seed",         type=int,   default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--focal_gamma",  type=float, default=DEFAULT_CONFIG["focal_gamma"])
    parser.add_argument("--focal_alpha",  type=float, default=DEFAULT_CONFIG["focal_alpha"])
    args = parser.parse_args()

    cfg = vars(args)
    train(cfg)
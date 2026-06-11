"""
Part 4 – Custom PyTorch Dataset and DataLoader
===============================================
Depends on: part1, part2, part3

Provides:
    - ECGDataset       : PyTorch Dataset with optional augmentation
    - get_dataloaders  : returns train/val/test DataLoader objects

Augmentations (training only):
    - Gaussian noise injection
    - Random amplitude scaling
    - Random time shift
    - Lead dropout (randomly zero out 1-2 leads)

Signal tensor shape fed to model: (12, 5000)  — (leads, time)

Run standalone:
    python part4_dataset.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional

from part2_label_mapping import SUPERCLASSES
from part3_data_split import (
    load_splits,
    load_labels,
    get_split_data,
    SIGNALS_CACHE,
    LABELS_CSV,
    SPLIT_CACHE,
)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_WORKERS  = 0      # Windows fix — multiprocessing causes pickle errors on Windows
PIN_MEMORY   = False  # PIN_MEMORY must be False when NUM_WORKERS=0 on Windows

# Augmentation hyperparameters (training only)
AUG_NOISE_STD     = 0.02   # σ for Gaussian noise (signal is z-scored, so small)
AUG_SCALE_RANGE   = (0.85, 1.15)  # random amplitude scale factor
AUG_SHIFT_MAX     = 50     # max time-shift in samples (±50 @ 500Hz = ±100ms)
AUG_LEAD_DROP_P   = 0.15   # probability of dropping each lead
AUG_LEAD_DROP_MAX = 2      # maximum number of leads to drop


# ──────────────────────────────────────────────
# AUGMENTATION FUNCTIONS
# ──────────────────────────────────────────────
def augment_signal(signal: np.ndarray) -> np.ndarray:
    """
    Apply randomised augmentations to a single ECG signal.

    Args:
        signal : np.ndarray, shape (5000, 12)  — time × leads

    Returns:
        np.ndarray, same shape, augmented
    """
    signal = signal.copy()

    # 1. Gaussian noise
    signal += np.random.normal(0, AUG_NOISE_STD, signal.shape).astype(np.float32)

    # 2. Random amplitude scale (applied uniformly across all leads)
    scale = np.random.uniform(*AUG_SCALE_RANGE)
    signal *= scale

    # 3. Random time shift — roll along time axis, fill boundary with zeros
    shift = np.random.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1)
    if shift != 0:
        signal = np.roll(signal, shift, axis=0)
        # Zero out wrapped boundary instead of allowing circular leakage
        if shift > 0:
            signal[:shift, :] = 0.0
        else:
            signal[shift:, :] = 0.0

    # 4. Lead dropout — randomly zero out up to AUG_LEAD_DROP_MAX leads
    n_drop = np.random.randint(0, AUG_LEAD_DROP_MAX + 1)
    if n_drop > 0:
        drop_leads = np.random.choice(signal.shape[1], n_drop, replace=False)
        for lead_idx in drop_leads:
            if np.random.rand() < AUG_LEAD_DROP_P:
                signal[:, lead_idx] = 0.0

    return signal


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class ECGDataset(Dataset):
    """
    PyTorch Dataset for 12-lead ECG multi-label classification.

    Args:
        signals   : np.ndarray, shape (N, 5000, 12)
        labels    : np.ndarray, shape (N, 5),  float32 binary
        augment   : if True, apply training-time augmentations
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        assert signals.shape[0] == labels.shape[0], \
            f"Signal/label count mismatch: {signals.shape[0]} vs {labels.shape[0]}"
        assert signals.shape[1:] == (5000, 12), \
            f"Unexpected signal shape: {signals.shape}"
        assert labels.shape[1] == len(SUPERCLASSES), \
            f"Expected {len(SUPERCLASSES)} label columns, got {labels.shape[1]}"

        self.signals = signals    # stored as numpy; converted to tensor in __getitem__
        self.labels  = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]   # (5000, 12)
        label  = self.labels[idx]    # (5,)

        # Optional augmentation (training only)
        if self.augment:
            signal = augment_signal(signal)

        # Transpose to (leads, time) = (12, 5000) for Conv1d
        signal_tensor = torch.from_numpy(signal.T.copy())   # (12, 5000)
        label_tensor  = torch.from_numpy(label)             # (5,)

        return signal_tensor, label_tensor


# ──────────────────────────────────────────────
# DATALOADERS
# ──────────────────────────────────────────────
def get_dataloaders(
    data: Dict[str, Dict],
    batch_size: int  = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from the split data dict
    returned by Part 3's get_split_data().

    Training loader:   shuffled + augmented
    Val / Test loaders: deterministic, no augmentation

    Args:
        data        : {'train': {'signals':…,'labels':…}, 'val':…, 'test':…}
        batch_size  : samples per batch
        num_workers : parallel data loading workers (0 = main process only, safe on Windows)
        pin_memory  : pin to page-locked memory (set False when num_workers=0)

    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    datasets = {
        "train": ECGDataset(data["train"]["signals"], data["train"]["labels"], augment=True),
        "val"  : ECGDataset(data["val"]["signals"],   data["val"]["labels"],   augment=False),
        "test" : ECGDataset(data["test"]["signals"],  data["test"]["labels"],  augment=False),
    }

    loaders = {}
    for split, dataset in datasets.items():
        is_train = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size         = batch_size,
            shuffle            = is_train,       # shuffle only training data
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            drop_last          = is_train,       # avoid single-sample batches at epoch end
            persistent_workers = (num_workers > 0),  # False when num_workers=0
        )

    return loaders


# ──────────────────────────────────────────────
# DATALOADER STATISTICS
# ──────────────────────────────────────────────
def print_dataloader_info(loaders: Dict[str, DataLoader]) -> None:
    """Print shapes and counts for verification."""
    print("\n── DataLoader Summary ────────────────────────────────────")
    for split, loader in loaders.items():
        ds = loader.dataset
        n  = len(ds)
        nb = len(loader)
        print(f"  {split:<6} │ records: {n:>6} │ batches: {nb:>5} │ "
              f"augment: {ds.augment}")

    # Show one batch to verify tensor shapes
    signals, labels = next(iter(loaders["train"]))
    print(f"\n  Sample batch:")
    print(f"    signals : {tuple(signals.shape)}  dtype={signals.dtype}")
    print(f"    labels  : {tuple(labels.shape)}   dtype={labels.dtype}")
    print(f"    signal range : [{signals.min():.3f}, {signals.max():.3f}]")
    print("──────────────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from part1_load_preprocess import load_metadata, load_all_waveforms, PTB_XL_PATH
    from part3_data_split import align_signals_labels, patient_wise_split

    # 1. Load all necessary data
    df_full      = load_metadata(PTB_XL_PATH)
    label_df     = load_labels(LABELS_CSV)
    signals_full = load_all_waveforms(df_full, PTB_XL_PATH)

    # 2. Align signals to labelled subset
    signals, label_df = align_signals_labels(signals_full, label_df, df_full.index)

    # 3. Load or compute splits
    try:
        train_idx, val_idx, test_idx = load_splits(SPLIT_CACHE)
        print("[INFO] Loaded split indices from cache.")
    except FileNotFoundError:
        print("[INFO] Computing splits...")
        train_idx, val_idx, test_idx = patient_wise_split(label_df)

    # 4. Build split data dict
    data = get_split_data(signals, label_df, train_idx, val_idx, test_idx)

    # 5. Build DataLoaders
    loaders = get_dataloaders(data, batch_size=BATCH_SIZE)

    # 6. Print info and verify shapes
    print_dataloader_info(loaders)

    # 7. Verify augmentation produces different tensors for same index
    ds = loaders["train"].dataset
    s1, _ = ds[0]
    s2, _ = ds[0]
    print(f"[INFO] Augmentation check — same index, different values: "
          f"{not torch.allclose(s1, s2)}")   # should be True
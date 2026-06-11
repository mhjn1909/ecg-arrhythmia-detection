"""
Part 3 – Patient-Wise Train / Validation / Test Split
======================================================
Depends on: part2_label_mapping.py (load_labels, SUPERCLASSES)

PTB-XL provides a stratified fold column (strat_fold 1–10).
We follow the official recommended split:
    - Folds 1–8  → Train
    - Fold  9    → Validation
    - Fold  10   → Test
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from part2_label_mapping import (
    load_labels,
    SUPERCLASSES,
    compute_class_weights,
)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
TRAIN_FOLDS = list(range(1, 9))   # folds 1–8
VAL_FOLD    = 9
TEST_FOLD   = 10

SIGNALS_CACHE = "./data_cache/signals.npy"
LABELS_CSV    = "./data_cache/labels.csv"
SPLIT_CACHE   = "./data_cache"


# ──────────────────────────────────────────────
# ALIGN SIGNALS ARRAY WITH LABEL DATAFRAME
# ──────────────────────────────────────────────
def align_signals_to_labels(
    signals: np.ndarray,
    full_label_df: pd.DataFrame,
    raw_meta_index: pd.Index,
) -> np.ndarray:
    pos_lookup = {ecg_id: i for i, ecg_id in enumerate(raw_meta_index)}
    indices = [pos_lookup[ecg_id] for ecg_id in full_label_df.index]
    return signals[indices]


def align_signals_labels(signals, full_label_df, raw_meta_index):
    """Wrapper used by run_all.py — returns (aligned_signals, label_df)."""
    aligned = align_signals_to_labels(signals, full_label_df, raw_meta_index)
    return aligned, full_label_df


# ──────────────────────────────────────────────
# CORE SPLIT FUNCTION
# ──────────────────────────────────────────────
def split_by_fold(
    label_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = label_df[label_df["strat_fold"].isin(TRAIN_FOLDS)].copy()
    val_df   = label_df[label_df["strat_fold"] == VAL_FOLD].copy()
    test_df  = label_df[label_df["strat_fold"] == TEST_FOLD].copy()
    return train_df, val_df, test_df


def patient_wise_split(
    label_df: pd.DataFrame,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Split using PTB-XL official strat_fold column.
    Returns (train_idx, val_idx, test_idx) as pd.Index of ecg_ids.
    """
    train_df, val_df, test_df = split_by_fold(label_df)
    verify_no_leakage(train_df, val_df, test_df)
    return train_df.index, val_df.index, test_df.index


# ──────────────────────────────────────────────
# SAVE / LOAD SPLIT INDICES
# ──────────────────────────────────────────────
def save_splits(
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
    out_dir: str = SPLIT_CACHE,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.Index(train_idx).to_frame(index=False, name="ecg_id").to_csv(out / "train_idx.csv", index=False)
    pd.Index(val_idx).to_frame(index=False,   name="ecg_id").to_csv(out / "val_idx.csv",   index=False)
    pd.Index(test_idx).to_frame(index=False,  name="ecg_id").to_csv(out / "test_idx.csv",  index=False)
    print(f"[INFO] Saved split indices → {out}/{{train,val,test}}_idx.csv")


def load_splits(
    cache_dir: str = SPLIT_CACHE,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Load pre-computed split indices.
    Raises FileNotFoundError if cache doesn't exist (triggers re-split in run_all).
    """
    d = Path(cache_dir)
    train_idx = pd.read_csv(d / "train_idx.csv")["ecg_id"].values
    val_idx   = pd.read_csv(d / "val_idx.csv")["ecg_id"].values
    test_idx  = pd.read_csv(d / "test_idx.csv")["ecg_id"].values
    return pd.Index(train_idx), pd.Index(val_idx), pd.Index(test_idx)


# ──────────────────────────────────────────────
# GET SPLIT DATA ARRAYS
# ──────────────────────────────────────────────
def get_split_data(
    signals: np.ndarray,
    label_df: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
) -> Dict[str, dict]:
    pos = {ecg_id: i for i, ecg_id in enumerate(label_df.index)}

    def _slice(idx):
        positions = [pos[e] for e in idx if e in pos]
        valid_idx = [e for e in idx if e in pos]
        X = signals[positions]
        y = label_df.loc[valid_idx, SUPERCLASSES].values.astype(np.float32)
        return X, y

    X_train, y_train = _slice(train_idx)
    X_val,   y_val   = _slice(val_idx)
    X_test,  y_test  = _slice(test_idx)

    return {
        "train": {"signals": X_train, "labels": y_train},
        "val":   {"signals": X_val,   "labels": y_val},
        "test":  {"signals": X_test,  "labels": y_test},
    }


# ──────────────────────────────────────────────
# VALIDATION: CONFIRM NO PATIENT LEAKAGE
# ──────────────────────────────────────────────
def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_pats = set(train_df["patient_id"])
    val_pats   = set(val_df["patient_id"])
    test_pats  = set(test_df["patient_id"])

    assert len(train_pats & val_pats)  == 0, f"Train/Val leakage!"
    assert len(train_pats & test_pats) == 0, f"Train/Test leakage!"
    assert len(val_pats   & test_pats) == 0, f"Val/Test leakage!"

    print("[INFO] ✓ No patient leakage detected across splits.")


# ──────────────────────────────────────────────
# SPLIT STATISTICS
# ──────────────────────────────────────────────
def print_split_statistics(
    label_df: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
) -> None:
    splits = {
        "Train": label_df.loc[label_df.index.isin(train_idx)],
        "Val":   label_df.loc[label_df.index.isin(val_idx)],
        "Test":  label_df.loc[label_df.index.isin(test_idx)],
    }

    print("\n── Split Statistics ──────────────────────────────────────")
    header = f"  {'Split':<7} {'Records':>8} {'Patients':>10}"
    for sc in SUPERCLASSES:
        header += f"  {sc:>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for name, df in splits.items():
        n_rec = len(df)
        n_pat = df["patient_id"].nunique()
        row   = f"  {name:<7} {n_rec:>8} {n_pat:>10}"
        for sc in SUPERCLASSES:
            prev = df[sc].sum() / n_rec * 100
            row += f"  {prev:>6.1f}%"
        print(row)

    print("──────────────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    label_df = load_labels(LABELS_CSV)
    print(f"[INFO] Total labelled records: {len(label_df)}")

    train_idx, val_idx, test_idx = patient_wise_split(label_df)

    print_split_statistics(label_df, train_idx, val_idx, test_idx)

    print("[INFO] Class weights (computed on TRAIN split only):")
    compute_class_weights(label_df.loc[train_idx])

    save_splits(train_idx, val_idx, test_idx)

    print(f"\n[INFO] Fold → Split mapping:")
    print(f"  Folds {TRAIN_FOLDS} → Train ({len(train_idx):,} records)")
    print(f"  Fold  {VAL_FOLD}              → Validation ({len(val_idx):,} records)")
    print(f"  Fold  {TEST_FOLD}             → Test ({len(test_idx):,} records)")
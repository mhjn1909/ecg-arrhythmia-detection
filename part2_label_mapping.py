"""
Part 2 – Map SCP Codes to Diagnostic Superclasses
===================================================
Depends on: part1_load_preprocess.py (load_metadata)

Reads the SCP statements reference file included in PTB-XL,
maps each record's scp_codes to one or more of the 5 superclasses,
and builds a binary multi-label matrix ready for training.

Superclasses:
    NORM  – Normal ECG
    MI    – Myocardial Infarction
    STTC  – ST/T-wave Change
    CD    – Conduction Disturbance
    HYP   – Hypertrophy

Run standalone:
    python part2_label_mapping.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── re-use Part 1 loader ──────────────────────────────────────────────────────
from part1_load_preprocess import load_metadata, PTB_XL_PATH

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]   # fixed label order

# Minimum likelihood threshold: PTB-XL annotators assign a confidence %
# to each SCP code. We keep codes with likelihood >= this value.
# 0 → keep all; 100 → only definitive annotations.
MIN_LIKELIHOOD = 0   # keep all (standard practice for PTB-XL benchmarks)


# ──────────────────────────────────────────────
# LOAD SCP REFERENCE TABLE
# ──────────────────────────────────────────────
def load_scp_statements(ptb_xl_path: str) -> pd.DataFrame:
    """
    Load scp_statements.csv which maps every SCP code to a diagnostic_class.

    Relevant column: 'diagnostic_class' ∈ {NORM, MI, STTC, CD, HYP, nan}
    Codes without a diagnostic_class are non-diagnostic (rhythm, form) → ignored.

    Returns:
        pd.DataFrame indexed by SCP code string.
    """
    path = os.path.join(ptb_xl_path, "scp_statements.csv")
    scp_df = pd.read_csv(path, index_col=0)

    # Keep only diagnostic codes (have a superclass assignment)
    diagnostic = scp_df[scp_df["diagnostic_class"].notna()].copy()
    print(f"[INFO] SCP statements total       : {len(scp_df)}")
    print(f"[INFO] Diagnostic SCP codes       : {len(diagnostic)}")
    print(f"[INFO] Superclasses found         : {sorted(diagnostic['diagnostic_class'].unique())}")
    return diagnostic


# ──────────────────────────────────────────────
# BUILD MULTI-LABEL MATRIX
# ──────────────────────────────────────────────
def build_label_matrix(
    df: pd.DataFrame,
    scp_statements: pd.DataFrame,
    min_likelihood: int = MIN_LIKELIHOOD,
) -> pd.DataFrame:
    """
    For every ECG record, produce a binary vector of length 5
    indicating which superclasses are present.

    Logic:
        - Iterate each record's scp_codes dict  {code: likelihood, …}
        - Filter by min_likelihood
        - Look up each code's diagnostic_class in scp_statements
        - Set corresponding superclass column to 1

    Records with NO diagnostic code surviving the filter get all-zero labels
    and are excluded from training (dropped) to avoid noisy supervision.

    Args:
        df              : metadata DataFrame from Part 1
        scp_statements  : diagnostic SCP reference from load_scp_statements()
        min_likelihood  : minimum annotator confidence (0–100)

    Returns:
        pd.DataFrame, index=ecg_id, columns=SUPERCLASSES + ['patient_id','strat_fold']
        Only records with at least one superclass label are retained.
    """
    # Map code → superclass for fast lookup
    code_to_superclass: dict = scp_statements["diagnostic_class"].to_dict()

    records = []
    for ecg_id, row in df.iterrows():
        label_vec = {sc: 0 for sc in SUPERCLASSES}
        has_any = False

        for code, likelihood in row["scp_codes"].items():
            # Skip if below confidence threshold
            if likelihood < min_likelihood:
                continue
            # Skip non-diagnostic codes
            superclass = code_to_superclass.get(code)
            if superclass is None:
                continue
            if superclass in label_vec:
                label_vec[superclass] = 1
                has_any = True

        if has_any:
            record = {
                "ecg_id"     : ecg_id,
                "patient_id" : row["patient_id"],
                "strat_fold" : row["strat_fold"],
                **label_vec,
            }
            records.append(record)

    label_df = pd.DataFrame(records).set_index("ecg_id")
    return label_df


# ──────────────────────────────────────────────
# STATISTICS & VALIDATION
# ──────────────────────────────────────────────
def print_label_statistics(label_df: pd.DataFrame) -> None:
    """Print class distribution, multi-label overlap, and co-occurrence."""
    print("\n── Label Statistics ──────────────────────────────────────")
    total = len(label_df)
    print(f"  Total labelled records : {total}")

    print(f"\n  {'Superclass':<8} {'Count':>7} {'Prevalence':>12}")
    print(f"  {'─'*8} {'─'*7} {'─'*12}")
    for sc in SUPERCLASSES:
        count = label_df[sc].sum()
        print(f"  {sc:<8} {count:>7}  {count/total*100:>10.1f}%")

    # Multi-label degree distribution
    label_df["n_labels"] = label_df[SUPERCLASSES].sum(axis=1)
    print(f"\n  Labels-per-record distribution:")
    for n, cnt in label_df["n_labels"].value_counts().sort_index().items():
        print(f"    {n} label(s): {cnt} records ({cnt/total*100:.1f}%)")
    label_df.drop(columns=["n_labels"], inplace=True)

    # Co-occurrence matrix
    print(f"\n  Co-occurrence matrix (% of total):")
    co = label_df[SUPERCLASSES].T.dot(label_df[SUPERCLASSES]) / total * 100
    print(co.round(1).to_string())
    print("──────────────────────────────────────────────────────────\n")


def compute_class_weights(label_df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.

    Formula (same as PyTorch's pos_weight recommendation):
        pos_weight[c] = (N - n_pos[c]) / n_pos[c]

    A higher weight penalises false negatives more for rare classes.

    Returns:
        np.ndarray, shape (5,), dtype float32
    """
    n_total = len(label_df)
    weights = []
    for sc in SUPERCLASSES:
        n_pos = label_df[sc].sum()
        n_neg = n_total - n_pos
        w = n_neg / (n_pos + 1e-8)
        weights.append(w)
    pos_weights = np.array(weights, dtype=np.float32)

    print("  Positive class weights (for BCEWithLogitsLoss pos_weight):")
    for sc, w in zip(SUPERCLASSES, pos_weights):
        print(f"    {sc:<6}: {w:.3f}")

    return pos_weights


# ──────────────────────────────────────────────
# SAVE LABELS
# ──────────────────────────────────────────────
def save_labels(label_df: pd.DataFrame, out_path: str = "./data_cache/labels.csv") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(out_path)
    print(f"[INFO] Labels saved → {out_path}")


def load_labels(path: str = "./data_cache/labels.csv") -> pd.DataFrame:
    """Reload labels produced by this module."""
    df = pd.read_csv(path, index_col="ecg_id")
    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load raw metadata
    df = load_metadata(PTB_XL_PATH)

    # 2. Load SCP reference
    scp_statements = load_scp_statements(PTB_XL_PATH)

    # 3. Build binary label matrix
    label_df = build_label_matrix(df, scp_statements, min_likelihood=MIN_LIKELIHOOD)
    print(f"[INFO] Records with valid labels  : {len(label_df)}")
    print(f"[INFO] Records dropped (no label) : {len(df) - len(label_df)}")

    # 4. Print statistics
    print_label_statistics(label_df)

    # 5. Compute class weights
    pos_weights = compute_class_weights(label_df)

    # 6. Save
    save_labels(label_df)

    # Preview
    print("\n[INFO] Label DataFrame head:")
    print(label_df[SUPERCLASSES + ["patient_id", "strat_fold"]].head(10).to_string())

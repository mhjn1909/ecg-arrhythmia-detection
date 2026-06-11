"""
Part 1 – Load PTB-XL Dataset and Preprocessing
================================================
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
from pathlib import Path
from typing import Tuple, Dict

# ──────────────────────────────────────────────
# CONFIGURATION (AUTO-RESOLVED)
# ──────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
PTB_XL_PATH   = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR     = os.path.join(PROJECT_ROOT, "data_cache")
CACHE_PATH    = os.path.join(CACHE_DIR, "signals.npy")

SAMPLING_RATE = 500
SIGNAL_LENGTH = 5000
NUM_LEADS     = 12
LOWCUT_HZ     = 0.5
HIGHCUT_HZ    = 150.0
NOTCH_HZ      = 50.0
NOTCH_Q       = 30.0


# ──────────────────────────────────────────────
# FILTER DESIGN
# ──────────────────────────────────────────────
def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band")


def _notch_filter_coeffs(notch_hz: float, q: float, fs: float):
    return iirnotch(notch_hz / (0.5 * fs), q)


def preprocess_signal(signal: np.ndarray, fs: float = SAMPLING_RATE) -> np.ndarray:
    b_bp, a_bp = _butter_bandpass(LOWCUT_HZ, HIGHCUT_HZ, fs)
    signal = filtfilt(b_bp, a_bp, signal, axis=0)

    b_n, a_n = _notch_filter_coeffs(NOTCH_HZ, NOTCH_Q, fs)
    signal = filtfilt(b_n, a_n, signal, axis=0)

    signal = np.clip(signal, -5.0, 5.0)

    mean   = signal.mean(axis=0, keepdims=True)
    std    = signal.std(axis=0,  keepdims=True) + 1e-8
    signal = (signal - mean) / std

    return signal.astype(np.float32)


# ──────────────────────────────────────────────
# LOAD METADATA
# ──────────────────────────────────────────────
def load_metadata(ptb_xl_path: str = PTB_XL_PATH) -> pd.DataFrame:
    csv_path = os.path.join(ptb_xl_path, "ptbxl_database.csv")
    print("[DEBUG] Looking for CSV at:", csv_path)
    assert os.path.exists(csv_path), f"File not found: {csv_path}"

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    print(f"[INFO] Loaded metadata: {len(df)} records")
    print(f"[INFO] Unique patients  : {df['patient_id'].nunique()}")
    print(f"[INFO] Strat folds      : {sorted(df['strat_fold'].unique())}")
    return df


# ──────────────────────────────────────────────
# LOAD SINGLE WAVEFORM
# ──────────────────────────────────────────────
def load_waveform(record_path: str, ptb_xl_path: str = PTB_XL_PATH) -> np.ndarray:
    full_path = os.path.join(ptb_xl_path, record_path)
    record    = wfdb.rdrecord(full_path)
    signal    = record.p_signal

    if signal.shape[0] < SIGNAL_LENGTH:
        pad    = np.zeros((SIGNAL_LENGTH - signal.shape[0], NUM_LEADS), dtype=np.float32)
        signal = np.vstack([signal, pad])
    else:
        signal = signal[:SIGNAL_LENGTH, :]

    return preprocess_signal(signal)


# ──────────────────────────────────────────────
# LOAD ALL WAVEFORMS
# ──────────────────────────────────────────────
def load_all_waveforms(
    df           : pd.DataFrame,
    ptb_xl_path  : str  = PTB_XL_PATH,
    cache_path   : str  = CACHE_PATH,
    force_reload : bool = False,
) -> np.ndarray:

    cache = Path(cache_path)

    if cache.exists() and not force_reload:
        signals = np.load(cache)
        print(f"[INFO] Loaded cache: {signals.shape}")
        return signals

    # Create cache directory with explicit permissions
    os.makedirs(str(cache.parent), exist_ok=True)

    n       = len(df)
    signals = np.zeros((n, SIGNAL_LENGTH, NUM_LEADS), dtype=np.float32)

    print(f"[INFO] Loading {n} waveforms from disk...")
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            signals[i] = load_waveform(row["filename_hr"], ptb_xl_path)
        except Exception as e:
            print(f"[WARN] Failed index={i}: {e}")

        if (i + 1) % 2000 == 0:
            print(f"[INFO] {i + 1}/{n} loaded")

    np.save(str(cache), signals)
    print(f"[INFO] Cache saved → {cache}")
    return signals


# ──────────────────────────────────────────────
# SANITY CHECK
# ──────────────────────────────────────────────
def sanity_check(signals: np.ndarray, df: pd.DataFrame) -> None:
    print("\n── Sanity Check ─────────────────────────────")
    print(f"  signals shape : {signals.shape}")
    print(f"  dtype         : {signals.dtype}")
    print(f"  global mean   : {signals.mean():.4f}")
    print(f"  global std    : {signals.std():.4f}")
    print(f"  min / max     : {signals.min():.4f} / {signals.max():.4f}")
    print(f"  NaN count     : {np.isnan(signals).sum()}")
    print(f"  Inf count     : {np.isinf(signals).sum()}")
    print(f"  metadata rows : {len(df)}")
    print("─────────────────────────────────────────────\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    df      = load_metadata(PTB_XL_PATH)
    signals = load_all_waveforms(df, PTB_XL_PATH)
    sanity_check(signals, df)
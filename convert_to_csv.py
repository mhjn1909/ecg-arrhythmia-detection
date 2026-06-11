"""
convert_to_csv.py – Convert PTB-XL WFDB records → CSV for interface testing
============================================================================
Run from your project root:
    python convert_to_csv.py --n 10

Produces:  sample_ecgs/ecg_001.csv, ecg_002.csv, ...
Each CSV:  5000 rows × 12 columns (leads), no header, values in mV (preprocessed)
"""

import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from part1_load_preprocess import load_metadata, load_waveform, PTB_XL_PATH


def export_samples(n: int = 10, out_dir: str = "./sample_ecgs") -> None:
    Path(out_dir).mkdir(exist_ok=True)
    df = load_metadata(PTB_XL_PATH)

    # Take n samples from the test fold (fold 10)
    test_df = df[df["strat_fold"] == 10].head(n)

    for i, (ecg_id, row) in enumerate(test_df.iterrows(), 1):
        try:
            signal = load_waveform(row["filename_hr"], PTB_XL_PATH)   # (5000, 12)
            out_path = Path(out_dir) / f"ecg_{i:03d}_id{ecg_id}.csv"
            np.savetxt(str(out_path), signal, delimiter=",", fmt="%.6f")
            scp = dict(row["scp_codes"])
            print(f"[{i:02d}] ecg_id={ecg_id}  → {out_path.name}   scp={scp}")
        except Exception as e:
            print(f"[WARN] ecg_id={ecg_id} failed: {e}")

    print(f"\n✓ Exported {n} ECG CSVs to {out_dir}/")
    print("Upload any of these to the Streamlit interface.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=10,
                        help="Number of records to export")
    parser.add_argument("--out_dir", default="./sample_ecgs",
                        help="Output directory")
    args = parser.parse_args()
    export_samples(args.n, args.out_dir)

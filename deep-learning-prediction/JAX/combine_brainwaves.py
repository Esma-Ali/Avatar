#!/usr/bin/env python
"""
combine_brainwaves.py
Author: Esma Ali
Combine OpenBCI EEG CSVs (whitespace-separated) from labeled subfolders into
data/brainwaves.csv for training. Writes data/brainwaves.csv.labels.json too.
"""

from pathlib import Path
from typing import Optional, List
import argparse
import json
import pandas as pd

DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_OUT_CSV  = Path("data/brainwaves.csv")
DEFAULT_NUM_CHANNELS = 8  # Cyton=8; Daisy=16

def read_one_csv(file_path: Path, label: str, num_channels: int) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path, header=None, skiprows=1, sep=r"\s+")
        if df.shape[1] < num_channels:
            print(f"  ! skip {file_path.name}: only {df.shape[1]} cols < {num_channels}")
            return None
        df = df.iloc[:, :num_channels].copy()
        df["label"] = label
        df.columns = [f"_c{i}" for i in range(num_channels)] + ["label"]
        return df
    except Exception as e:
        print(f"  ! error reading {file_path}: {e}")
        return None

def collect_labels(raw_root: Path) -> List[Path]:
    if not raw_root.exists():
        raise SystemExit(f"Error: raw data folder not found: {raw_root}")
    label_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if not label_dirs:
        raise SystemExit(f"No labeled subfolders found under: {raw_root}")
    return label_dirs

def main():
    ap = argparse.ArgumentParser(description="Combine OpenBCI EEG CSVs into one dataset.")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--channels", type=int, default=DEFAULT_NUM_CHANNELS)
    args = ap.parse_args()

    raw_root: Path = args.raw
    out_csv: Path  = args.out
    num_channels: int = args.channels

    print(f"[combine] scanning: {raw_root}")
    label_dirs = collect_labels(raw_root)
    print(f"[combine] found class folders: {[p.name for p in label_dirs]}")

    all_rows: List[pd.DataFrame] = []
    total_files = 0
    used_files = 0

    for folder in label_dirs:
        label = folder.name
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            print(f"  (no CSV files in {folder})")
            continue
        print(f"[{label}] {len(csvs)} files")
        total_files += len(csvs)
        for fp in csvs:
            df = read_one_csv(fp, label, num_channels)
            if df is not None:
                all_rows.append(df)
                used_files += 1

    if not all_rows:
        raise SystemExit("No valid CSV rows found. Check file format or --channels value.")

    full = pd.concat(all_rows, ignore_index=True)
    print(f"[combine] raw combined shape: {full.shape}")

    # encode labels (no sklearn)
    full["label"] = full["label"].astype("category")
    classes = list(full["label"].cat.categories)
    mapping = {str(cls): int(i) for i, cls in enumerate(classes)}
    full["label"] = full["label"].cat.codes.astype("int32")

    # shuffle (deterministic)
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_csv, index=False)

    labels_json = out_csv.with_suffix(".labels.json")
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"[combine] wrote CSV   → {out_csv.resolve()}")
    print(f"[combine] wrote labels → {labels_json.resolve()}")
    print(f"[combine] files: used {used_files}/{total_files} CSVs; channels={num_channels}")
    print(f"[combine] classes: {mapping}")

if __name__ == "__main__":
    main()

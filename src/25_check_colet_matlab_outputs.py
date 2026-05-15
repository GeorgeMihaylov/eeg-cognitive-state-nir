#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check COLET MATLAB conversion outputs.

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\25_check_colet_matlab_outputs.py
"""

from pathlib import Path
import pandas as pd


ROOT = Path(".").resolve()
REPORT_DIR = ROOT / "reports" / "wearable_pm_alignment"

files = [
    REPORT_DIR / "colet_matlab_structure_inventory.csv",
    REPORT_DIR / "colet_matlab_annotation_probe.csv",
]

for path in files:
    print("=" * 80)
    print(path)
    if not path.exists():
        print("MISSING")
        continue

    df = pd.read_csv(path)
    print(f"rows={len(df)} columns={list(df.columns)}")
    print(df.head(30).to_string(index=False))

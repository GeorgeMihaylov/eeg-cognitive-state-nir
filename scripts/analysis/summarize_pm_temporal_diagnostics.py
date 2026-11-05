"""Assemble the existing compact diagnostic tables; never train a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/diagnostics/pm_temporal_predictability_audit_v1"))
    args = parser.parse_args()
    root = args.report_dir
    paths = {"history_support": "history_support_global.csv", "acf": "acf_summary.csv",
             "mutual_information": "mi_surrogate_summary.csv",
             "permutation_entropy": "pe_final_v1/permutation_entropy_summary.csv",
             "ssa": "ssa_summary.csv", "fnn": "fnn_summary.csv"}
    payload = {"schema_version": "temporal-diagnostics-summary-v1", "result_status": "diagnostic_and_exploratory",
               "classifier_fits_in_diagnostics": 0, "canonical_alignment": "EEG[T-10,T] -> PM[T,T+10]",
               "tables": {}, "source_sha256": {},
               "limitations": ["equal-record medians are descriptive, records within subjects are dependent",
                   "PE order-specific length gates change support: m5 only eight records",
                   "FNN fixed feature subset and long-record subset; no reliable embedding dimension established",
                   "SSA rank reduction relative to permutation does not imply deterministic dynamics",
                   "feature temporal dependence does not establish additional PM prediction information"],
               "decision": {"predictive_full_run_scientifically_motivated": True,
                            "predictive_full_run_authorized_or_executed": False,
                            "new_sequence_architecture_justified": False,
                            "next": "separately authorized matched H10-H120 LightGBM experiment"}}
    for name, relative in paths.items():
        path = root / relative
        frame = pd.read_csv(path)
        payload["tables"][name] = json.loads(frame.to_json(orient="records"))
        payload["source_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("history_support_summary.json", "pe_final_v1/permutation_entropy_protocol.json",
                 "ssa_protocol.json", "fnn_protocol.json", "state_space_execution.json"):
        payload.setdefault("protocols_and_status", {})[name] = json.loads((root / name).read_text(encoding="utf-8"))
        payload["source_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "temporal_diagnostics_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run target-free SSA and restricted FNN diagnostics; no classifiers are fitted."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis import temporal_state_space as state
from bench.analysis.temporal_feature_diagnostics import contiguous_segments
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low


def summarize(frame: pd.DataFrame, metrics: list[str], extra: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature medians within record/family, followed by equal-record medians."""
    valid = frame.loc[frame.status.eq("ok")].copy()
    all_features = valid.assign(feature_family="all")
    both = pd.concat([valid, all_features], ignore_index=True)
    keys = ["record_id", "subject_id", "feature_family", *extra]
    by_record = both.groupby(keys, sort=True)[metrics].median().reset_index()
    counts = both.groupby(keys, sort=True).size().rename("n_valid_features").reset_index()
    by_record = by_record.merge(counts, on=keys, validate="one_to_one")
    groups = ["feature_family", *extra]
    summary = by_record.groupby(groups, sort=True)[metrics].median().reset_index()
    n = by_record.groupby(groups).agg(n_records=("record_id", "nunique"), n_subjects=("subject_id", "nunique")).reset_index()
    return by_record, summary.merge(n, on=groups, validate="one_to_one")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/diagnostics/pm_temporal_predictability_audit_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = low.load_config(root / "experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json")
    context = low.prepare_protocol(config, root=root, feature_cache_dir=args.feature_cache_dir,
                                   output_dir=root / "benchmark_results/_temporal_ssa_protocol")
    full, matrix, names = context.full, context.matrix, context.feature_names
    families = [name.split("__", 1)[0] for name in names]
    # Fixed canonical IDs, not selected by observed dynamics or PM outcomes.
    representatives = [j for family in sorted(set(families))
                       for j in [i for i, f in enumerate(families) if f == family][:5]]
    common = {
        "experiment_type": "exploratory", "seed": 42, "surrogates": 10,
        "segments": "longest exact-contiguous 10s segment per record; earliest breaks ties",
        "scaling": "segment median/IQR; zero IQR and nonfinite trajectories skipped",
        "surrogate": "independent temporal permutation within same segment; preserve marginal values",
        "aggregation": "median features within record/family; equal-record median globally; all weights features equally",
        "independent_unit_caution": "records within participant are dependent; no inferential p values",
        "reference_protocol_hash": context.protocol["protocol_hash"],
        "input_sha256": hashlib.sha256(memoryview(matrix).cast("B")).hexdigest(),
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), Path(state.__file__), root / "bench/analysis/temporal_feature_diagnostics.py"]},
    }
    low._atomic_json(output / "ssa_protocol.json", {**common, "minimum_windows": 100,
        "L": "min(60,floor(N/2))", "L_rationale": "fixed <=600s trajectory, <=half segment; no optimization",
        "energy": "squared singular values / sum; computed via symmetric Hankel Gram eigenvalues",
        "effective_rank": "exp(Shannon entropy of energy)", "participation_ratio": "1/sum(energy**2)",
        "concentration": "sum(energy**2)", "delta": "surrogate median minus observed"})
    low._atomic_json(output / "fnn_protocol.json", {**common, "tau_windows": 1,
        "dimensions": list(range(1, 9)), "Rtol": 10, "Atol": 2,
        "theiler_windows": "max(12,m*tau)", "minimum_vectors": "max(500,50*(m+1))",
        "minimum_valid_neighbors": 500, "zero_distance_rule": "exclude distances<=1e-12; skip if >10%",
        "length_rationale": "conservative exploratory heuristic; no guarantee of reliable dimension",
        "representative_selection": "first five canonical feature IDs per family; frozen before results",
        "representative_features": [names[i] for i in representatives],
        "reference": "https://doi.org/10.1103/PhysRevA.45.3403",
        "interpretation": "curves only; no automatic attractor dimension or predictive horizon selection"})
    rows, fnn_rows, segments, spectra = [], [], [], []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for record_number, (record, group) in enumerate(full.groupby("record_id", sort=True)):
            indices = max(contiguous_segments(group), key=len)
            metadata = {"record_id": str(record), "subject_id": str(group.subject_id.iloc[0])}
            segments.append({**metadata, "n_windows": len(indices),
                             "t_start": float(full.loc[indices[0], "t_start"]),
                             "t_end": float(full.loc[indices[-1], "t_end"]),
                             "ssa_status": "ok" if len(indices) >= 100 else "insufficient_sequence_length"})
            for j, name in enumerate(names):
                values = np.asarray(matrix[indices, j], dtype=float)
                meta = {**metadata, "feature": name, "feature_family": families[j]}
                # Independent streams: FNN eligibility never changes SSA randomness.
                rng = np.random.default_rng(np.random.SeedSequence([42, record_number, j, 0]))
                energy, observed = state.ssa_energy(values)
                row = {**meta, **observed}
                if observed["status"] == "ok":
                    surrogate = [state.ssa_energy(rng.permutation(values)) for _ in range(10)]
                    for metric in ("effective_rank", "participation_ratio", "rank80", "rank90", "rank95",
                                   "leading_energy", "spectral_concentration"):
                        row[f"observed_{metric}"] = observed[metric]
                        row[f"surrogate_{metric}"] = float(np.median([s[1][metric] for s in surrogate]))
                        row[f"delta_{metric}"] = row[f"surrogate_{metric}"] - observed[metric]
                    surrogate_energy = np.median(np.array([s[0] for s in surrogate]), axis=0)
                    # Compact spectra go to runtime only; no trajectory matrices saved.
                    spectra.append({**meta, "energy": energy.tolist(), "cumulative_energy": np.cumsum(energy).tolist(),
                                    "surrogate_median_energy": surrogate_energy.tolist()})
                rows.append(row)
                if j in representatives:
                    obs_curve = state.fnn_curve(values)
                    valid = any(r["status"] == "ok" for r in obs_curve)
                    fnn_rng = np.random.default_rng(np.random.SeedSequence([42, record_number, j, 1]))
                    sur_curves = [state.fnn_curve(fnn_rng.permutation(values)) for _ in range(10)] if valid else []
                    for k, obs in enumerate(obs_curve):
                        result = {**meta, **obs, "observed_fnn_fraction": obs["fnn_fraction"],
                                  "surrogate_fnn_fraction": np.nan, "delta_fnn_fraction": np.nan}
                        if obs["status"] == "ok":
                            sur = [curve[k]["fnn_fraction"] for curve in sur_curves if curve[k]["status"] == "ok"]
                            result["valid_surrogates"] = len(sur)
                            if len(sur) == 10:
                                result["surrogate_fnn_fraction"] = float(np.median(sur))
                                result["delta_fnn_fraction"] = result["surrogate_fnn_fraction"] - obs["fnn_fraction"]
                            else:
                                result["status"] = "unreliable_surrogates"
                        fnn_rows.append(result)
            print(f"Diagnostics record {record_number + 1}/86: N={len(indices)}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    ssa = pd.DataFrame(rows)
    fnn = pd.DataFrame(fnn_rows)
    metrics = [c for c in ssa if c.startswith(("observed_", "surrogate_", "delta_"))]
    ssa_record, ssa_summary = summarize(ssa, metrics, [])
    fnn_record, fnn_summary = summarize(fnn, ["observed_fnn_fraction", "surrogate_fnn_fraction", "delta_fnn_fraction"], ["dimension"])
    for name, frame in [("ssa_by_feature", ssa), ("ssa_by_record_family", ssa_record), ("ssa_summary", ssa_summary),
                        ("fnn_by_feature", fnn), ("fnn_by_record_family", fnn_record), ("fnn_summary", fnn_summary),
                        ("state_space_segments", pd.DataFrame(segments))]:
        low._write_csv(output / f"{name}.csv", frame)
    runtime = root / "benchmark_results/pm_temporal_state_space_v1"
    low._atomic_json(runtime / "ssa_spectra.json", {"spectra": spectra})
    low._atomic_json(output / "state_space_execution.json", {
        "status": "complete", "elapsed_s": time.perf_counter()-started,
        "ssa_status_counts": ssa.status.value_counts().to_dict(),
        "fnn_status_counts": fnn.status.value_counts().to_dict(), "classifier_fits": 0,
        "spectra_path": str(runtime.relative_to(root) / "ssa_spectra.json")})
    print(ssa_summary.to_string(index=False))
    print(fnn_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

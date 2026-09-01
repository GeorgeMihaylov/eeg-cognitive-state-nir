from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.analysis.auxiliary_corn_policy_statistics import (
    AuxiliaryCornPolicyStatistics,
    average_subject_metrics_across_seeds,
    calculate_policy_subject_metrics,
    require_three_way_alignment,
)


def _base_frame(seed: int, feature_group: str, method: str) -> pd.DataFrame:
    rows = []
    for index in range(53):
        fold = index % 5 + 1
        for label_index in range(5):
            true = label_index
            pred = true
            token = index * 5 + label_index
            if method == "corn" and (token + seed) % 11 == 0:
                pred = min(4, true + 1)
            if method == "policy" and (token + seed) % 17 == 0:
                pred = max(0, true - 1)
            probabilities = np.full(5, 0.025, dtype=float)
            probabilities[pred] = 0.9
            row = {
                "sequence_id": f"sequence_{index:03d}_{label_index}",
                "fold": fold,
                "subject_id": f"subject_{index:03d}",
                "record_id": f"record_{index:03d}",
                "source": "gpn_data" if index % 2 else "Old_EEG",
                "y_true": true,
                "y_pred": pred,
                **{f"proba_{label}": float(probabilities[label]) for label in range(5)},
            }
            if method == "corn":
                row["expected_rank"] = float(probabilities @ np.arange(5))
            if method == "policy":
                row.update({
                    "feature_group": feature_group,
                    "seed": seed,
                    "policy_branch": (
                        "categorical_fallback" if fold == 1 and seed == 42
                        else "joint_selected"
                    ),
                    "aux_available": not (fold == 1 and seed == 42),
                    "aux_ordinal_prediction": min(4, pred + 1),
                    "categorical_expected_rank": float(probabilities @ np.arange(5)),
                    "selected_auxiliary_weight": (
                        np.nan if fold == 1 and seed == 42 else 0.25
                    ),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _by_seed() -> dict[int, dict[str, pd.DataFrame]]:
    result: dict[int, dict[str, pd.DataFrame]] = {}
    for seed in (7, 42, 123):
        result[seed] = {}
        for group in ("eeg_only", "eeg_pow"):
            for method in ("categorical", "corn", "policy"):
                result[seed][f"{method}_{group}"] = _base_frame(seed, group, method)
    return result


def test_three_way_alignment_accepts_exact_identity() -> None:
    frames = {
        method: _base_frame(7, "eeg_pow", method)
        for method in ("categorical", "corn", "policy")
    }
    audit = require_three_way_alignment(frames)
    assert audit["exact_match"] is True
    assert audit["rows"] == 265
    assert audit["subjects"] == 53


def test_three_way_alignment_rejects_changed_subject() -> None:
    frames = {
        method: _base_frame(7, "eeg_pow", method)
        for method in ("categorical", "corn", "policy")
    }
    frames["policy"].loc[0, "subject_id"] = "changed"
    with pytest.raises(ValueError, match="alignment failed"):
        require_three_way_alignment(frames)


def test_subject_metrics_and_seed_average_have_expected_shape() -> None:
    subject_seed = calculate_policy_subject_metrics(_by_seed())
    assert len(subject_seed) == 3 * 2 * 3 * 53
    policy = subject_seed[subject_seed["method"] == "policy"]
    assert policy["auxiliary_coverage_fraction"].between(0, 1).all()
    averaged = average_subject_metrics_across_seeds(subject_seed)
    assert len(averaged) == 3 * 2 * 53
    assert averaged.groupby("run_key")["subject_id"].nunique().eq(53).all()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_analysis_executes_end_to_end(tmp_path: Path) -> None:
    run_index = []
    policy_frames = []
    for seed in (7, 42, 123):
        for group in ("eeg_only", "eeg_pow"):
            policy_frames.append(_base_frame(seed, group, "policy"))
            for method in ("categorical", "corn"):
                run_dir = tmp_path / "runs" / f"{method}_{group}_seed{seed}"
                prediction_dir = run_dir / "task" / "group_kfold_subject"
                prediction_dir.mkdir(parents=True)
                _base_frame(seed, group, method).to_parquet(
                    prediction_dir / "predictions.parquet", index=False
                )
                run_index.append({
                    "method": method,
                    "feature_group": group,
                    "seed": seed,
                    "run_directory": str(run_dir),
                })

    ordinal_summary = tmp_path / "ordinal_summary.json"
    ordinal_summary.write_text(
        json.dumps({"run_index": run_index}), encoding="utf-8"
    )
    policy_input = tmp_path / "policy_input.parquet"
    pd.concat(policy_frames, ignore_index=True).to_parquet(policy_input, index=False)
    outcomes = []
    for seed in (7, 42, 123):
        for group in ("eeg_only", "eeg_pow"):
            for fold in range(1, 6):
                fallback = fold == 1 and seed == 42
                outcomes.append({
                    "selection_id": f"{group}_seed{seed}_fold{fold:02d}",
                    "feature_group": group,
                    "seed": seed,
                    "outer_fold": fold,
                    "policy_branch": "categorical_fallback" if fallback else "joint_selected",
                    "fallback_reason": "guard" if fallback else None,
                    "outer_test_rows": int((_base_frame(seed, group, "policy")["fold"] == fold).sum()),
                })
    policy_summary = tmp_path / "policy_summary.json"
    policy_summary.write_text(json.dumps({
        "status": "completed",
        "ready_for_subject_level_analysis": True,
        "selection_units_joint": 24,
        "selection_units_fallback": 6,
        "selected_lambda_counts": {"0.25": 24},
        "artifacts": {"subject_level_analysis_input": str(policy_input)},
        "outcomes": outcomes,
    }), encoding="utf-8")
    source = tmp_path / "source.parquet"
    pd.DataFrame({"value": [1, 2, 3]}).to_parquet(source, index=False)
    config = tmp_path / "analysis.yaml"
    report = tmp_path / "report.md"
    summary = tmp_path / "summary.json"
    decision = tmp_path / "decision.md"
    output = tmp_path / "output"
    config.write_text(yaml.safe_dump({
        "analysis": {
            "type": "auxiliary_corn_policy_statistics",
            "ordinal_run_summary": str(ordinal_summary),
            "policy_summary": str(policy_summary),
            "output_dir": str(output),
            "report_path": str(report),
            "summary_path": str(summary),
            "decision_report_path": str(decision),
            "bootstrap_samples": 100,
            "random_state": 42,
        },
        "expected": {
            "sequences": 265,
            "subjects": 53,
            "source_parquet": str(source),
            "source_parquet_sha256": _sha256(source),
        },
    }), encoding="utf-8")

    analysis = AuxiliaryCornPolicyStatistics(config)
    plan = analysis.plan()
    assert plan["valid"] is True
    result = analysis.execute()
    assert result["status"] == "completed"
    assert result["subject_seed_rows"] == 954
    assert result["averaged_subject_rows"] == 318
    assert result["alignment_exact"] is True
    assert report.is_file()
    assert summary.is_file()
    assert decision.is_file()
    assert (output / "subject_multiseed_metrics.parquet").is_file()
    assert (output / "paired_comparisons.parquet").is_file()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["policy_composition"]["fallback_units"] == 6
    assert len(payload["primary_hypotheses"]) == 6

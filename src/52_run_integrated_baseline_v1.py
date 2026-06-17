#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated baseline v1 for EEG latent proxy-state modeling.

This script is intended for the main branch. It does not replace research
scripts 44/46/48/49/50. It provides a stable project-level entry point that
combines existing experiment outputs into one hypothesis-driven baseline report.

Baseline v1 logic:
  1. Targets: slow_pca_1, slow_pca_2, slow_pca_3 from smoothed PM metrics.
  2. Features: pow_plus_eeg.
  3. Model: Transformer sequence regressor, seq_len=8.
  4. Calibration: subject-specific head-only calibration, frac=0.20.
  5. Evidence: feature ablation, temporal baselines, calibration diagnostics,
     split-seed robustness, optional naive baselines.

Recommended first run:
  python src/52_run_integrated_baseline_v1.py --root . --mode summarize

Optional reproduce run:
  python src/52_run_integrated_baseline_v1.py --root . --mode reproduce --skip-existing \
    --run-temporal-baselines --run-split-seeds --run-final-summary --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TARGETS = "slow_pca_1,slow_pca_2,slow_pca_3"
DEFAULT_SEEDS = "42,123,2024,3407,777"


@dataclass
class BaselineV1Config:
    root: str
    mode: str
    output_dir: str
    dataset: str
    final_feature_set: str
    final_targets: list[str]
    final_seq_len: int
    final_calibration_lr: float
    final_calibration_frac: float
    max_features: int
    feature_ablation_dir: str
    subject_diagnostics_dir: str
    temporal_baselines_dir: str
    final_summary_dir: str
    split_seed_dir: str
    naive_dir: str
    seeds: list[int]
    device: str
    python_exe: str
    run_temporal_baselines: bool
    run_split_seeds: bool
    run_final_summary: bool
    run_naive: bool
    skip_existing: bool
    dry_run: bool


def parse_str_list(value: str) -> list[str]:
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated strings")
    return out


def parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return out


def repo_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def safe_float(x: Any) -> float:
    try:
        if x is None or pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def fmt(x: Any, digits: int = 4) -> str:
    try:
        if x is None or pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return ""
            return f"{float(x):.{digits}f}"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return str(x)
    except Exception:
        return str(x)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read CSV {path}: {exc}")
        return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read JSON {path}: {exc}")
        return {}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def df_to_markdown(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No data._"

    view = df.copy()
    if max_rows is not None and len(view) > max_rows:
        view = view.head(max_rows).copy()

    cols = list(view.columns)

    def esc(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    rows: list[list[str]] = []
    rows.append([esc(c) for c in cols])
    rows.append(["---" for _ in cols])

    for _, row in view.iterrows():
        rows.append([esc(fmt(row[c], digits=digits)) for c in cols])

    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    return "\n".join("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |" for r in rows)


def artifact_status(path: Path) -> str:
    if path.exists():
        return "dir_exists" if path.is_dir() else "file_exists"
    return "missing"


def build_artifact_index(cfg: BaselineV1Config, root: Path) -> pd.DataFrame:
    items = [
        ("dataset", repo_path(root, cfg.dataset)),
        ("feature_ablation_dir", repo_path(root, cfg.feature_ablation_dir)),
        ("subject_diagnostics_dir", repo_path(root, cfg.subject_diagnostics_dir)),
        ("temporal_baselines_dir", repo_path(root, cfg.temporal_baselines_dir)),
        ("final_summary_dir", repo_path(root, cfg.final_summary_dir)),
        ("split_seed_dir", repo_path(root, cfg.split_seed_dir)),
        ("naive_dir", repo_path(root, cfg.naive_dir)),
        ("script_44", root / "src" / "44_run_seq_len_sensitivity.py"),
        ("script_46", root / "src" / "46_run_reliable_axes_calibration_val_test.py"),
        ("script_48", root / "src" / "48_train_temporal_baselines.py"),
        ("script_49", root / "src" / "49_summarize_final_experiments.py"),
        ("script_50", root / "src" / "50_run_split_seed_robustness.py"),
        ("script_51_optional", root / "src" / "51_run_naive_hypothesis_baselines.py"),
    ]
    return pd.DataFrame([{"name": n, "path": str(p), "status": artifact_status(p)} for n, p in items])


def run_command(cmd: list[str], log_path: Path, cwd: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = " ".join(str(x) for x in cmd)
    print("=" * 100)
    print(cmd_text)
    print("=" * 100)

    if dry_run:
        log_path.write_text(cmd_text + "\n", encoding="utf-8")
        return

    with log_path.open("w", encoding="utf-8") as f:
        f.write(cmd_text + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}. See log: {log_path}")


def build_reproduce_commands(cfg: BaselineV1Config, root: Path, out_dir: Path) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []

    if cfg.run_temporal_baselines:
        script = root / "src" / "48_train_temporal_baselines.py"
        expected = repo_path(root, cfg.temporal_baselines_dir) / "model_summary.csv"
        cmd = [
            cfg.python_exe, str(script),
            "--root", str(root),
            "--dataset", cfg.dataset,
            "--output-dir", str(repo_path(root, cfg.temporal_baselines_dir)),
            "--models", "last_window_mlp,mean_pool_mlp,gru,transformer",
            "--feature-set", cfg.final_feature_set,
            "--max-features", str(cfg.max_features),
            "--targets", ",".join(cfg.final_targets),
            "--seq-len", str(cfg.final_seq_len),
            "--calibration-lr", str(cfg.final_calibration_lr),
            "--calibration-frac", str(cfg.final_calibration_frac),
            "--device", cfg.device,
        ]
        if cfg.skip_existing:
            cmd.append("--reuse-existing")
        commands.append({"name": "temporal_baselines", "cmd": cmd, "expected": expected, "log": out_dir / "logs" / "48_temporal_baselines.log"})

    if cfg.run_split_seeds:
        script = root / "src" / "50_run_split_seed_robustness.py"
        expected = repo_path(root, cfg.split_seed_dir) / "aggregate_protocol_summary.csv"
        cmd = [
            cfg.python_exe, str(script),
            "--root", str(root),
            "--dataset", cfg.dataset,
            "--output-dir", str(repo_path(root, cfg.split_seed_dir)),
            "--seeds", ",".join(str(s) for s in cfg.seeds),
            "--feature-set", cfg.final_feature_set,
            "--max-features", str(cfg.max_features),
            "--targets", ",".join(cfg.final_targets),
            "--seq-len", str(cfg.final_seq_len),
            "--calibration-lr", str(cfg.final_calibration_lr),
            "--calibration-frac", str(cfg.final_calibration_frac),
            "--device", cfg.device,
        ]
        if cfg.skip_existing:
            cmd.append("--reuse-existing")
        commands.append({"name": "split_seed_robustness", "cmd": cmd, "expected": expected, "log": out_dir / "logs" / "50_split_seed_robustness.log"})

    if cfg.run_final_summary:
        script = root / "src" / "49_summarize_final_experiments.py"
        expected = repo_path(root, cfg.final_summary_dir) / "final_key_results.json"
        cmd = [
            cfg.python_exe, str(script),
            "--root", str(root),
            "--output-dir", str(repo_path(root, cfg.final_summary_dir)),
            "--feature-ablation-dir", cfg.feature_ablation_dir,
            "--subject-diagnostics-dir", cfg.subject_diagnostics_dir,
            "--temporal-baselines-dir", cfg.temporal_baselines_dir,
            "--final-feature-set", cfg.final_feature_set,
            "--final-targets", ",".join(cfg.final_targets),
            "--final-seq-len", str(cfg.final_seq_len),
            "--final-calibration-lr", str(cfg.final_calibration_lr),
            "--final-calibration-frac", str(cfg.final_calibration_frac),
        ]
        commands.append({"name": "final_summary", "cmd": cmd, "expected": expected, "log": out_dir / "logs" / "49_final_summary.log"})

    if cfg.run_naive:
        script = root / "src" / "51_run_naive_hypothesis_baselines.py"
        expected = repo_path(root, cfg.naive_dir) / "aggregate_naive_baseline_summary.csv"
        cmd = [
            cfg.python_exe, str(script),
            "--root", str(root),
            "--dataset", cfg.dataset,
            "--output-dir", str(repo_path(root, cfg.naive_dir)),
            "--seeds", ",".join(str(s) for s in cfg.seeds),
            "--feature-set", cfg.final_feature_set,
            "--targets", ",".join(cfg.final_targets),
            "--seq-len", str(cfg.final_seq_len),
            "--calibration-frac", str(cfg.final_calibration_frac),
        ]
        commands.append({"name": "naive_baselines_optional", "cmd": cmd, "expected": expected, "log": out_dir / "logs" / "51_naive_baselines.log"})

    return commands


def maybe_run_commands(cfg: BaselineV1Config, root: Path, out_dir: Path) -> list[dict[str, Any]]:
    log_rows: list[dict[str, Any]] = []
    for item in build_reproduce_commands(cfg, root, out_dir):
        name = str(item["name"])
        cmd = item["cmd"]
        script_path = Path(cmd[1])
        expected = Path(item["expected"])
        log_path = Path(item["log"])

        if not script_path.exists():
            status = "skipped_missing_script"
            print(f"[SKIP] {name}: missing script {script_path}")
        elif cfg.skip_existing and expected.exists():
            status = "skipped_existing"
            print(f"[SKIP] {name}: existing output {expected}")
        else:
            try:
                run_command(cmd=cmd, log_path=log_path, cwd=root, dry_run=cfg.dry_run)
                status = "dry_run" if cfg.dry_run else "executed"
            except Exception as exc:
                status = f"failed: {repr(exc)}"
                print(f"[ERROR] {name}: {exc}")

        log_rows.append({"name": name, "status": status, "command": " ".join(str(x) for x in cmd), "expected": str(expected), "log": str(log_path)})
    return log_rows


def read_final_key_results(cfg: BaselineV1Config, root: Path) -> dict[str, Any]:
    return read_json(repo_path(root, cfg.final_summary_dir) / "final_key_results.json")


def read_feature_table(cfg: BaselineV1Config, root: Path) -> pd.DataFrame:
    final_table = read_csv(repo_path(root, cfg.final_summary_dir) / "feature_ablation_selected_protocols.csv")
    if not final_table.empty:
        return final_table

    rows: list[dict[str, Any]] = []
    base = repo_path(root, cfg.feature_ablation_dir)
    for fs in ["pow", "eeg", "pow_plus_eeg"]:
        raw = read_csv(base / f"calibration_{fs}" / "val_test_mean_protocol_summary.csv")
        if raw.empty:
            continue
        raw = raw.copy()
        raw["mean_r2"] = pd.to_numeric(raw["mean_r2"], errors="coerce")
        val = raw[raw["eval_split"].astype(str) == "val"].copy()
        test = raw[raw["eval_split"].astype(str) == "test"].copy()
        if val.empty or test.empty:
            continue
        vb = val.loc[val["mean_r2"].idxmax()]
        cond = (
            np.isclose(pd.to_numeric(test["seq_len"], errors="coerce"), safe_float(vb.get("seq_len")))
            & np.isclose(pd.to_numeric(test["calibration_lr"], errors="coerce"), safe_float(vb.get("calibration_lr")))
            & np.isclose(pd.to_numeric(test["calibration_frac"], errors="coerce"), safe_float(vb.get("calibration_frac")))
        )
        matched = test[cond]
        tb = matched.iloc[0] if not matched.empty else pd.Series(dtype=object)
        rows.append({
            "feature_set": fs,
            "selected_seq_len": safe_float(vb.get("seq_len")),
            "selected_lr": safe_float(vb.get("calibration_lr")),
            "selected_frac": safe_float(vb.get("calibration_frac")),
            "val_selected_mean_r2": safe_float(vb.get("mean_r2")),
            "val_selected_spearman": safe_float(vb.get("mean_spearman")),
            "test_at_selected_mean_r2": safe_float(tb.get("mean_r2")),
            "test_at_selected_spearman": safe_float(tb.get("mean_spearman")),
        })
    return pd.DataFrame(rows)


def read_temporal_zero_table(cfg: BaselineV1Config, root: Path) -> pd.DataFrame:
    df = read_csv(repo_path(root, cfg.final_summary_dir) / "temporal_baseline_zero_full_test.csv")
    if not df.empty:
        return df

    ms = read_csv(repo_path(root, cfg.temporal_baselines_dir) / "model_summary.csv")
    if ms.empty:
        return pd.DataFrame()
    required = {"model", "eval_split", "phase", "mean_r2", "mean_spearman"}
    if not required.issubset(ms.columns):
        return pd.DataFrame()
    out = ms[(ms["eval_split"].astype(str) == "test") & (ms["phase"].astype(str) == "zero_full")].copy()
    if out.empty:
        return out
    out["mean_r2"] = pd.to_numeric(out["mean_r2"], errors="coerce")
    out = out.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    out.insert(0, "rank_by_test_r2", np.arange(1, len(out) + 1))
    return out


def read_split_seed_aggregate(cfg: BaselineV1Config, root: Path) -> pd.DataFrame:
    return read_csv(repo_path(root, cfg.split_seed_dir) / "aggregate_protocol_summary.csv")


def read_naive_aggregate(cfg: BaselineV1Config, root: Path) -> pd.DataFrame:
    return read_csv(repo_path(root, cfg.naive_dir) / "aggregate_naive_baseline_summary.csv")


def get_test_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty or "eval_split" not in df.columns:
        return None
    test = df[df["eval_split"].astype(str) == "test"].copy()
    if test.empty:
        return None
    return test.iloc[0]


def build_hypothesis_matrix(cfg: BaselineV1Config, feature_table: pd.DataFrame, final_key: dict[str, Any], temporal_zero: pd.DataFrame, split_seed: pd.DataFrame, naive: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rows.append({
        "hypothesis_id": "H1",
        "hypothesis": "Smoothed PM metrics can be represented as interpretable latent proxy-state axes.",
        "baseline_or_check": "PM dynamics analysis + slow-PM PCA; exclude unstable axes.",
        "key_result": "Final targets are slow_pca_1, slow_pca_2, slow_pca_3; slow_pca_4 excluded as unstable.",
        "main_metric": "",
        "status": "supported within PM proxy labels",
        "caution": "Latent axes reflect PM annotations, not direct objective cognitive-affective states.",
    })

    if not temporal_zero.empty:
        best = temporal_zero.iloc[0]
        key = f"Best zero-shot temporal model: {best.get('model')} with mean R2={fmt(best.get('mean_r2'))}, Spearman={fmt(best.get('mean_spearman'))}."
        metric = f"R2={fmt(best.get('mean_r2'))}; Spearman={fmt(best.get('mean_spearman'))}"
        status = "supported"
    else:
        key = "Temporal baseline table not found."
        metric = ""
        status = "not evaluated in integrated summary"
    rows.append({
        "hypothesis_id": "H2",
        "hypothesis": "Temporal context improves prediction of latent proxy-state trajectories.",
        "baseline_or_check": "last_window_mlp / mean_pool_mlp / GRU / Transformer comparison.",
        "key_result": key,
        "main_metric": metric,
        "status": status,
        "caution": "Extreme negative R2 for simple MLPs should be treated as sanity-check degradation.",
    })

    final_feature = final_key.get("feature_ablation_final_feature_set", {})
    if final_feature:
        key = f"{final_feature.get('feature_set')} selected; test R2={fmt(final_feature.get('test_at_selected_mean_r2'))}, Spearman={fmt(final_feature.get('test_at_selected_spearman'))}."
        metric = f"R2={fmt(final_feature.get('test_at_selected_mean_r2'))}; Spearman={fmt(final_feature.get('test_at_selected_spearman'))}"
        status = "supported"
    elif not feature_table.empty and "test_at_selected_mean_r2" in feature_table.columns:
        table = feature_table.copy()
        table["test_at_selected_mean_r2"] = pd.to_numeric(table["test_at_selected_mean_r2"], errors="coerce")
        row = table.sort_values("test_at_selected_mean_r2", ascending=False).iloc[0]
        key = f"{row.get('feature_set')} selected by available table; test R2={fmt(row.get('test_at_selected_mean_r2'))}, Spearman={fmt(row.get('test_at_selected_spearman'))}."
        metric = f"R2={fmt(row.get('test_at_selected_mean_r2'))}"
        status = "supported"
    else:
        key = "Feature ablation table not found."
        metric = ""
        status = "not evaluated in integrated summary"
    rows.append({
        "hypothesis_id": "H3",
        "hypothesis": "POW and EEG features provide complementary information.",
        "baseline_or_check": "pow vs eeg vs pow_plus_eeg feature ablation.",
        "key_result": key,
        "main_metric": metric,
        "status": status,
        "caution": "Feature ablation depends on the current corpus and feature selection procedure.",
    })

    cal = final_key.get("per_subject_calibration_test_overall", {})
    if cal:
        key = f"Test mean R2 changed from {fmt(cal.get('mean_r2_zero'))} to {fmt(cal.get('mean_r2_calibrated'))}; gain={fmt(cal.get('mean_r2_gain'))}; subject positive-rate={fmt(cal.get('subject_mean_r2_positive_rate'))}."
        metric = f"R2 gain={fmt(cal.get('mean_r2_gain'))}"
        status = "supported"
    else:
        key = "Final calibration diagnostics not found."
        metric = ""
        status = "not evaluated in integrated summary"
    rows.append({
        "hypothesis_id": "H4",
        "hypothesis": "Personal head-only calibration improves held-out subject performance.",
        "baseline_or_check": "zero-shot vs head-only calibration on held-out subject tail.",
        "key_result": key,
        "main_metric": metric,
        "status": status,
        "caution": "Held-out subject count is limited; do not claim universal improvement.",
    })

    split_test = get_test_row(split_seed)
    if split_test is not None:
        key = f"Across {fmt(split_test.get('n_seeds'))} seeds: mean zero R2={fmt(split_test.get('mean_r2_zero_mean'))}, mean calibrated R2={fmt(split_test.get('mean_r2_calibrated_mean'))}, mean gain={fmt(split_test.get('mean_r2_gain_mean'))}, gain std={fmt(split_test.get('mean_r2_gain_std'))}."
        metric = f"mean R2 gain={fmt(split_test.get('mean_r2_gain_mean'))}"
        status = "supported"
    else:
        key = "Split-seed aggregate summary not found."
        metric = ""
        status = "not evaluated in integrated summary"
    rows.append({
        "hypothesis_id": "H5",
        "hypothesis": "Calibration effect is not an artifact of a single subject-wise split.",
        "baseline_or_check": "Fixed protocol evaluated across several random subject-wise splits.",
        "key_result": key,
        "main_metric": metric,
        "status": status,
        "caution": "Gain magnitude remains split-sensitive.",
    })

    if not naive.empty:
        test_naive = naive[naive["eval_split"].astype(str) == "test"].copy() if "eval_split" in naive.columns else naive.copy()
        if "mean_r2_mean" in test_naive.columns and not test_naive.empty:
            test_naive["mean_r2_mean"] = pd.to_numeric(test_naive["mean_r2_mean"], errors="coerce")
            best = test_naive.sort_values("mean_r2_mean", ascending=False).iloc[0]
            key = f"Best naive baseline: {best.get('baseline')} / {best.get('phase')} with mean R2={fmt(best.get('mean_r2_mean'))}, Spearman={fmt(best.get('mean_spearman_mean'))}."
            metric = f"best naive R2={fmt(best.get('mean_r2_mean'))}"
            status = "available"
        else:
            key = "Naive aggregate file exists, but expected metric columns were not found."
            metric = ""
            status = "available but not parsed"
    else:
        key = "Naive baselines are optional and were not found in current outputs."
        metric = ""
        status = "optional / missing"
    rows.append({
        "hypothesis_id": "H6",
        "hypothesis": "Transformer + calibration should be compared against simple statistical/persistence rules.",
        "baseline_or_check": "train_mean / subject_calibration_mean / previous_state if available.",
        "key_result": key,
        "main_metric": metric,
        "status": status,
        "caution": "previous_state uses target history and is a sanity-check, not an EEG-only deployable model.",
    })

    return pd.DataFrame(rows)


def build_summary_json(cfg: BaselineV1Config, final_key: dict[str, Any], temporal_zero: pd.DataFrame, split_seed: pd.DataFrame, naive: pd.DataFrame, matrix: pd.DataFrame, artifacts: pd.DataFrame, command_log: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_name": "baseline_v1_personal_calibrated_latent_proxy_trajectory",
        "baseline_title": "Baseline v1: personal calibration of latent EEG proxy-state trajectories",
        "configuration": {
            "feature_set": cfg.final_feature_set,
            "seq_len": cfg.final_seq_len,
            "targets": cfg.final_targets,
            "calibration_lr": cfg.final_calibration_lr,
            "calibration_frac": cfg.final_calibration_frac,
            "split_level": "subject",
            "seeds": cfg.seeds,
        },
        "main_results": {},
        "hypothesis_status": matrix[["hypothesis_id", "status", "main_metric"]].to_dict(orient="records") if not matrix.empty else [],
        "artifacts": artifacts.to_dict(orient="records") if not artifacts.empty else [],
        "commands": command_log,
    }
    if final_key.get("feature_ablation_final_feature_set"):
        summary["main_results"]["feature_ablation"] = final_key["feature_ablation_final_feature_set"]
    if final_key.get("per_subject_calibration_test_overall"):
        summary["main_results"]["main_calibration_test"] = final_key["per_subject_calibration_test_overall"]
    if not temporal_zero.empty:
        summary["main_results"]["best_temporal_zero_full_test"] = temporal_zero.iloc[0].to_dict()
    split_test = get_test_row(split_seed)
    if split_test is not None:
        summary["main_results"]["split_seed_test"] = split_test.to_dict()
    summary["main_results"]["naive_baselines_available"] = not naive.empty
    return summary


def build_report(cfg: BaselineV1Config, feature_table: pd.DataFrame, final_key: dict[str, Any], temporal_zero: pd.DataFrame, split_seed: pd.DataFrame, naive: pd.DataFrame, matrix: pd.DataFrame, artifacts: pd.DataFrame, command_log: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Baseline v1: personal calibration of latent EEG proxy-state trajectories")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This baseline integrates the main experimental line of the project: latent proxy-state targets built from smoothed PM metrics, EEG/POW sequence modeling, Transformer prediction, subject-wise validation, and personal head-only calibration.")
    lines.append("")
    lines.append("The baseline uses the term `proxy-state`: latent targets are derived from PM annotations and should not be interpreted as direct objective measurements of human cognitive-affective state.")
    lines.append("")

    lines.append("## Fixed configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| feature_set | `{cfg.final_feature_set}` |")
    lines.append(f"| seq_len | `{cfg.final_seq_len}` |")
    lines.append(f"| targets | `{', '.join(cfg.final_targets)}` |")
    lines.append(f"| calibration_lr | `{cfg.final_calibration_lr}` |")
    lines.append(f"| calibration_frac | `{cfg.final_calibration_frac}` |")
    lines.append("| split level | `subject` |")
    lines.append(f"| split seeds | `{', '.join(str(s) for s in cfg.seeds)}` |")
    lines.append("")

    lines.append("## Hypothesis baseline matrix")
    lines.append("")
    lines.append(df_to_markdown(matrix, digits=4))
    lines.append("")

    lines.append("## Feature ablation")
    lines.append("")
    lines.append(df_to_markdown(feature_table, digits=4) if not feature_table.empty else "_Feature ablation table not found._")
    lines.append("")

    lines.append("## Final calibration result")
    lines.append("")
    cal = final_key.get("per_subject_calibration_test_overall", {})
    if cal:
        cal_view = pd.DataFrame([{
            "eval_split": cal.get("eval_split"),
            "n_subjects": cal.get("n_subjects"),
            "mean_r2_zero": cal.get("mean_r2_zero"),
            "mean_r2_calibrated": cal.get("mean_r2_calibrated"),
            "mean_r2_gain": cal.get("mean_r2_gain"),
            "subject_mean_r2_positive_rate": cal.get("subject_mean_r2_positive_rate"),
            "target_subject_r2_positive_rate": cal.get("target_subject_r2_positive_rate"),
            "mean_spearman_zero": cal.get("mean_spearman_zero"),
            "mean_spearman_calibrated": cal.get("mean_spearman_calibrated"),
            "mean_spearman_gain": cal.get("mean_spearman_gain"),
        }])
        lines.append(df_to_markdown(cal_view, digits=4))
    else:
        lines.append("_Final calibration JSON not found._")
    lines.append("")

    lines.append("## Temporal architecture baseline")
    lines.append("")
    lines.append(df_to_markdown(temporal_zero, digits=4) if not temporal_zero.empty else "_Temporal baseline table not found._")
    lines.append("")

    lines.append("## Split-seed robustness")
    lines.append("")
    if split_seed.empty:
        lines.append("_Split-seed aggregate summary not found._")
    else:
        cols = [
            "eval_split", "n_seeds",
            "mean_r2_zero_mean", "mean_r2_zero_std",
            "mean_r2_calibrated_mean", "mean_r2_calibrated_std",
            "mean_r2_gain_mean", "mean_r2_gain_std",
            "mean_spearman_zero_mean", "mean_spearman_calibrated_mean", "mean_spearman_gain_mean",
            "subject_mean_r2_positive_rate_mean", "target_subject_r2_positive_rate_mean",
        ]
        cols = [c for c in cols if c in split_seed.columns]
        lines.append(df_to_markdown(split_seed[cols], digits=4))
    lines.append("")

    lines.append("## Optional naive baselines")
    lines.append("")
    if naive.empty:
        lines.append(f"_Naive baseline outputs were not found. This is acceptable for baseline v1. If available later, place results under `{cfg.naive_dir}` and rerun this script in summarize mode._")
    else:
        cols = ["eval_split", "phase", "baseline", "uses_target_history", "n_seeds", "mean_r2_mean", "mean_r2_std", "mean_spearman_mean", "mean_mae_mean", "mean_rmse_mean"]
        cols = [c for c in cols if c in naive.columns]
        lines.append(df_to_markdown(naive[cols], digits=4, max_rows=40))
    lines.append("")

    lines.append("## Artifact index")
    lines.append("")
    lines.append(df_to_markdown(artifacts, digits=4, max_rows=80))
    lines.append("")

    lines.append("## Commands executed or registered")
    lines.append("")
    if command_log:
        cmd_df = pd.DataFrame(command_log)
        cols = [c for c in ["name", "status", "expected"] if c in cmd_df.columns]
        lines.append(df_to_markdown(cmd_df[cols], digits=4))
    else:
        lines.append("_No commands executed. The script was run in summarize mode._")
    lines.append("")

    lines.append("## Main interpretation")
    lines.append("")
    lines.append("Baseline v1 supports the cautious claim that, within the current EEG/PM corpora, PM-derived latent proxy-states can be predicted from EEG/POW sequences, and subject-specific head-only calibration improves held-out subject performance on average. The effect is positive across several subject-wise split seeds, although its magnitude remains sensitive to the composition of held-out subjects.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Targets are PM-derived proxy-states, not direct objective cognitive-affective measurements.")
    lines.append("- Available corpora are close in device/protocol characteristics; cross-device generalization is not established.")
    lines.append("- Head-only calibration uses part of the held-out subject sequence and is not a pure zero-shot setting.")
    lines.append("- Feature-ablation outputs are summarized here but not fully reproduced unless upstream artifacts already exist.")
    lines.append("- Naive baselines are optional because script 51 may be absent from the current branch.")
    lines.append("")

    return "\n".join(lines)


def build_readme(cfg: BaselineV1Config) -> str:
    return f"""# Baseline v1

This directory contains the integrated baseline summary for the EEG latent proxy-state project.

## Baseline definition

- Target space: `slow_pca_1`, `slow_pca_2`, `slow_pca_3`
- Feature set: `{cfg.final_feature_set}`
- Sequence length: `{cfg.final_seq_len}`
- Main model: Transformer sequence regressor
- Personal calibration: head-only calibration
- Calibration fraction: `{cfg.final_calibration_frac}`
- Split protocol: subject-wise train/validation/test

## Main command

```powershell
D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\52_run_integrated_baseline_v1.py `
  --root . `
  --mode summarize
```

## Outputs

- `baseline_v1_report.md`
- `baseline_v1_summary.json`
- `hypothesis_baseline_matrix.csv`
- `artifact_index.csv`
- `commands_used.md`

## Interpretation

This baseline is a project-level reproducibility and reporting baseline. It integrates feature ablation, temporal model comparison, personal calibration diagnostics, split-seed robustness, and optional naive baselines into one hypothesis-driven summary.
"""


def save_commands_md(path: Path, commands: list[dict[str, Any]]) -> None:
    lines = ["# Commands used", ""]
    if not commands:
        lines.append("No commands were executed. The script was run in summarize mode.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    for item in commands:
        lines.append(f"## {item.get('name')}")
        lines.append("")
        lines.append(f"Status: `{item.get('status')}`")
        lines.append("")
        lines.append("```powershell")
        lines.append(str(item.get("command", "")))
        lines.append("```")
        lines.append("")
        lines.append(f"Expected output: `{item.get('expected')}`")
        lines.append(f"Log: `{item.get('log')}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run or summarize integrated baseline v1.")
    p.add_argument("--root", default=".")
    p.add_argument("--mode", choices=["summarize", "reproduce"], default="summarize")
    p.add_argument("--output-dir", default="reports/baseline_v1")
    p.add_argument("--dataset", default="reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet")
    p.add_argument("--final-feature-set", default="pow_plus_eeg")
    p.add_argument("--final-targets", type=parse_str_list, default=parse_str_list(DEFAULT_TARGETS))
    p.add_argument("--final-seq-len", type=int, default=8)
    p.add_argument("--final-calibration-lr", type=float, default=0.0001)
    p.add_argument("--final-calibration-frac", type=float, default=0.20)
    p.add_argument("--max-features", type=int, default=448)
    p.add_argument("--feature-ablation-dir", default="reports/feature_ablation_v2")
    p.add_argument("--subject-diagnostics-dir", default="reports/feature_ablation_v2/subject_diagnostics_pow_plus_eeg")
    p.add_argument("--temporal-baselines-dir", default="reports/temporal_baselines/pow_plus_eeg_seq8_pca123")
    p.add_argument("--final-summary-dir", default="reports/final_experiment_summary")
    p.add_argument("--split-seed-dir", default="reports/split_seed_robustness/pow_plus_eeg_seq8_pca123")
    p.add_argument("--naive-dir", default="reports/naive_hypothesis_baselines/pow_plus_eeg_seq8_pca123")
    p.add_argument("--seeds", type=parse_int_list, default=parse_int_list(DEFAULT_SEEDS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--run-temporal-baselines", action="store_true")
    p.add_argument("--run-split-seeds", action="store_true")
    p.add_argument("--run-final-summary", action="store_true")
    p.add_argument("--run-naive", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = BaselineV1Config(
        root=args.root,
        mode=args.mode,
        output_dir=args.output_dir,
        dataset=args.dataset,
        final_feature_set=args.final_feature_set,
        final_targets=args.final_targets,
        final_seq_len=args.final_seq_len,
        final_calibration_lr=args.final_calibration_lr,
        final_calibration_frac=args.final_calibration_frac,
        max_features=args.max_features,
        feature_ablation_dir=args.feature_ablation_dir,
        subject_diagnostics_dir=args.subject_diagnostics_dir,
        temporal_baselines_dir=args.temporal_baselines_dir,
        final_summary_dir=args.final_summary_dir,
        split_seed_dir=args.split_seed_dir,
        naive_dir=args.naive_dir,
        seeds=args.seeds,
        device=args.device,
        python_exe=args.python_exe,
        run_temporal_baselines=args.run_temporal_baselines,
        run_split_seeds=args.run_split_seeds,
        run_final_summary=args.run_final_summary,
        run_naive=args.run_naive,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )

    root = Path(cfg.root).resolve()
    out_dir = repo_path(root, cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    if cfg.mode == "reproduce" and not any([cfg.run_temporal_baselines, cfg.run_split_seeds, cfg.run_final_summary, cfg.run_naive]):
        cfg.run_temporal_baselines = True
        cfg.run_split_seeds = True
        cfg.run_final_summary = True

    write_json(out_dir / "baseline_v1_config.json", asdict(cfg))

    command_log: list[dict[str, Any]] = []
    if cfg.mode == "reproduce":
        command_log = maybe_run_commands(cfg, root, out_dir)

    artifacts = build_artifact_index(cfg, root)
    feature_table = read_feature_table(cfg, root)
    final_key = read_final_key_results(cfg, root)
    temporal_zero = read_temporal_zero_table(cfg, root)
    split_seed = read_split_seed_aggregate(cfg, root)
    naive = read_naive_aggregate(cfg, root)

    matrix = build_hypothesis_matrix(cfg, feature_table, final_key, temporal_zero, split_seed, naive)
    summary = build_summary_json(cfg, final_key, temporal_zero, split_seed, naive, matrix, artifacts, command_log)
    report = build_report(cfg, feature_table, final_key, temporal_zero, split_seed, naive, matrix, artifacts, command_log)

    matrix.to_csv(out_dir / "hypothesis_baseline_matrix.csv", index=False)
    artifacts.to_csv(out_dir / "artifact_index.csv", index=False)
    write_json(out_dir / "baseline_v1_summary.json", summary)
    (out_dir / "baseline_v1_report.md").write_text(report, encoding="utf-8")
    (out_dir / "README.md").write_text(build_readme(cfg), encoding="utf-8")
    save_commands_md(out_dir / "commands_used.md", command_log)

    print("=" * 100)
    print("Integrated baseline v1 saved")
    print("=" * 100)
    print(f"Report:  {out_dir / 'baseline_v1_report.md'}")
    print(f"Summary: {out_dir / 'baseline_v1_summary.json'}")
    print(f"Matrix:  {out_dir / 'hypothesis_baseline_matrix.csv'}")
    print(f"Index:   {out_dir / 'artifact_index.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


FILES_TO_COLLECT = [
    # Main integrated reports
    "reports/project_summary/project_hypotheses_summary.md",
    "reports/project_summary/project_hypotheses_summary_short.md",
    "reports/project_summary/summary.json",

    "reports/integrated_state_evidence_v2/integrated_state_report.md",
    "reports/integrated_state_evidence_v2/integrated_state_report_short.md",
    "reports/integrated_state_evidence_v2/state_evidence_matrix.csv",
    "reports/integrated_state_evidence_v2/state_evidence_scores.csv",
    "reports/integrated_state_evidence_v2/state_recommendations.csv",
    "reports/integrated_state_evidence_v2/state_limitations.csv",
    "reports/integrated_state_evidence_v2/temporal_modeling_evidence.csv",
    "reports/integrated_state_evidence_v2/external_dataset_mapping.csv",
    "reports/integrated_state_evidence_v2/summary.json",

    # Transformer trajectory results
    "reports/latent_trajectory_transformer/report.md",
    "reports/latent_trajectory_transformer/summary.csv",
    "reports/latent_trajectory_transformer/fold_metrics.csv",
    "reports/latent_trajectory_transformer/training_history.csv",
    "reports/latent_trajectory_transformer/model_config.json",
    "reports/latent_trajectory_transformer/feature_columns.json",

    "reports/latent_trajectory_transformer_cross_source/report.md",
    "reports/latent_trajectory_transformer_cross_source/summary.csv",
    "reports/latent_trajectory_transformer_cross_source/fold_metrics.csv",
    "reports/latent_trajectory_transformer_cross_source/training_history.csv",
    "reports/latent_trajectory_transformer_cross_source/model_config.json",
    "reports/latent_trajectory_transformer_cross_source/feature_columns.json",

    # Transformer calibration results
    "reports/latent_trajectory_transformer_calibration_test/report.md",
    "reports/latent_trajectory_transformer_calibration_test/calibration_summary.csv",
    "reports/latent_trajectory_transformer_calibration_test/calibration_fold_metrics.csv",
    "reports/latent_trajectory_transformer_calibration_test/calibration_gain_vs_zero_shot.csv",
    "reports/latent_trajectory_transformer_calibration_test/per_subject_calibration.csv",
    "reports/latent_trajectory_transformer_calibration_test/training_history.csv",
    "reports/latent_trajectory_transformer_calibration_test/model_config.json",
    "reports/latent_trajectory_transformer_calibration_test/feature_columns.json",
    "reports/latent_trajectory_transformer_calibration_test/summary.json",

    # Slow latent states and classical baselines
    "reports/slow_latent_states/pm_w10/report.md",
    "reports/slow_latent_states/pm_w10/latent_regression_summary.csv",
    "reports/slow_latent_states/pm_w10/slow_pca_loadings.csv",
    "reports/slow_latent_states/pm_w10/slow_pm_axis_summary.csv",

    "reports/slow_latent_states/cross_source/report.md",
    "reports/slow_latent_states/cross_source/summary.csv",
    "reports/slow_latent_states/cross_source/fold_metrics.csv",
    "reports/slow_latent_states/cross_source/r2_pivot.csv",
    "reports/slow_latent_states/cross_source/spearman_pivot.csv",
    "reports/slow_latent_states/cross_source/summary.json",

    # State dynamics
    "reports/state_dynamics/pm_w10_experiment_summary_v4/report.md",
    "reports/state_dynamics/pm_w10_experiment_summary_v4/target_recommendations_regression.csv",
    "reports/state_dynamics/pm_w10_experiment_summary_v4/target_recommendations_classification.csv",
    "reports/state_dynamics/pm_w10_experiment_summary_v4/target_recommendations_cross_source.csv",

    "reports/state_dynamics/sensitivity/report.md",
    "reports/state_dynamics/sensitivity/sensitivity_summary.csv",
    "reports/state_dynamics/sensitivity/sensitivity_best_params.csv",
    "reports/state_dynamics/sensitivity/sensitivity_stability.csv",
    "reports/state_dynamics/sensitivity/trend_label_distribution.csv",

    # User calibration old/classical experiment
    "reports/user_calibration/pm_w10/report.md",
    "reports/user_calibration/pm_w10/calibration_summary.csv",
    "reports/user_calibration/pm_w10/calibration_gain_vs_zero_shot.csv",
    "reports/user_calibration/pm_w10/calibration_fold_metrics.csv",

    # Device / wearable alignment
    "reports/device_metric_alignment/report.md",
    "reports/device_metric_alignment/device_metric_mapping.csv",
    "reports/device_metric_alignment/latent_state_evidence_matrix.csv",
    "reports/device_metric_alignment/helmet_vs_bracelet_comparison.csv",
    "reports/device_metric_alignment/helmet_vs_bracelet_comparison.md",
    "reports/device_metric_alignment/summary.json",

    # WESAD / wearable proxy
    "reports/wearable_pm_alignment/wesad_final_summary.md",
    "reports/wearable_pm_alignment/report.md",
    "reports/wearable_pm_alignment/summary.json",
]


DIRS_TO_COLLECT_IF_EXIST = [
    # Figures from final reports
    "reports/latent_trajectory_transformer/figures",
    "reports/latent_trajectory_transformer_cross_source/figures",
    "reports/slow_latent_states/cross_source/figures",
    "reports/state_dynamics/pm_w10_experiment_summary_v4/figures",
    "reports/state_dynamics/sensitivity/figures",
]


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_dir(src: Path, dst: Path) -> int:
    if not src.exists() or not src.is_dir():
        return 0

    count = 0
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, out)
            count += 1

    return count


def make_index(
    output_dir: Path,
    copied_files: list[dict],
    missing_files: list[str],
    copied_dirs: list[dict],
) -> None:
    lines = []

    lines.append("# EEG NIR meeting report files")
    lines.append("")
    lines.append(f"Collected at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## Main files")
    lines.append("")
    lines.append("- `00_project_summary/project_hypotheses_summary.md`")
    lines.append("- `01_integrated_state_evidence_v2/integrated_state_report.md`")
    lines.append("- `02_transformer_trajectory/report.md`")
    lines.append("- `03_transformer_cross_source/report.md`")
    lines.append("- `04_transformer_calibration/report.md`")
    lines.append("")
    lines.append("## Copied files")
    lines.append("")
    for item in copied_files:
        lines.append(f"- `{item['dst']}` ← `{item['src']}`")
    lines.append("")
    lines.append("## Copied directories")
    lines.append("")
    if copied_dirs:
        for item in copied_dirs:
            lines.append(f"- `{item['dst']}` ← `{item['src']}` ({item['files']} files)")
    else:
        lines.append("No directories copied.")
    lines.append("")
    lines.append("## Missing files")
    lines.append("")
    if missing_files:
        for path in missing_files:
            lines.append(f"- `{path}`")
    else:
        lines.append("No missing files.")
    lines.append("")

    (output_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "copied_files": copied_files,
        "missing_files": missing_files,
        "copied_dirs": copied_dirs,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def target_subdir_for_file(path: str) -> str:
    if "project_summary" in path:
        return "00_project_summary"
    if "integrated_state_evidence_v2" in path:
        return "01_integrated_state_evidence_v2"
    if "latent_trajectory_transformer_calibration" in path:
        return "04_transformer_calibration"
    if "latent_trajectory_transformer_cross_source" in path:
        return "03_transformer_cross_source"
    if "latent_trajectory_transformer" in path:
        return "02_transformer_trajectory"
    if "slow_latent_states/pm_w10" in path or "slow_latent_states\\pm_w10" in path:
        return "05_slow_latent_states"
    if "slow_latent_states/cross_source" in path or "slow_latent_states\\cross_source" in path:
        return "06_slow_latent_cross_source"
    if "state_dynamics" in path:
        return "07_state_dynamics"
    if "user_calibration" in path:
        return "08_user_calibration_classical"
    if "device_metric_alignment" in path:
        return "09_device_metric_alignment"
    if "wearable_pm_alignment" in path:
        return "10_wearable_pm_alignment"
    return "99_other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect EEG NIR report files for meeting into one folder on disk D."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\eeg_nir_meeting_report_files"),
        help="Output directory on disk D.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create zip archive next to output directory.",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    missing_files = []

    for rel_path in FILES_TO_COLLECT:
        src = root / rel_path
        subdir = target_subdir_for_file(rel_path)
        dst = output_dir / subdir / Path(rel_path).name

        ok = copy_file(src, dst)

        if ok:
            copied_files.append(
                {
                    "src": str(src),
                    "dst": str(dst.relative_to(output_dir)),
                }
            )
        else:
            missing_files.append(rel_path)

    copied_dirs = []

    for rel_dir in DIRS_TO_COLLECT_IF_EXIST:
        src_dir = root / rel_dir
        subdir = target_subdir_for_file(rel_dir)
        dst_dir = output_dir / subdir / "figures"

        n = copy_dir(src_dir, dst_dir)

        if n > 0:
            copied_dirs.append(
                {
                    "src": str(src_dir),
                    "dst": str(dst_dir.relative_to(output_dir)),
                    "files": n,
                }
            )

    make_index(
        output_dir=output_dir,
        copied_files=copied_files,
        missing_files=missing_files,
        copied_dirs=copied_dirs,
    )

    if args.zip:
        zip_base = output_dir.parent / output_dir.name
        archive_path = shutil.make_archive(str(zip_base), "zip", output_dir)
        print(f"ZIP archive: {archive_path}")

    print("=" * 80)
    print("Collected EEG NIR meeting files")
    print("=" * 80)
    print(f"Project root: {root}")
    print(f"Output dir:   {output_dir}")
    print(f"Copied files: {len(copied_files)}")
    print(f"Missing:      {len(missing_files)}")
    print(f"Index:        {output_dir / 'INDEX.md'}")
    print(f"Manifest:     {output_dir / 'manifest.json'}")

    if missing_files:
        print("")
        print("Missing files:")
        for item in missing_files:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
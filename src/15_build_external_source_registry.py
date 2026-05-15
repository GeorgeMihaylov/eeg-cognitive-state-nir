#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build external source registry for wearable / eye-tracking / multimodal PM alignment.

Outputs:
  reports/wearable_pm_alignment/source_registry.csv
  reports/wearable_pm_alignment/source_registry.md
  reports/wearable_pm_alignment/source_registry.json
  reports/wearable_pm_alignment/pm_external_signal_mapping.csv
  reports/wearable_pm_alignment/pm_external_signal_mapping.md
  reports/wearable_pm_alignment/report.md

Run from project root:
  D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\15_build_external_source_registry.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ExternalSource:
    source_id: str
    name: str
    source_type: str
    modality: str
    device: str
    signals: str
    n_subjects: Optional[str]
    protocol: str
    labels: str
    pm_relevance: str
    priority: int
    recommended_stage: str
    access_url: str
    paper_url: str
    license_notes: str
    access_notes: str
    project_use: str
    risks: str


def build_sources() -> list[ExternalSource]:
    return [
        ExternalSource(
            source_id="wesad",
            name="WESAD: Wearable Stress and Affect Detection",
            source_type="dataset",
            modality="wearable_physiology",
            device="Empatica E4 wrist device + chest sensors",
            signals="BVP; ECG; EDA; EMG; respiration; body temperature; three-axis acceleration",
            n_subjects="15",
            protocol="Stress and affect elicitation with baseline and stress/affective conditions",
            labels="stress; amusement; meditation/baseline-like conditions; subject annotations",
            pm_relevance="Stress; Excitement; Relaxation",
            priority=1,
            recommended_stage="first_wearable_baseline",
            access_url="https://archive.ics.uci.edu/ml/datasets/WESAD%2B%28Wearable%2BStress%2Band%2BAffect%2BDetection%29",
            paper_url="https://ubi29.informatik.uni-siegen.de/usi/pdf/ubi_icmi2018.pdf",
            license_notes="Check UCI and original dataset terms before redistribution",
            access_notes="Public dataset page; Kaggle mirrors exist but primary source is preferable",
            project_use="Build first wearable stress baseline; implement EDA/BVP/TEMP/ACC feature extraction; subject-wise validation",
            risks="Small number of subjects; labels are not Emotiv PM metrics; chest signals may not match wrist-only setup",
        ),
        ExternalSource(
            source_id="colet",
            name="COLET: Cognitive Load Eye-Tracking Dataset",
            source_type="dataset",
            modality="eye_tracking",
            device="Eye-tracker",
            signals="gaze; fixations; saccades; pupil-related features depending on release",
            n_subjects="47",
            protocol="Visual search puzzles with different task difficulty and duration",
            labels="cognitive workload / task difficulty",
            pm_relevance="Attention; Focus; Engagement; Interest",
            priority=2,
            recommended_stage="first_eye_tracking_baseline",
            access_url="https://www.sciencedirect.com/science/article/pii/S0169260722003716",
            paper_url="https://pubmed.ncbi.nlm.nih.gov/36122625/",
            license_notes="Check dataset access terms from the article/data supplement",
            access_notes="Dataset access may require following article links or contacting authors",
            project_use="Build first eye-tracking workload baseline; map gaze/pupil features to attention/focus-like states",
            risks="Not PM-labeled; access may be less direct than WESAD; task is visual-search-specific",
        ),
        ExternalSource(
            source_id="universe",
            name="UNIVERSE: Unobtrusive measurement of mental workload and stress",
            source_type="dataset",
            modality="wearable_eeg_plus_physiology",
            device="Muse S EEG headband + Empatica E4 wristband",
            signals="wearable EEG; EDA; PPG; ACC; TEMP",
            n_subjects="24",
            protocol="Long-duration cognitive load and stress elicitation paradigm",
            labels="mental workload; stress-related protocol labels and annotations",
            pm_relevance="Stress; Focus; Attention; Engagement; Relaxation",
            priority=3,
            recommended_stage="first_multimodal_fusion_dataset",
            access_url="https://zenodo.org/records/10371068",
            paper_url="https://www.nature.com/articles/s41597-024-03738-7",
            license_notes="Check Zenodo license for reuse and redistribution",
            access_notes="Public Zenodo record",
            project_use="Study EEG-only vs wearable-only vs EEG+wearable; closest external source to multimodal PM alignment",
            risks="Larger and more complex than WESAD; labels differ from Emotiv PM metrics",
        ),
        ExternalSource(
            source_id="catsa",
            name="CATSA: Cognitive Attention and Task-Based Stress Assessment",
            source_type="dataset",
            modality="wearable_physiology",
            device="Empatica E4",
            signals="BVP; EDA; three-axis acceleration; average HR",
            n_subjects="50",
            protocol="Baseline and induced cognitive tasks for cognitive attention and task-based stress",
            labels="task/baseline conditions; stress/cognitive attention context",
            pm_relevance="Stress; Attention; Focus; Engagement",
            priority=4,
            recommended_stage="second_wearable_cognitive_stress_dataset",
            access_url="https://data.mendeley.com/datasets/jwbz4ggws5",
            paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12686876/",
            license_notes="Check Mendeley Data license",
            access_notes="Public Mendeley Data page",
            project_use="Validate wearable features for cognitive stress and attention-like tasks after WESAD baseline",
            risks="Newer dataset; task labels require careful interpretation",
        ),
        ExternalSource(
            source_id="physionet_induced_stress",
            name="Wearable Device Dataset from Induced Stress and Structured Exercise Sessions",
            source_type="dataset",
            modality="wearable_physiology",
            device="Empatica E4",
            signals="BVP; activity/motion; skin temperature; EDA",
            n_subjects="not_fixed_in_registry",
            protocol="Math tasks; emotional tasks; rest periods; aerobic and anaerobic exercise",
            labels="stress tasks; rest; exercise conditions",
            pm_relevance="Stress; Excitement; Relaxation",
            priority=5,
            recommended_stage="movement_confounding_control",
            access_url="https://physionet.org/content/wearable-device-dataset/",
            paper_url="https://www.nature.com/articles/s41597-025-04845-9",
            license_notes="Check PhysioNet credentialing/license requirements",
            access_notes="PhysioNet access may require account and credentialing depending on dataset",
            project_use="Separate cognitive/emotional stress from physical activity; test ACC-based confounding control",
            risks="Exercise confounds; not PM-labeled; protocol differs from office/learning setting",
        ),
        ExternalSource(
            source_id="empatica_e4_stress",
            name="EmpaticaE4Stress / PPG and EDA dataset",
            source_type="dataset",
            modality="wearable_physiology",
            device="Empatica E4",
            signals="BVP; EDA; TEMP; HR; IBI; ACC",
            n_subjects="29",
            protocol="Stress/anxiety/distress-related physiological data collection",
            labels="stress/anxiety/distress-related labels depending on dataset files",
            pm_relevance="Stress; Excitement; Relaxation",
            priority=6,
            recommended_stage="additional_empatica_stress_validation",
            access_url="https://data.mendeley.com/datasets/kb42z77m2g/2",
            paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10847510/",
            license_notes="Check Mendeley Data license",
            access_notes="Public Mendeley Data page",
            project_use="Compact Empatica E4 stress/anxiety baseline; compare with WESAD-style features",
            risks="Protocol/labels need inspection; not PM-labeled",
        ),
        ExternalSource(
            source_id="swell_kw",
            name="SWELL-KW: Knowledge Work Dataset",
            source_type="dataset",
            modality="workload_stress_context",
            device="Physiological and computer interaction signals depending on subset",
            signals="HR/HRV-derived data in common derived versions; computer interaction; self-report stress",
            n_subjects="25",
            protocol="Knowledge work tasks with time pressure and email interruptions",
            labels="stress; workload/task conditions; self-report",
            pm_relevance="Stress; Engagement; Focus; Attention",
            priority=7,
            recommended_stage="knowledge_work_protocol_reference",
            access_url="https://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html",
            paper_url="https://www.researchgate.net/publication/265300675_The_SWELL_Knowledge_Work_Dataset_for_Stress_and_User_Modeling_Research",
            license_notes="Check original dataset terms and derived dataset licenses",
            access_notes="Original dataset page and derived Kaggle HRV version available",
            project_use="Reference protocol for office-like cognitive workload and interruptions",
            risks="Signals differ by subset; raw wearable data may be limited; not PM-labeled",
        ),
        ExternalSource(
            source_id="epistress",
            name="EPIStress",
            source_type="dataset",
            modality="wearable_eeg_plus_physiology",
            device="Wearable EEG + peripheral wearable sensors",
            signals="wearable EEG; PPG; ACC; EDA; TEMP",
            n_subjects="20 epilepsy patients",
            protocol="Cognitive stress elicitation in clinical population",
            labels="stress-related protocol labels",
            pm_relevance="Stress; Excitement; Relaxation",
            priority=8,
            recommended_stage="clinical_multimodal_reference",
            access_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12658113/",
            paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12658113/",
            license_notes="Check article/dataset access notes",
            access_notes="Dataset access must be checked from article",
            project_use="Reference multimodal stress protocol; not first practical dataset due to clinical population",
            risks="Clinical epilepsy population; limited transfer to healthy users",
        ),
        ExternalSource(
            source_id="affective_road",
            name="AffectiveROAD",
            source_type="dataset",
            modality="wearable_physiology_contextual",
            device="Empatica E4 and driving-context sensors depending on version",
            signals="EDA; BVP/HR; ACC; contextual driver-state data",
            n_subjects="not_fixed_in_registry",
            protocol="Real-world or simulator driving stress/attention scenario",
            labels="driver stress/attention-related annotations",
            pm_relevance="Stress; Excitement; Attention",
            priority=9,
            recommended_stage="real_world_stress_reference",
            access_url="https://www.empatica.com/blog/using-the-e4-to-measure-stress-intensity-through-machine-learning-methods/",
            paper_url="https://www.researchgate.net/publication/324594898_AffectiveROAD_System_and_Database_to_Assess_Driver%27s_Attention",
            license_notes="Check dataset availability and paper terms",
            access_notes="Access path may require following paper/dataset instructions",
            project_use="Reference for real-world stress and movement/context confounding",
            risks="Driving-specific; may not map to office/learning PM states",
        ),
        ExternalSource(
            source_id="empathic_school",
            name="EmpathicSchool",
            source_type="dataset",
            modality="wearable_physiology_plus_video",
            device="Empatica E4 + facial/video features",
            signals="EDA; HR/PPG-derived features; skin temperature; ACC; facial/video features",
            n_subjects="not_fixed_in_registry",
            protocol="Student tasks including presentation preparation and exam-like settings",
            labels="emotion/stress-related labels",
            pm_relevance="Stress; Engagement; Excitement",
            priority=10,
            recommended_stage="education_context_reference",
            access_url="https://www.nature.com/articles/s41597-025-05812-0",
            paper_url="https://www.nature.com/articles/s41597-025-05812-0",
            license_notes="Check dataset availability and license from Scientific Data record",
            access_notes="Dataset access should be checked from article",
            project_use="Reference for education stress and engagement scenario",
            risks="May include video/facial data privacy constraints; labels may not align with PM",
        ),
        ExternalSource(
            source_id="gazebase",
            name="GazeBase",
            source_type="dataset",
            modality="eye_tracking",
            device="Eye-tracker",
            signals="fixations; saccades; gaze trajectories; eye movement task data",
            n_subjects="large-scale",
            protocol="Multiple eye movement tasks",
            labels="task/session identity; no direct workload PM labels",
            pm_relevance="Feature extraction support; representation learning; auxiliary eye movement modeling",
            priority=11,
            recommended_stage="auxiliary_eye_tracking_reference",
            access_url="https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257",
            paper_url="https://www.nature.com/articles/s41597-021-00959-y",
            license_notes="Check Figshare license",
            access_notes="Public Figshare repository",
            project_use="Auxiliary source for eye-movement feature extraction and potential pretraining",
            risks="No direct cognitive workload labels; less directly aligned with PM",
        ),
        ExternalSource(
            source_id="gazeload",
            name="GAZELOAD",
            source_type="dataset_or_preprint",
            modality="eye_tracking",
            device="Meta Aria smart glasses",
            signals="gaze data; pupil-related signals; smart-glasses task context",
            n_subjects="26",
            protocol="Industrial human-robot collaboration tasks with mental workload estimation",
            labels="mental workload/task context",
            pm_relevance="Focus; Attention; Stress; Engagement",
            priority=12,
            recommended_stage="smart_glasses_workload_reference",
            access_url="https://arxiv.org/html/2601.21829v1",
            paper_url="https://arxiv.org/html/2601.21829v1",
            license_notes="Preprint/source access should be checked",
            access_notes="Check whether dataset is publicly released from preprint links",
            project_use="Reference for smart-glasses workload protocol and gaze features",
            risks="Preprint/new source; dataset availability may be limited",
        ),
        ExternalSource(
            source_id="pupil_microsaccades_cognitive_load",
            name="Pupil diameter and microsaccades as indicators of cognitive load",
            source_type="paper",
            modality="eye_tracking",
            device="Eye-tracker",
            signals="pupil diameter; microsaccades; gaze-related measurements",
            n_subjects="paper_dependent",
            protocol="Cognitive load experiments",
            labels="cognitive load/task demand",
            pm_relevance="Focus; Attention; Stress/arousal control",
            priority=13,
            recommended_stage="eye_tracking_feature_methodology",
            access_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC6138399/",
            paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC6138399/",
            license_notes="Open PMC article",
            access_notes="Article only, not necessarily dataset",
            project_use="Methodological support for pupil and microsaccade features",
            risks="Feature interpretation may be task-dependent; pupil is confounded by illumination and arousal",
        ),
        ExternalSource(
            source_id="wearable_mental_workload_stress_emotion",
            name="Wearable Technologies for Mental Workload, Stress, and Emotional State",
            source_type="paper",
            modality="wearable_eeg_plus_physiology",
            device="Empatica E4 + Muse 2",
            signals="wrist physiology; wearable EEG",
            n_subjects="paper_dependent",
            protocol="Mental workload, stress and emotional state assessment",
            labels="workload/stress/emotional state",
            pm_relevance="Stress; Focus; Attention; Engagement; Relaxation",
            priority=14,
            recommended_stage="methodology_and_protocol_reference",
            access_url="https://www.mdpi.com/1424-8220/21/7/2332",
            paper_url="https://pubmed.ncbi.nlm.nih.gov/33810613/",
            license_notes="Open-access article",
            access_notes="Article only unless data links are provided",
            project_use="Reference for consumer-grade Muse + Empatica experimental design",
            risks="May not provide directly reusable data; protocol must be adapted",
        ),
    ]


def build_pm_mapping() -> list[dict[str, str]]:
    return [
        {
            "pm_metric": "Stress",
            "wearable_physiology": "EDA increase; HR increase; HRV decrease; BVP changes; TEMP trend; ACC for movement control",
            "eye_tracking": "pupil dilation; blink/gaze instability; reduced fixation stability",
            "expected_feasibility": "high",
            "first_datasets": "WESAD; PhysioNet Induced Stress; CATSA",
            "notes": "Most promising wearable-only target.",
        },
        {
            "pm_metric": "Excitement",
            "wearable_physiology": "EDA increase; HR increase; BVP amplitude/variability",
            "eye_tracking": "pupil dilation; gaze variability",
            "expected_feasibility": "high_to_medium",
            "first_datasets": "WESAD; EmpaticaE4Stress",
            "notes": "Likely arousal-driven; control against physical movement.",
        },
        {
            "pm_metric": "Relaxation",
            "wearable_physiology": "HRV increase; EDA decrease; HR decrease; TEMP trend",
            "eye_tracking": "blink/gaze stability; reduced pupil arousal response",
            "expected_feasibility": "high_to_medium",
            "first_datasets": "WESAD; UNIVERSE",
            "notes": "May require longer windows for robust HRV.",
        },
        {
            "pm_metric": "Engagement",
            "wearable_physiology": "EDA dynamics; HR dynamics; task context",
            "eye_tracking": "AOI dwell time; fixation pattern; AOI transitions; gaze entropy",
            "expected_feasibility": "medium_to_high",
            "first_datasets": "CATSA; SWELL-KW; COLET",
            "notes": "Likely benefits from task/context markers.",
        },
        {
            "pm_metric": "Focus",
            "wearable_physiology": "HRV; low movement; optional EEG",
            "eye_tracking": "fixation stability; pupil dynamics; gaze entropy; reduced distraction",
            "expected_feasibility": "high_with_eye_tracking",
            "first_datasets": "COLET; UNIVERSE; GAZELOAD",
            "notes": "Eye-tracking is probably stronger than wrist-only physiology.",
        },
        {
            "pm_metric": "Attention",
            "wearable_physiology": "weak without EEG/task context; HRV may help indirectly",
            "eye_tracking": "fixations; saccades; gaze entropy; pupil dynamics; AOI dwell time",
            "expected_feasibility": "high_with_eye_tracking",
            "first_datasets": "COLET; CATSA; UNIVERSE",
            "notes": "Eye-tracking and wearable EEG are the most relevant sources.",
        },
        {
            "pm_metric": "Interest",
            "wearable_physiology": "arousal proxy through EDA/HR",
            "eye_tracking": "dwell time; AOI transitions; gaze preference; gaze entropy",
            "expected_feasibility": "medium",
            "first_datasets": "COLET; eye-tracking datasets with AOI/task-interest labels",
            "notes": "Likely task-dependent and harder to validate.",
        },
    ]


def sources_to_dataframe(sources: list[ExternalSource]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(s) for s in sources])
    return df.sort_values(["priority", "source_id"]).reset_index(drop=True)


def build_source_registry_md(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Реестр внешних источников данных для wearable / eye-tracking / PM alignment")
    lines.append("")
    lines.append("## Назначение")
    lines.append("")
    lines.append(
        "Реестр фиксирует датасеты, статьи и устройства, релевантные для сопоставления PM-метрик Emotiv "
        "с внешними физиологическими и поведенческими сигналами."
    )
    lines.append("")
    lines.append("## Приоритетные источники")
    lines.append("")
    short_cols = [
        "priority",
        "source_id",
        "name",
        "modality",
        "signals",
        "n_subjects",
        "pm_relevance",
        "recommended_stage",
        "access_url",
        "paper_url",
    ]
    lines.append(df[short_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Детальное описание источников")
    lines.append("")
    for _, row in df.iterrows():
        lines.append(f"## {row['priority']}. {row['name']}")
        lines.append("")
        fields = [
            ("source_id", f"`{row['source_id']}`"),
            ("type", f"`{row['source_type']}`"),
            ("modality", f"`{row['modality']}`"),
            ("device", row["device"]),
            ("signals", row["signals"]),
            ("n_subjects", row["n_subjects"]),
            ("protocol", row["protocol"]),
            ("labels", row["labels"]),
            ("PM relevance", row["pm_relevance"]),
            ("recommended stage", f"`{row['recommended_stage']}`"),
            ("access", row["access_url"]),
            ("paper", row["paper_url"]),
            ("license notes", row["license_notes"]),
            ("access notes", row["access_notes"]),
            ("project use", row["project_use"]),
            ("risks", row["risks"]),
        ]
        for key, value in fields:
            lines.append(f"- **{key}:** {value}")
        lines.append("")
    return "\n".join(lines)


def build_pm_mapping_md(mapping_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Mapping PM-метрик к внешним wearable и eye-tracking сигналам")
    lines.append("")
    lines.append("## Сводная таблица")
    lines.append("")
    lines.append(mapping_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Практический порядок проверки")
    lines.append("")
    lines.append("1. `Stress`, `Excitement`, `Relaxation`: начать с wearable physiology на WESAD.")
    lines.append("2. `Attention`, `Focus`, `Engagement`: начать с eye-tracking на COLET.")
    lines.append("3. Multimodal comparison: перейти к UNIVERSE после первых двух baseline.")
    lines.append("")
    lines.append("## Важные временные масштабы")
    lines.append("")
    lines.append("| Сигнал | Рекомендуемые окна | Комментарий |")
    lines.append("|---|---:|---|")
    lines.append("| EEG / PM | 10 s | совместимо с текущим EEG pipeline |")
    lines.append("| EDA | 10-60 s | возможен лаг 5-30 s |")
    lines.append("| HRV | 30-120 s | 10 s часто мало для устойчивых HRV-признаков |")
    lines.append("| Eye-tracking | 1-30 s | зависит от задачи и AOI-разметки |")
    lines.append("| ACC | 10-60 s | контроль движения и артефактов |")
    lines.append("")
    return "\n".join(lines)


def build_report_md(df: pd.DataFrame, mapping_df: pd.DataFrame, created_at: str) -> str:
    lines: list[str] = []
    lines.append("# External source registry report")
    lines.append("")
    lines.append(f"Created at: `{created_at}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total sources: **{len(df)}**")
    lines.append(f"- Dataset sources: **{int((df['source_type'] == 'dataset').sum())}**")
    lines.append(f"- Paper/methodology sources: **{int((df['source_type'] != 'dataset').sum())}**")
    lines.append("")
    lines.append("## Sources by modality")
    lines.append("")
    modality_counts = df["modality"].value_counts().reset_index()
    modality_counts.columns = ["modality", "count"]
    lines.append(modality_counts.to_markdown(index=False))
    lines.append("")
    lines.append("## Recommended first datasets")
    lines.append("")
    first = df[df["priority"].isin([1, 2, 3, 4, 5])].copy()
    cols = ["priority", "source_id", "name", "modality", "project_use"]
    lines.append(first[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## PM mapping")
    lines.append("")
    lines.append(mapping_df[["pm_metric", "expected_feasibility", "first_datasets"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Recommended next scripts")
    lines.append("")
    lines.append("```text")
    lines.append("src/16_prepare_wesad_dataset.py")
    lines.append("src/17_train_wesad_stress_baseline.py")
    lines.append("src/18_prepare_colet_dataset.py")
    lines.append("src/19_train_colet_workload_baseline.py")
    lines.append("```")
    lines.append("")
    lines.append("## Recommended practical order")
    lines.append("")
    lines.append("1. WESAD: wearable stress baseline.")
    lines.append("2. COLET: eye-tracking workload/attention baseline.")
    lines.append("3. UNIVERSE: EEG + wearable multimodal comparison.")
    lines.append("4. CATSA / PhysioNet: validation and stress-vs-motion checks.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external source registry for wearable and eye-tracking PM alignment.")
    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument("--output-dir", type=str, default="reports/wearable_pm_alignment", help="Output directory.")
    parser.add_argument("--priority-threshold", type=int, default=0, help="Also create priority subset if > 0.")
    return parser.parse_args()


def resolve_output_dir(root: Path, output_dir: str) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return root / path


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = resolve_output_dir(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = build_sources()
    df = sources_to_dataframe(sources)
    mapping_df = pd.DataFrame(build_pm_mapping())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_path = output_dir / "source_registry.csv"
    json_path = output_dir / "source_registry.json"
    md_path = output_dir / "source_registry.md"
    mapping_csv_path = output_dir / "pm_external_signal_mapping.csv"
    mapping_md_path = output_dir / "pm_external_signal_mapping.md"
    report_path = output_dir / "report.md"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps([asdict(s) for s in sources], ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_source_registry_md(df), encoding="utf-8")

    mapping_df.to_csv(mapping_csv_path, index=False, encoding="utf-8")
    mapping_md_path.write_text(build_pm_mapping_md(mapping_df), encoding="utf-8")
    report_path.write_text(build_report_md(df, mapping_df, created_at), encoding="utf-8")

    if args.priority_threshold and args.priority_threshold > 0:
        priority_df = df[df["priority"] <= args.priority_threshold].copy()
        pr_csv = output_dir / f"source_registry_priority_le_{args.priority_threshold}.csv"
        pr_md = output_dir / f"source_registry_priority_le_{args.priority_threshold}.md"
        priority_df.to_csv(pr_csv, index=False, encoding="utf-8")
        pr_md.write_text(build_source_registry_md(priority_df), encoding="utf-8")

    print("=" * 80)
    print("External source registry built")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Output dir: {output_dir}")
    print(f"Sources: {len(df)}")
    print("")
    print("Saved:")
    for path in [csv_path, json_path, md_path, mapping_csv_path, mapping_md_path, report_path]:
        print(f"  {path}")
    print("")
    print("Recommended first datasets:")
    first_cols = ["priority", "source_id", "name", "modality", "recommended_stage"]
    print(df[df["priority"] <= 5][first_cols].to_string(index=False))


if __name__ == "__main__":
    main()

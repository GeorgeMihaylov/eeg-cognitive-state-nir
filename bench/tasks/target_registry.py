"""Canonical executable target registry and legacy configuration bridge."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

from .target_spec import TargetSpec


PM_METRICS = (
    "attention",
    "engagement",
    "excitement",
    "stress",
    "relaxation",
    "interest",
    "focus",
)
PM_TARGET_COLUMNS = tuple(f"target_{metric}" for metric in PM_METRICS)
FEATURE_INPUTS = ("eeg", "pow", "eeg_pow")
REGRESSION_METRICS = ("mae", "rmse", "r2", "spearman")
CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "kappa",
)


class LegacyTargetConfigWarning(UserWarning):
    """Signals that a legacy target name was resolved explicitly."""


def _continuous_spec(metric: str) -> TargetSpec:
    return TargetSpec(
        target_id=f"pm_{metric}_regression",
        display_name=f"PM {metric.title()} regression",
        target_type="continuous_regression",
        processed_columns=(f"target_{metric}",),
        output_names=(metric,),
        output_dim=1,
        missing_value_policy="drop_rows_with_missing_target",
        cohort_policy="target_complete_cases_inside_fixed_outer_subject_folds",
        transform_policy="identity",
        fit_scope="none",
        recommended_metrics=REGRESSION_METRICS,
        allowed_feature_inputs=FEATURE_INPUTS,
        raw_input_supported=True,
        execution_status="executable",
        registry_status="canonical",
    )


def _candidate_specs() -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    for metric in PM_METRICS:
        specs.append(
            TargetSpec(
                target_id=f"pm_{metric}_active_proxy",
                display_name=f"PM {metric.title()} active proxy",
                target_type="binary_proxy",
                processed_columns=(f"PM.{metric.title()}.IsActive__mean",),
                output_names=(f"{metric}_active",),
                output_dim=1,
                missing_value_policy="drop_rows_with_missing_target",
                cohort_policy="target_complete_cases_inside_fixed_outer_subject_folds",
                transform_policy="identity_binary_proxy_pending_semantic_validation",
                fit_scope="none",
                recommended_metrics=(
                    "balanced_accuracy",
                    "macro_f1",
                    "average_precision",
                ),
                allowed_feature_inputs=FEATURE_INPUTS,
                raw_input_supported=False,
                execution_status="disabled",
                registry_status="requires_semantic_validation",
            )
        )
        for q in (3, 5):
            specs.append(
                TargetSpec(
                    target_id=f"pm_{metric}_q{q}_fold_local",
                    display_name=f"PM {metric.title()} fold-local Q{q}",
                    target_type="derived_ordinal_classification",
                    processed_columns=(f"target_{metric}",),
                    output_names=(f"{metric}_q{q}",),
                    output_dim=1,
                    missing_value_policy="drop_rows_with_missing_target",
                    cohort_policy=(
                        "target_complete_cases_inside_fixed_outer_subject_folds"
                    ),
                    transform_policy=f"fold_local_quantile_q{q}",
                    fit_scope="outer_train_only",
                    recommended_metrics=CLASSIFICATION_METRICS
                    + ("ordinal_mae", "adjacent_accuracy"),
                    allowed_feature_inputs=FEATURE_INPUTS,
                    raw_input_supported=True,
                    execution_status="disabled",
                    registry_status="registered_candidate",
                )
            )
    specs.extend(
        [
            TargetSpec(
                target_id="pm_activity_multilabel_7",
                display_name="Seven PM activity proxies",
                target_type="multilabel_proxy",
                processed_columns=tuple(
                    f"PM.{metric.title()}.IsActive__mean" for metric in PM_METRICS
                ),
                output_names=tuple(f"{metric}_active" for metric in PM_METRICS),
                output_dim=7,
                missing_value_policy="drop_rows_unless_all_outputs_present",
                cohort_policy="complete_cases_inside_fixed_outer_subject_folds",
                transform_policy="identity_multilabel_pending_semantic_validation",
                fit_scope="none",
                recommended_metrics=("macro_f1", "average_precision"),
                allowed_feature_inputs=FEATURE_INPUTS,
                raw_input_supported=False,
                execution_status="disabled",
                registry_status="requires_semantic_validation",
            ),
            TargetSpec(
                target_id="pm_long_term_excitement_regression",
                display_name="PM long-term excitement regression",
                target_type="continuous_regression",
                processed_columns=("target_long_term_excitement",),
                output_names=("long_term_excitement",),
                output_dim=1,
                missing_value_policy="unavailable_in_processed_dataset",
                cohort_policy="no_executable_cohort",
                transform_policy="identity",
                fit_scope="none",
                recommended_metrics=REGRESSION_METRICS,
                allowed_feature_inputs=FEATURE_INPUTS,
                raw_input_supported=False,
                execution_status="disabled",
                registry_status="requires_processed_target_materialization",
            ),
        ]
    )
    return specs


_SPECS = [
    *(_continuous_spec(metric) for metric in PM_METRICS),
    TargetSpec(
        target_id="pm_multioutput_regression_7",
        display_name="Seven-output PM regression",
        target_type="multioutput_regression",
        processed_columns=PM_TARGET_COLUMNS,
        output_names=PM_METRICS,
        output_dim=7,
        missing_value_policy="drop_rows_unless_all_seven_targets_present",
        cohort_policy="complete_cases_inside_fixed_outer_subject_folds",
        transform_policy="identity",
        fit_scope="none",
        recommended_metrics=REGRESSION_METRICS,
        allowed_feature_inputs=FEATURE_INPUTS,
        raw_input_supported=True,
        execution_status="executable",
        registry_status="canonical",
    ),
    TargetSpec(
        target_id="label_focus_q5_legacy",
        display_name="Legacy global focus quintile label",
        target_type="legacy_classification",
        processed_columns=("label_q5",),
        output_names=("focus_q5",),
        output_dim=1,
        missing_value_policy="drop_rows_with_missing_target",
        cohort_policy="target_complete_cases_inside_fixed_outer_subject_folds",
        transform_policy="precomputed_global_quantile_label_no_refit",
        fit_scope="legacy_global_pre_split",
        recommended_metrics=CLASSIFICATION_METRICS
        + ("auc", "ordinal_mae", "adjacent_accuracy", "severe_error_rate"),
        allowed_feature_inputs=FEATURE_INPUTS,
        raw_input_supported=True,
        execution_status="executable",
        registry_status="legacy_global_benchmark_label",
    ),
    *_candidate_specs(),
]

TARGET_REGISTRY: dict[str, TargetSpec] = {spec.target_id: spec for spec in _SPECS}
if len(TARGET_REGISTRY) != len(_SPECS):
    raise RuntimeError("Canonical target registry contains duplicate target IDs")

LEGACY_TARGET_ALIASES = {
    "label_q5": "label_focus_q5_legacy",
    "target_main": "pm_focus_regression",
    **{f"target_{metric}": f"pm_{metric}_regression" for metric in PM_METRICS},
}


def list_target_specs(*, executable_only: bool = False) -> tuple[TargetSpec, ...]:
    specs = tuple(TARGET_REGISTRY.values())
    if executable_only:
        specs = tuple(spec for spec in specs if spec.is_executable)
    return specs


def get_target_spec(target_id: str, *, require_executable: bool = True) -> TargetSpec:
    try:
        spec = TARGET_REGISTRY[str(target_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown target_id {target_id!r}. Available: {sorted(TARGET_REGISTRY)}"
        ) from exc
    if require_executable and not spec.is_executable:
        raise ValueError(
            f"Target {target_id!r} is registered but disabled "
            f"({spec.registry_status})"
        )
    return spec


def resolve_target_spec(
    config: Mapping[str, Any],
    *,
    default_target_id: str | None = None,
    allow_ad_hoc_legacy: bool = True,
) -> TargetSpec:
    """Resolve canonical ``target_id`` or warn while bridging legacy configs."""

    present = [
        name for name in ("target_id", "target_col", "target_cols") if name in config
    ]
    if "target_id" in present and len(present) != 1:
        raise ValueError("target_id cannot be combined with target_col or target_cols")
    if "target_col" in present and "target_cols" in present:
        raise ValueError(
            "Dataset config must define either 'target_col' or 'target_cols', not both"
        )
    if "target_id" in config:
        return get_target_spec(str(config["target_id"]))

    if "target_cols" in config:
        columns = tuple(str(value) for value in config["target_cols"])
        if columns == PM_TARGET_COLUMNS:
            return _warn_legacy(
                "target_cols", "pm_multioutput_regression_7"
            )
        if not allow_ad_hoc_legacy:
            raise ValueError(
                "Non-canonical target_cols are not supported; use an explicit target_id"
            )
        warnings.warn(
            "Legacy ad-hoc target_cols are deprecated; register an explicit target_id",
            LegacyTargetConfigWarning,
            stacklevel=2,
        )
        return TargetSpec(
            target_id="legacy_ad_hoc_multioutput",
            display_name="Legacy ad-hoc multi-output target",
            target_type="multioutput_regression",
            processed_columns=columns,
            output_names=columns,
            output_dim=len(columns),
            missing_value_policy="drop_rows_unless_all_outputs_present",
            cohort_policy="complete_cases_inside_fixed_outer_subject_folds",
            transform_policy="identity",
            fit_scope="none",
            recommended_metrics=REGRESSION_METRICS,
            allowed_feature_inputs=FEATURE_INPUTS,
            raw_input_supported=False,
            execution_status="executable",
            registry_status="deprecated_ad_hoc_legacy",
        )

    if "target_col" in config:
        target_col = str(config["target_col"])
        if target_col == "target_main" and bool(
            config.get("_legacy_implicit_target_main_classification", False)
        ):
            warnings.warn(
                "Legacy target_main classification is deprecated; new tasks must "
                "use an explicit target_id",
                LegacyTargetConfigWarning,
                stacklevel=2,
            )
            return TargetSpec(
                target_id="legacy_ad_hoc_target_main_classification",
                display_name="Legacy discretized target_main classification",
                target_type="legacy_classification",
                processed_columns=("target_main",),
                output_names=("target_main_class",),
                output_dim=1,
                missing_value_policy="drop_rows_with_missing_target",
                cohort_policy="target_complete_cases_inside_fixed_outer_subject_folds",
                transform_policy="legacy_loader_discretization",
                fit_scope="legacy_full_loaded_cohort",
                recommended_metrics=CLASSIFICATION_METRICS,
                allowed_feature_inputs=FEATURE_INPUTS,
                raw_input_supported=False,
                execution_status="executable",
                registry_status="deprecated_ad_hoc_legacy",
            )
        target_id = LEGACY_TARGET_ALIASES.get(target_col)
        if target_id is not None:
            return _warn_legacy(target_col, target_id)
        if not allow_ad_hoc_legacy:
            raise ValueError(
                f"Legacy target_col {target_col!r} has no canonical target_id"
            )
        warnings.warn(
            f"Legacy ad-hoc target_col {target_col!r} is deprecated; register target_id",
            LegacyTargetConfigWarning,
            stacklevel=2,
        )
        task_type = str(config.get("task_type", "classification"))
        is_regression = task_type == "regression"
        return TargetSpec(
            target_id=f"legacy_ad_hoc_{target_col}",
            display_name=f"Legacy ad-hoc target {target_col}",
            target_type=(
                "continuous_regression" if is_regression else "legacy_classification"
            ),
            processed_columns=(target_col,),
            output_names=(target_col,),
            output_dim=1,
            missing_value_policy="drop_rows_with_missing_target",
            cohort_policy="target_complete_cases_inside_fixed_outer_subject_folds",
            transform_policy="legacy_loader_behavior",
            fit_scope="legacy_config",
            recommended_metrics=(
                REGRESSION_METRICS if is_regression else CLASSIFICATION_METRICS
            ),
            allowed_feature_inputs=FEATURE_INPUTS,
            raw_input_supported=False,
            execution_status="executable",
            registry_status="deprecated_ad_hoc_legacy",
        )

    if default_target_id is None:
        raise ValueError(
            "Dataset config must define an explicit target_id; implicit "
            "target_main fallback is forbidden"
        )
    warnings.warn(
        f"Missing target_id uses deprecated compatibility default {default_target_id!r}",
        LegacyTargetConfigWarning,
        stacklevel=2,
    )
    return get_target_spec(default_target_id)


def _warn_legacy(alias: str, target_id: str) -> TargetSpec:
    warnings.warn(
        f"Legacy target {alias!r} maps explicitly to target_id {target_id!r}",
        LegacyTargetConfigWarning,
        stacklevel=3,
    )
    return get_target_spec(target_id)

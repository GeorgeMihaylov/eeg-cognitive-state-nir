"""Backend-neutral AutoML specifications and deterministic validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping


PARAMETER_TYPES = frozenset({"categorical", "integer", "float", "log_float"})


def stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SearchParameterSpec:
    path: str
    type: str
    choices: tuple[Any, ...] = ()
    low: int | float | None = None
    high: int | float | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("Search parameter path must be non-empty")
        if self.type not in PARAMETER_TYPES:
            raise ValueError(
                f"Unsupported search parameter type {self.type!r}; "
                f"available={sorted(PARAMETER_TYPES)}"
            )
        if self.type == "categorical":
            if not self.choices:
                raise ValueError(f"{self.path}: categorical choices must not be empty")
            if self.low is not None or self.high is not None:
                raise ValueError(
                    f"{self.path}: categorical parameters cannot define bounds"
                )
        else:
            if self.choices:
                raise ValueError(
                    f"{self.path}: numeric parameters cannot define choices"
                )
            if self.low is None or self.high is None:
                raise ValueError(f"{self.path}: numeric bounds are required")
            if self.low >= self.high:
                raise ValueError(f"{self.path}: low must be smaller than high")
            if self.type == "log_float" and self.low <= 0:
                raise ValueError(f"{self.path}: log_float low must be positive")
            if self.type == "integer" and (
                not isinstance(self.low, Integral)
                or not isinstance(self.high, Integral)
            ):
                raise ValueError(f"{self.path}: integer bounds must be integers")

    @classmethod
    def from_dict(
        cls,
        path: str,
        values: Mapping[str, Any],
    ) -> "SearchParameterSpec":
        allowed = {"type", "choices", "low", "high"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"{path}: unknown search-space fields: {unknown}")
        return cls(
            path=path,
            type=str(values.get("type", "")),
            choices=tuple(values.get("choices", ())),
            low=values.get("low"),
            high=values.get("high"),
        )

    def validate(self, value: Any) -> None:
        if self.type == "categorical":
            if value not in self.choices:
                raise ValueError(
                    f"{self.path}={value!r} is not one of {list(self.choices)!r}"
                )
            return
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{self.path} must be numeric")
        if self.type == "integer" and not isinstance(value, Integral):
            raise ValueError(f"{self.path} must be an integer")
        if not self.low <= value <= self.high:
            raise ValueError(
                f"{self.path}={value!r} is outside [{self.low}, {self.high}]"
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.type == "categorical":
            result["choices"] = list(self.choices)
        else:
            result["low"] = self.low
            result["high"] = self.high
        return result


@dataclass(frozen=True)
class SearchSpaceSpec:
    parameters: tuple[SearchParameterSpec, ...]
    constraints: tuple[str, ...] = ("d_model_divisible_by_nhead",)

    def __post_init__(self) -> None:
        paths = [parameter.path for parameter in self.parameters]
        if not paths:
            raise ValueError("Search space must contain at least one parameter")
        if len(paths) != len(set(paths)):
            raise ValueError("Search parameter paths must be unique")
        unknown_constraints = sorted(
            set(self.constraints) - {"d_model_divisible_by_nhead"}
        )
        if unknown_constraints:
            raise ValueError(f"Unknown search constraints: {unknown_constraints}")

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Mapping[str, Any]],
        *,
        constraints: tuple[str, ...] = ("d_model_divisible_by_nhead",),
    ) -> "SearchSpaceSpec":
        return cls(
            parameters=tuple(
                SearchParameterSpec.from_dict(path, spec)
                for path, spec in values.items()
            ),
            constraints=constraints,
        )

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(parameter.path for parameter in self.parameters)

    def validate_parameters(
        self,
        values: Mapping[str, Any],
        *,
        require_all: bool = True,
    ) -> None:
        unknown = sorted(set(values) - set(self.paths))
        if unknown:
            raise ValueError(f"Unknown search parameter paths: {unknown}")
        if require_all:
            missing = sorted(set(self.paths) - set(values))
            if missing:
                raise ValueError(f"Missing search parameter paths: {missing}")
        by_path = {parameter.path: parameter for parameter in self.parameters}
        for path, value in values.items():
            by_path[path].validate(value)
        if "d_model_divisible_by_nhead" in self.constraints:
            d_model = values.get("model.params.d_model")
            nhead = values.get("model.params.nhead")
            if d_model is not None and nhead is not None:
                if int(d_model) % int(nhead) != 0:
                    raise ValueError(
                        "Invalid Transformer shape: d_model must be divisible by nhead"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": {
                parameter.path: parameter.to_dict()
                for parameter in self.parameters
            },
            "constraints": list(self.constraints),
        }

    def config_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class AutoMLStudySpec:
    name: str
    objective_metric: str
    direction: str
    sampler_seed: int
    base_config_path: Path
    output_root: Path
    storage_path: Path
    inner_protocol: str
    inner_splits: int
    n_trials: int
    timeout_seconds: float | None
    max_epochs: int | None
    max_windows: int | None
    evaluate_best: bool
    search_space: SearchSpaceSpec

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Study name must be non-empty")
        if self.objective_metric != "balanced_accuracy":
            raise ValueError("Initial AutoML objective must be balanced_accuracy")
        if self.direction != "maximize":
            raise ValueError("Initial AutoML direction must be maximize")
        if self.inner_protocol != "group_kfold_subject":
            raise ValueError("Inner protocol must be group_kfold_subject")
        if self.inner_splits < 2:
            raise ValueError("inner_splits must be at least 2")
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        if self.max_epochs is not None and self.max_epochs < 1:
            raise ValueError("max_epochs must be positive when provided")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": {
                "name": self.name,
                "objective_metric": self.objective_metric,
                "direction": self.direction,
                "sampler_seed": self.sampler_seed,
            },
            "base_config": {"path": str(self.base_config_path)},
            "artifacts": {
                "output_root": str(self.output_root),
                "storage": str(self.storage_path),
            },
            "evaluation": {
                "nested": True,
                "inner_protocol": self.inner_protocol,
                "inner_splits": self.inner_splits,
                "evaluate_best": self.evaluate_best,
            },
            "search": {
                "n_trials": self.n_trials,
                "timeout_seconds": self.timeout_seconds,
                "max_epochs": self.max_epochs,
                "max_windows": self.max_windows,
            },
            "search_space": self.search_space.to_dict()["parameters"],
            "constraints": list(self.search_space.constraints),
        }

    def config_hash(self) -> str:
        return stable_hash(self.to_dict())

"""Explicit EEG channel-order and cross-dataset selection contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Single production source of truth for project Emotiv raw tensors.  The
# ``EEG.`` prefix is part of the project CSV column namespace.
PROJECT_EMOTIV_CHANNEL_ORDER = (
    "EEG.AF3",
    "EEG.F7",
    "EEG.F3",
    "EEG.FC5",
    "EEG.T7",
    "EEG.P7",
    "EEG.O1",
    "EEG.O2",
    "EEG.P8",
    "EEG.T8",
    "EEG.FC6",
    "EEG.F4",
    "EEG.F8",
    "EEG.AF4",
)

COG_BCI_AUXILIARY_CHANNELS = ("ECG1",)
COG_BCI_MAPPING_PATH = (
    Path(__file__).resolve().parent / "channel_maps" / "cog_bci_emotiv.json"
)


class ChannelHarmonizationError(ValueError):
    """Raised when an explicit channel-selection contract cannot be satisfied."""


def _metadata_value(metadata: Any, name: str, default: Any = None) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def _stable_layout_id(channel_names: Sequence[str]) -> str:
    payload = "".join(f"{name}\n" for name in channel_names).encode("utf-8")
    return f"layout-{hashlib.sha256(payload).hexdigest()[:12]}"


@dataclass(frozen=True)
class ChannelLayout:
    """Ordered EEG/auxiliary metadata for one physical recording layout."""

    layout_id: str
    channel_names: tuple[str, ...]
    eeg_channel_names: tuple[str, ...]
    auxiliary_channel_names: tuple[str, ...]
    has_cz: bool

    @classmethod
    def from_metadata(cls, metadata: Any) -> "ChannelLayout":
        all_names = tuple(
            str(value)
            for value in _metadata_value(metadata, "channel_names_total", ())
        )
        eeg_names = tuple(
            str(value)
            for value in _metadata_value(metadata, "eeg_channel_names", ())
        )
        auxiliary = tuple(
            str(value)
            for value in _metadata_value(
                metadata, "auxiliary_channel_names", ()
            )
        )
        if not all_names:
            all_names = (*eeg_names, *auxiliary)
        layout_id = str(
            _metadata_value(metadata, "channel_layout_id", "")
            or _stable_layout_id(all_names)
        )
        return cls(
            layout_id=layout_id,
            channel_names=all_names,
            eeg_channel_names=eeg_names,
            auxiliary_channel_names=auxiliary,
            has_cz=bool(
                _metadata_value(
                    metadata,
                    "has_cz",
                    any(name == "Cz" for name in eeg_names),
                )
            ),
        )


@dataclass(frozen=True)
class ChannelMappingEntry:
    """One explicit source-to-canonical channel mapping."""

    emotiv_channel: str
    cog_bci_channel: str
    match_type: str
    evidence: str
    coordinate_distance_mm: float | None = None
    status: str = "explicit_alias_match"

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotiv_channel": self.emotiv_channel,
            "cog_bci_channel": self.cog_bci_channel,
            "match_type": self.match_type,
            "source_name": self.cog_bci_channel,
            "target_name": self.emotiv_channel,
            "coordinate_distance_mm": self.coordinate_distance_mm,
            "evidence": self.evidence,
            "status": self.status,
        }


@dataclass(frozen=True)
class ChannelSelectionValidation:
    """Validation result before a Raw object is copied or changed."""

    policy_name: str
    source_layout_id: str
    source_names: tuple[str, ...]
    output_names: tuple[str, ...]
    missing_required: tuple[str, ...]
    ambiguous_required: tuple[str, ...]
    excluded_auxiliary: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.missing_required and not self.ambiguous_required

    def require_valid(self) -> None:
        if self.valid:
            return
        raise ChannelHarmonizationError(
            f"Channel policy {self.policy_name!r} cannot be applied to "
            f"{self.source_layout_id}: missing={list(self.missing_required)}, "
            f"ambiguous={list(self.ambiguous_required)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "source_layout_id": self.source_layout_id,
            "source_names": list(self.source_names),
            "output_names": list(self.output_names),
            "missing_required": list(self.missing_required),
            "ambiguous_required": list(self.ambiguous_required),
            "excluded_auxiliary": list(self.excluded_auxiliary),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class ChannelSelectionPolicy:
    """An ordered, non-signal-transforming channel selection policy."""

    name: str
    mode: str
    required_names: tuple[str, ...] = ()
    mappings: tuple[ChannelMappingEntry, ...] = ()

    def validate(self, record_metadata: Any) -> ChannelSelectionValidation:
        layout = ChannelLayout.from_metadata(record_metadata)
        eeg_names = layout.eeg_channel_names
        duplicate_names = {
            name for name in eeg_names if eeg_names.count(name) > 1
        }
        casefold_groups: dict[str, list[str]] = {}
        for name in eeg_names:
            casefold_groups.setdefault(name.casefold(), []).append(name)
        case_ambiguities = {
            name
            for names in casefold_groups.values()
            if len(set(names)) > 1
            for name in names
        }

        if self.mode == "native":
            ambiguous = tuple(sorted(duplicate_names | case_ambiguities))
            return ChannelSelectionValidation(
                policy_name=self.name,
                source_layout_id=layout.layout_id,
                source_names=eeg_names,
                output_names=eeg_names,
                missing_required=(),
                ambiguous_required=ambiguous,
                excluded_auxiliary=layout.auxiliary_channel_names,
            )

        if self.mode == "required_exact":
            missing = tuple(name for name in self.required_names if name not in eeg_names)
            ambiguous = tuple(
                name
                for name in self.required_names
                if name in duplicate_names or name in case_ambiguities
            )
            return ChannelSelectionValidation(
                policy_name=self.name,
                source_layout_id=layout.layout_id,
                source_names=tuple(
                    name for name in self.required_names if name in eeg_names
                ),
                output_names=tuple(
                    name for name in self.required_names if name in eeg_names
                ),
                missing_required=missing,
                ambiguous_required=ambiguous,
                excluded_auxiliary=layout.auxiliary_channel_names,
            )

        if self.mode != "mapped":
            raise ChannelHarmonizationError(
                f"Unknown channel policy mode {self.mode!r}"
            )

        by_target: dict[str, list[ChannelMappingEntry]] = {}
        for mapping in self.mappings:
            by_target.setdefault(mapping.emotiv_channel, []).append(mapping)
        source_names: list[str] = []
        output_names: list[str] = []
        missing: list[str] = []
        ambiguous: list[str] = []
        for target_name in self.required_names:
            # An exact source name always has priority over configured aliases.
            if target_name in eeg_names:
                candidates = [target_name]
            else:
                candidates = [
                    item.cog_bci_channel
                    for item in by_target.get(target_name, [])
                    if item.cog_bci_channel in eeg_names
                ]
            candidates = list(dict.fromkeys(candidates))
            if not candidates:
                missing.append(target_name)
                continue
            if len(candidates) > 1:
                ambiguous.append(target_name)
                continue
            source_name = candidates[0]
            if (
                source_name in duplicate_names
                or source_name in case_ambiguities
                or source_name in layout.auxiliary_channel_names
            ):
                ambiguous.append(target_name)
                continue
            source_names.append(source_name)
            output_names.append(target_name)
        return ChannelSelectionValidation(
            policy_name=self.name,
            source_layout_id=layout.layout_id,
            source_names=tuple(source_names),
            output_names=tuple(output_names),
            missing_required=tuple(missing),
            ambiguous_required=tuple(ambiguous),
            excluded_auxiliary=layout.auxiliary_channel_names,
        )

    def select_names(self, record_metadata: Any) -> list[str]:
        validation = self.validate(record_metadata)
        validation.require_valid()
        return list(validation.output_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "required_names": list(self.required_names),
            "mappings": [mapping.to_dict() for mapping in self.mappings],
        }


@dataclass(frozen=True)
class ChannelSelectionResult:
    """Selected Raw object plus deterministic selection provenance."""

    raw: Any
    validation: ChannelSelectionValidation
    provenance: Mapping[str, Any]

    @property
    def selected_names(self) -> tuple[str, ...]:
        return self.validation.output_names


def _layout_from_raw(raw: Any) -> ChannelLayout:
    names = tuple(str(name) for name in raw.ch_names)
    channel_types = tuple(str(value) for value in raw.get_channel_types())
    auxiliary = tuple(
        name
        for name, channel_type in zip(names, channel_types)
        if channel_type != "eeg" or name in COG_BCI_AUXILIARY_CHANNELS
    )
    eeg = tuple(name for name in names if name not in auxiliary)
    return ChannelLayout(
        layout_id=_stable_layout_id(names),
        channel_names=names,
        eeg_channel_names=eeg,
        auxiliary_channel_names=auxiliary,
        has_cz="Cz" in eeg,
    )


def apply_channel_policy(
    raw: Any,
    policy: ChannelSelectionPolicy,
    *,
    record_metadata: Any | None = None,
    copy: bool = True,
) -> ChannelSelectionResult:
    """Apply only ordered picking/renaming; never transform signal samples."""

    metadata = record_metadata if record_metadata is not None else _layout_from_raw(raw)
    validation = policy.validate(metadata)
    validation.require_valid()
    missing_from_raw = [
        name for name in validation.source_names if name not in raw.ch_names
    ]
    if missing_from_raw:
        raise ChannelHarmonizationError(
            f"Raw object is missing channels selected by {policy.name!r}: "
            f"{missing_from_raw}"
        )
    original_raw_names = tuple(str(name) for name in raw.ch_names)
    selected = raw.copy() if copy else raw
    selected.pick(list(validation.source_names))
    rename = {
        source: output
        for source, output in zip(
            validation.source_names, validation.output_names
        )
        if source != output
    }
    if rename:
        selected.rename_channels(rename)
    provenance = {
        "schema_version": 1,
        "policy_name": policy.name,
        "source_layout_id": validation.source_layout_id,
        "source_channel_names": list(original_raw_names),
        "selected_source_names": list(validation.source_names),
        "selected_channel_names": list(validation.output_names),
        "excluded_auxiliary_channels": list(validation.excluded_auxiliary),
        "renamed_channels": rename,
        "copy": bool(copy),
        "operations": ["pick", *(["rename_channels"] if rename else [])],
        "resampling": False,
        "filtering": False,
        "rereferencing": False,
    }
    return ChannelSelectionResult(
        raw=selected,
        validation=validation,
        provenance=provenance,
    )


def compute_common_eeg_channel_order(records: Iterable[Any]) -> tuple[str, ...]:
    """Preserve first-layout order while intersecting all record EEG layouts."""

    layouts = [ChannelLayout.from_metadata(record) for record in records]
    if not layouts:
        raise ChannelHarmonizationError(
            "Cannot compute common EEG channels from an empty record collection"
        )
    common = set(layouts[0].eeg_channel_names)
    for layout in layouts[1:]:
        common.intersection_update(layout.eeg_channel_names)
    ordered = tuple(
        name for name in layouts[0].eeg_channel_names if name in common
    )
    if len(ordered) != len(common):
        raise ChannelHarmonizationError(
            "Common EEG channel order is ambiguous in the first layout"
        )
    return ordered


def load_cog_bci_emotiv_mapping(
    path: Path | str = COG_BCI_MAPPING_PATH,
) -> tuple[ChannelMappingEntry, ...]:
    """Load and validate the small tracked namespace mapping artifact."""

    with Path(path).open(encoding="utf-8") as input_file:
        document = json.load(input_file)
    if int(document.get("schema_version", -1)) != 1:
        raise ChannelHarmonizationError(
            "Unsupported COG-BCI/Emotiv channel mapping schema"
        )
    canonical_order = tuple(str(value) for value in document["canonical_order"])
    if canonical_order != PROJECT_EMOTIV_CHANNEL_ORDER:
        raise ChannelHarmonizationError(
            "Tracked COG-BCI mapping order does not match the production "
            "Emotiv channel contract"
        )
    mappings = tuple(
        ChannelMappingEntry(
            emotiv_channel=str(item["emotiv_channel"]),
            cog_bci_channel=str(item["cog_bci_channel"]),
            match_type=str(item["match_type"]),
            evidence=str(item["evidence"]),
            coordinate_distance_mm=(
                None
                if item.get("coordinate_distance_mm") is None
                else float(item["coordinate_distance_mm"])
            ),
            status=str(item.get("status", item["match_type"])),
        )
        for item in document["mapping"]
    )
    if tuple(item.emotiv_channel for item in mappings) != canonical_order:
        raise ChannelHarmonizationError(
            "Tracked COG-BCI mapping entries are not in canonical order"
        )
    return mappings


def build_cog_bci_channel_policy(
    name: str,
    *,
    records: Iterable[Any],
) -> ChannelSelectionPolicy:
    """Build one of the three supported COG-BCI policies."""

    normalized = str(name).strip()
    if normalized == "cog_bci_native":
        return ChannelSelectionPolicy(name=normalized, mode="native")
    if normalized == "cog_bci_common":
        common = compute_common_eeg_channel_order(records)
        return ChannelSelectionPolicy(
            name=normalized,
            mode="required_exact",
            required_names=common,
        )
    if normalized == "emotiv_common":
        return ChannelSelectionPolicy(
            name=normalized,
            mode="mapped",
            required_names=PROJECT_EMOTIV_CHANNEL_ORDER,
            mappings=load_cog_bci_emotiv_mapping(),
        )
    raise ChannelHarmonizationError(
        f"Unknown COG-BCI channel policy {name!r}; available="
        "['cog_bci_common', 'cog_bci_native', 'emotiv_common']"
    )


def channel_contract_json(policy: ChannelSelectionPolicy) -> str:
    """Stable semantic serialization used by tests and audit artifacts."""

    return json.dumps(
        policy.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

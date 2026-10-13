from __future__ import annotations

import copy
import math
from collections.abc import Mapping, MutableMapping, Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, cast

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from mahi_mahi.motif_visualization import RenderOptions


class ResourcePath(Protocol):
    name: str

    def read_text(self, encoding: str | None = None) -> str: ...


ConfigPath = str | Path | ResourcePath


def validate_fraction_or_infinity(value: float) -> float:
    if math.isnan(value) or value < 0 or (value > 1 and not math.isinf(value)):
        raise ValueError("must be between 0 and 1 inclusive, or Infinity")
    return value


FractionOrInfinity = Annotated[float, AfterValidator(validate_fraction_or_infinity)]


class MotifPlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sort_mode: Literal["length_desc", "length_asc", "keep_input"]
    split_by: Literal["none", "strand", "haplotype", "allele"]
    show_strand_sidebar: bool
    show_haplotype_sidebar: bool
    show_allele_sidebar: bool
    show_motif_legend: bool
    show_motif_boundaries: bool
    hide_non_spanning: bool
    hide_unused: bool
    hide_suspicious: bool
    mask_suspicious: bool

    def to_render_options(self) -> RenderOptions:
        from mahi_mahi.motif_visualization import RenderOptions

        return RenderOptions(**self.model_dump())


class ClusteringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    non_spanning_read_policy: Literal["default", "exclude"]
    hap_max_mismatch_fraction: float = Field(ge=0, le=1)
    exclude_haplotag_inconsistent_reads: bool
    plot_first_attempt_density: bool


class MahiMahiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    extension: int = Field(ge=0)
    region_merge_distance: int = Field(ge=0)
    min_region_coverage: int = Field(ge=0)
    min_read_support_per_allele: int = Field(ge=1)
    include_extra_trs: bool
    generate_clustering_plots: bool
    generate_motif_plots: bool
    generate_methylation_plots: bool
    methylation_state_confidence_threshold: float = Field(ge=0, le=1)
    motif_plot: MotifPlotConfig
    clustering: ClusteringConfig
    very_wide_cluster_minimum_q75_q50_ratio: float = Field(ge=0)
    strand_bias_run_min_tr_length: int = Field(ge=0)
    strand_bias_run_min_long_tr_fraction: float = Field(ge=0, le=1)
    strand_bias_kmer_coverage_threshold: FractionOrInfinity
    strand_bias_motif_composition_difference_threshold: FractionOrInfinity
    strand_bias_reads_fraction_threshold: FractionOrInfinity
    strand_bias_hide_motifs_reads_fraction_threshold: FractionOrInfinity
    strand_bias_exclude_suspicious_reads: Literal["size_and_motifs", "motifs_only", "none"]
    strand_bias_min_remaining_reads_per_allele: int = Field(ge=0)
    strand_bias_method: Literal["statistical_testing", "counts_threshold"]
    min_flank_length: int = Field(ge=1)
    max_flank_length: int = Field(ge=0)
    flank_realignment_check_epsilon: int = Field(ge=0)
    repeated_sequence_entropy_threshold: float = Field(ge=0)
    aba_middle_exact_fraction_threshold: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_hide_motif_thresholds(self) -> MahiMahiConfig:
        if self.strand_bias_hide_motifs_reads_fraction_threshold < self.strand_bias_reads_fraction_threshold:
            raise ValueError("strand_bias_hide_motifs_reads_fraction_threshold must be at least as strict as strand_bias_reads_fraction_threshold")
        return self


class MahiMahiConfigFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    defaults: dict[str, Any]
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)


def default_config_path() -> ResourcePath:
    return cast(ResourcePath, resources.files("mahi_mahi.data").joinpath("config.yaml"))


def _read_text(path: ConfigPath) -> str:
    if isinstance(path, str | Path):
        return Path(path).read_text(encoding="utf-8")
    return path.read_text(encoding="utf-8")


def _yaml_error_detail(exc: yaml.YAMLError) -> str:
    problem = getattr(exc, "problem", None)
    mark = getattr(exc, "problem_mark", None)
    if problem and mark:
        return f"{problem} at line {mark.line + 1}, column {mark.column + 1}"
    return str(exc)


def _load_yaml(path: ConfigPath) -> dict[str, Any]:
    try:
        data = yaml.safe_load(_read_text(path))
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed config in {path}: {_yaml_error_detail(exc)}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected config to be a mapping at top level in {path}, got {type(data).__name__}")
    return data


def _deep_update(base: MutableMapping[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _set_path(cfg: MutableMapping[str, Any], dotted_key: str, value: object) -> None:
    if not dotted_key:
        raise ValueError("Override key cannot be empty")
    parts = dotted_key.split(".")
    if any(part == "" for part in parts):
        raise ValueError(f"Override key contains an empty path segment: {dotted_key!r}")
    cursor: MutableMapping[str, Any] = cfg
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, MutableMapping):
            raise TypeError(f"Cannot set {dotted_key!r}: {part!r} is not a mapping")
        cursor = current
    cursor[parts[-1]] = value


def _parse_override_value(raw: str) -> object:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed config override value {raw!r}: {_yaml_error_detail(exc)}") from exc


def _parse_set_overrides(overrides: Sequence[str] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Config override must be KEY=VALUE, got: {item!r}")
        key, raw = item.split("=", 1)
        _set_path(parsed, key, _parse_override_value(raw))
    return parsed


def _merged_config_payload(config_file: MahiMahiConfigFile, profile: str | None) -> dict[str, Any]:
    payload = copy.deepcopy(config_file.defaults)
    if profile is None:
        return payload
    profile_patch = config_file.profiles.get(profile)
    if profile_patch is None:
        available = ", ".join(sorted(config_file.profiles)) or "<none>"
        raise ValueError(f"Unknown profile {profile!r}; available profiles: {available}")
    _deep_update(payload, profile_patch)
    return payload


def _validate_profile_patches(config_file: MahiMahiConfigFile) -> None:
    MahiMahiConfig.model_validate(config_file.defaults)
    for profile_name, profile_patch in config_file.profiles.items():
        payload = copy.deepcopy(config_file.defaults)
        _deep_update(payload, profile_patch)
        try:
            MahiMahiConfig.model_validate(payload)
        except ValueError as exc:
            raise ValueError(f"Invalid config profile {profile_name!r}: {exc}") from exc


def load_config(
    path: ConfigPath | None = None,
    *,
    profile: str | None = None,
    set_overrides: Sequence[str] | None = None,
) -> MahiMahiConfig:
    raw_config = _load_yaml(default_config_path())
    if path is not None:
        _deep_update(raw_config, _load_yaml(path))
    config_file = MahiMahiConfigFile.model_validate(raw_config)
    _validate_profile_patches(config_file)
    payload = _merged_config_payload(config_file, profile)
    _deep_update(payload, _parse_set_overrides(set_overrides))
    return MahiMahiConfig.model_validate(payload)

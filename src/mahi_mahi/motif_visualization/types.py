from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

RGBColor: TypeAlias = tuple[float, float, float]
SortMode: TypeAlias = Literal["length_desc", "length_asc", "keep_input"]
SplitMode: TypeAlias = Literal["none", "strand", "haplotype", "allele"]
SegmentMode: TypeAlias = Literal["motif", "methylation"]


@dataclass(frozen=True)
class PlotRead:
    patterns: tuple[tuple[str, int], ...]
    strand: str
    read_id: str
    extra: dict[str, object]


@dataclass(frozen=True)
class PlotPanel:
    key: str
    title: str
    reads: tuple[PlotRead, ...]
    expected_motifs: tuple[str, ...]
    repeat_unit_length: int | None
    repeat_id: str
    chrom: str
    start: int
    end: int
    variant_id: str
    repeat_unit: str


@dataclass(frozen=True)
class RenderOptions:
    sort_mode: SortMode = "length_desc"
    split_by: SplitMode = "allele"
    #: What the blocks of the main plot represent.
    segment_mode: SegmentMode = "motif"
    methylation_state_confidence_threshold: float = 0.80
    show_strand_sidebar: bool = False
    show_haplotype_sidebar: bool = True
    show_allele_sidebar: bool = False
    show_motif_legend: bool = True
    show_other_in_legend: bool = True
    show_motif_boundaries: bool = True
    hide_non_spanning: bool = False
    hide_unused: bool = False
    hide_suspicious: bool = False
    mask_suspicious: bool = True
    strand_separator_rows: int = 2
    haplotype_separator_rows: int = 2
    allele_separator_rows: int = 2
    motif_other_label: str = "Other"
    motif_other_color: RGBColor = (0.75, 0.75, 0.75)
    strand_colors: Mapping[str, RGBColor] = field(
        default_factory=lambda: {
            "+": (0.00, 0.50, 0.35),
            "-": (0.90, 0.60, 0.00),
            "unknown": (0.50, 0.50, 0.50),
        }
    )
    # A single violet hue in two lightnesses: distinct from the green/amber strand sidebar and from
    # the motif palette used in the main plot. Haplotype and allele deliberately share colors.
    haplotype_colors: Mapping[str, RGBColor] = field(
        default_factory=lambda: {
            "1": (0.29, 0.00, 0.51),
            "2": (0.72, 0.58, 0.91),
            "0": (0.50, 0.50, 0.50),
        }
    )
    allele_colors: Mapping[str, RGBColor] = field(
        default_factory=lambda: {
            "1": (0.29, 0.00, 0.51),
            "2": (0.72, 0.58, 0.91),
            "0": (0.50, 0.50, 0.50),
        }
    )


@dataclass(frozen=True)
class NonSpanningReadInfo:
    row_index: int
    end: Literal["left", "right"]
    display_offset: int
    read_length: int


@dataclass(frozen=True)
class RenderSegment:
    motif: str
    label: str
    color: RGBColor
    x_start: int
    x_end: int
    motif_length: int


@dataclass(frozen=True)
class RenderReadRow:
    row_index: int
    read: PlotRead
    read_length: int
    display_offset: int
    is_suspicious: bool
    strand: str
    haplotype: str
    allele: str
    segments: tuple[RenderSegment, ...]
    motif_boundaries: tuple[int, ...]
    section_boundaries: tuple[int, ...]


@dataclass(frozen=True)
class RenderSeparatorRow:
    row_index: int


RenderRow: TypeAlias = RenderReadRow | RenderSeparatorRow


@dataclass(frozen=True)
class AxisDomain:
    motif_cols: int
    default_x_axis_limits: tuple[float, float]
    repeat_unit_length: int | None


@dataclass(frozen=True)
class PanelLegends:
    pattern_colors: Mapping[str, RGBColor]
    motif_filter_active: bool
    masked_suspicious_present: bool
    present_strands: set[str]
    present_haplotypes: set[str]
    present_alleles: set[str]


@dataclass(frozen=True)
class PreparedPanel:
    status: Literal["ok", "insufficient_reads", "no_drawable_data"]
    message: str | None
    rows: tuple[RenderRow, ...]
    axis: AxisDomain
    legends: PanelLegends
    separator_starts: tuple[int, ...]
    non_spanning_reads: tuple[NonSpanningReadInfo, ...]
    options: RenderOptions

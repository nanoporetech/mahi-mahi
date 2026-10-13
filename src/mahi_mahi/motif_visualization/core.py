from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import numpy as np

from mahi_mahi.methylation import METHYLATED, NO_CALL, UNCERTAIN, UNMETHYLATED, cpg_offsets

from .palette import palette
from .types import (
    AxisDomain,
    NonSpanningReadInfo,
    PanelLegends,
    PlotPanel,
    PlotRead,
    PreparedPanel,
    RenderOptions,
    RenderReadRow,
    RenderRow,
    RenderSegment,
    RenderSeparatorRow,
    RGBColor,
    SortMode,
)

MAX_COLORED_MOTIFS = 20
MOTIF_MIN_REPEAT_COUNT = 5
MOTIF_MIN_READ_SUPPORT = 3

_NOT_CPG_LABEL = "Not a CpG"
_MISSING_METHYLATION_LABEL = "Missing data"
_UNCERTAIN_METHYLATION_LABEL = "Uncertain methylation"
_METHYLATION_STATE_LABELS = {
    METHYLATED: "Modified CpG",
    UNMETHYLATED: "Unmethylated CpG",
    NO_CALL: _MISSING_METHYLATION_LABEL,
    UNCERTAIN: _UNCERTAIN_METHYLATION_LABEL,
}
#: Legend order is dict order, so keep the informative states first.
METHYLATION_COLORS: Mapping[str, RGBColor] = {
    "Modified CpG": (0.70, 0.11, 0.11),
    "Unmethylated CpG": (0.13, 0.44, 0.71),
    _MISSING_METHYLATION_LABEL: (191 / 255, 167 / 255, 127 / 255),
    _UNCERTAIN_METHYLATION_LABEL: (0.72, 0.72, 0.72),
    _NOT_CPG_LABEL: (0.88, 0.88, 0.88),
}


def prepare_panel(panel: PlotPanel, options: RenderOptions, read_threshold: int = 1) -> PreparedPanel:
    reads = _filter_reads(panel.reads, options)
    if len(reads) < read_threshold:
        return _placeholder_prepared("insufficient_reads", f"Not enough reads for {panel.title}", options)

    motif_repeat_counts = _compute_motif_repeat_counts(reads)
    motif_read_support_counts = _compute_motif_read_support_counts(reads)
    selected_motifs, motif_filter_active = _resolve_motif_display_policy(
        observed_motifs=tuple(motif_repeat_counts),
        expected_motifs=panel.expected_motifs,
        motif_repeat_counts=motif_repeat_counts,
        motif_read_support_counts=motif_read_support_counts,
    )
    pattern_colors = _build_pattern_colors(selected_motifs, options)
    if options.segment_mode == "methylation":
        # the blocks are methylation states rather than motifs, so the motif palette and
        # the "Other" grouping of rare motifs do not apply
        selected_motifs, motif_filter_active = [], False
        pattern_colors = dict(METHYLATION_COLORS)
    ordered_rows, separator_starts = _order_reads(reads, options)
    rows, non_spanning_reads, min_start, max_end, legends = _build_render_rows(
        ordered_rows,
        pattern_colors=pattern_colors,
        selected_motifs=set(selected_motifs),
        motif_filter_active=motif_filter_active,
        options=options,
    )
    num_rows = len(rows)
    longest_used_read = max(
        (patterns_length(read.patterns) for read in reads if _read_is_used_for_allele_size(read) is True),
        default=None,
    )
    visible_max_end = min(max_end, 2 * longest_used_read) if longest_used_read is not None else max_end
    motif_cols = visible_max_end - min_start
    visible_bp = visible_max_end - min_start
    x_padding = max(1.0, 0.01 * float(visible_bp))
    x_left = float(min_start) - x_padding
    x_right = float(visible_max_end) + x_padding
    x_left = max(0.0, x_left)
    axis = AxisDomain(
        motif_cols=motif_cols,
        default_x_axis_limits=(x_left, x_right),
        repeat_unit_length=panel.repeat_unit_length,
    )
    if num_rows == 0 or motif_cols == 0 or visible_bp == 0:
        return PreparedPanel(
            status="no_drawable_data",
            message=f"No drawable motif data for {panel.title}",
            rows=rows,
            axis=axis,
            legends=legends,
            separator_starts=tuple(separator_starts),
            non_spanning_reads=tuple(non_spanning_reads),
            options=options,
        )
    return PreparedPanel(
        status="ok",
        message=None,
        rows=rows,
        axis=axis,
        legends=legends,
        separator_starts=tuple(separator_starts),
        non_spanning_reads=tuple(non_spanning_reads),
        options=options,
    )


def _placeholder_prepared(
    status: Literal["insufficient_reads", "no_drawable_data"],
    message: str,
    options: RenderOptions,
) -> PreparedPanel:
    return PreparedPanel(
        status=status,
        message=message,
        rows=(),
        axis=AxisDomain(motif_cols=0, default_x_axis_limits=(0.0, 0.0), repeat_unit_length=None),
        legends=PanelLegends(
            pattern_colors={},
            motif_filter_active=False,
            masked_suspicious_present=False,
            present_strands=set(),
            present_haplotypes=set(),
            present_alleles=set(),
        ),
        separator_starts=(),
        non_spanning_reads=(),
        options=options,
    )


def _filter_reads(reads: Sequence[PlotRead], options: RenderOptions) -> list[PlotRead]:
    result = list(reads)
    if options.hide_non_spanning:
        result = [read for read in result if read_is_spanning(read)]
    if options.hide_suspicious:
        result = [read for read in result if not _read_is_suspicious(read)]
    if options.hide_unused:
        result = [read for read in result if _read_is_used_for_allele_size(read) is not False]
    return result


def _order_reads(reads: Sequence[PlotRead], options: RenderOptions) -> tuple[list[PlotRead | None], list[int]]:
    if options.split_by == "haplotype":
        return _group_reads_by(reads, _read_haplotype, ("1", "2", "0"), options.sort_mode, options.haplotype_separator_rows)
    if options.split_by == "strand":
        return _group_reads_by(
            reads,
            lambda read: _normalize_strand(read.strand),
            ("+", "-", "unknown"),
            options.sort_mode,
            options.strand_separator_rows,
        )
    if options.split_by == "allele":
        return _group_reads_by(reads, _read_allele, ("1", "2", "0"), options.sort_mode, options.allele_separator_rows)
    if options.split_by == "none":
        ordered: list[PlotRead | None] = [*_sort_reads(reads, options.sort_mode)]
        return ordered, []
    raise ValueError(f"Invalid split_by={options.split_by!r}")


def _group_reads_by(
    reads: Sequence[PlotRead],
    key_fn: Callable[[PlotRead], str],
    group_order: Sequence[str],
    sort_mode: SortMode,
    separator_rows: int,
) -> tuple[list[PlotRead | None], list[int]]:
    grouped: dict[str, list[PlotRead]] = {group: [] for group in group_order}
    for read in reads:
        grouped.setdefault(str(key_fn(read)), []).append(read)

    rows: list[PlotRead | None] = []
    separators: list[int] = []
    for label in group_order:
        group = grouped.get(label, [])
        if not group:
            continue
        if rows and separator_rows > 0:
            separators.append(len(rows))
            rows.extend([None] * separator_rows)
        rows.extend(_sort_reads(group, sort_mode))
    return rows, separators


def _sort_reads(reads: Sequence[PlotRead], sort_mode: SortMode) -> list[PlotRead]:
    indexed_reads = list(enumerate(reads))
    if sort_mode == "keep_input":
        return [read for _, read in indexed_reads]
    if sort_mode == "length_desc":
        return [read for _, read in sorted(indexed_reads, key=lambda item: (-patterns_length(item[1].patterns), item[0]))]
    if sort_mode == "length_asc":
        return [read for _, read in sorted(indexed_reads, key=lambda item: (patterns_length(item[1].patterns), item[0]))]
    raise ValueError(f"Invalid sort_mode={sort_mode!r}")


def _build_render_rows(
    ordered_rows: Sequence[PlotRead | None],
    *,
    pattern_colors: Mapping[str, RGBColor],
    selected_motifs: set[str],
    motif_filter_active: bool,
    options: RenderOptions,
) -> tuple[tuple[RenderRow, ...], list[NonSpanningReadInfo], int, int, PanelLegends]:
    rows: list[RenderRow] = []
    non_spanning_reads: list[NonSpanningReadInfo] = []
    min_start = 0
    max_end = 0
    present_strands: set[str] = set()
    present_haplotypes: set[str] = set()
    present_alleles: set[str] = set()
    masked_suspicious_present = False

    for row_index, read in enumerate(ordered_rows):
        if read is None:
            rows.append(RenderSeparatorRow(row_index=row_index))
            continue
        read_length = patterns_length(read.patterns)
        display_offset = 0
        non_spanning_end = _read_non_spanning_end(read)
        if non_spanning_end is not None:
            non_spanning_reads.append(
                NonSpanningReadInfo(
                    row_index=row_index,
                    end=non_spanning_end,
                    display_offset=display_offset,
                    read_length=read_length,
                )
            )
        min_start = min(min_start, display_offset)
        max_end = max(max_end, display_offset + read_length)

        strand = _normalize_strand(read.strand)
        haplotype = _read_haplotype(read)
        allele = _read_allele(read)
        present_strands.add(strand)
        present_haplotypes.add(haplotype)
        present_alleles.add(allele)
        is_suspicious = _read_is_suspicious(read)
        masked_suspicious_present = masked_suspicious_present or (options.mask_suspicious and is_suspicious)
        if options.segment_mode == "methylation":
            segments, boundaries = _render_methylation_segments_for_read(read, pattern_colors=pattern_colors)
        else:
            segments, boundaries = _render_segments_for_read(
                read,
                pattern_colors=pattern_colors,
                selected_motifs=selected_motifs,
                motif_filter_active=motif_filter_active,
                options=options,
            )
        rows.append(
            RenderReadRow(
                row_index=row_index,
                read=read,
                read_length=read_length,
                display_offset=display_offset,
                is_suspicious=is_suspicious,
                strand=str(read.strand),
                haplotype=haplotype,
                allele=allele,
                segments=segments,
                motif_boundaries=boundaries,
                section_boundaries=_read_section_boundaries(read),
            )
        )

    legends = PanelLegends(
        pattern_colors=pattern_colors,
        motif_filter_active=motif_filter_active,
        masked_suspicious_present=masked_suspicious_present,
        present_strands=present_strands,
        present_haplotypes=present_haplotypes,
        present_alleles=present_alleles,
    )
    return tuple(rows), non_spanning_reads, min_start, max_end, legends


def _render_segments_for_read(
    read: PlotRead,
    *,
    pattern_colors: Mapping[str, RGBColor],
    selected_motifs: set[str],
    motif_filter_active: bool,
    options: RenderOptions,
) -> tuple[tuple[RenderSegment, ...], tuple[int, ...]]:
    cursor = 0
    segments: list[RenderSegment] = []
    boundaries: list[int] = []
    unknown_color = (0.5, 0.5, 0.5)
    for motif, raw_count in read.patterns:
        count = int(raw_count)
        if count <= 0 or not motif:
            continue
        motif_length = len(motif)
        block_start = cursor
        block_end = cursor + motif_length * count
        cursor = block_end
        if motif_filter_active and motif not in selected_motifs:
            label = options.motif_other_label
            color = options.motif_other_color
        else:
            label = motif
            color = pattern_colors.get(motif, unknown_color)
        segments.append(RenderSegment(motif=motif, label=label, color=color, x_start=block_start, x_end=block_end, motif_length=motif_length))
        boundaries.extend(block_start + motif_length * (repeat_index + 1) for repeat_index in range(count))
    if boundaries and boundaries[-1] == cursor:
        boundaries.pop()
    return tuple(segments), tuple(boundaries)


def _render_methylation_segments_for_read(
    read: PlotRead,
    *,
    pattern_colors: Mapping[str, RGBColor],
) -> tuple[tuple[RenderSegment, ...], tuple[int, ...]]:
    """Colour each base by its combined modification state.

    Both bases of a CpG are coloured, and adjacent bases sharing a state are merged into one
    block so that a long read does not produce one segment per base.  No motif boundaries are
    reported, because the blocks no longer correspond to motifs.
    """
    sequence = "".join(motif * int(raw_count) for motif, raw_count in read.patterns if motif and int(raw_count) > 0)
    raw_states = read.extra.get("mc_sites")
    states = raw_states if isinstance(raw_states, str) else None

    labels = [_NOT_CPG_LABEL] * len(sequence)
    for site_index, offset in enumerate(cpg_offsets(sequence)):
        state = states[site_index] if states is not None and site_index < len(states) else NO_CALL
        label = _METHYLATION_STATE_LABELS.get(state, _MISSING_METHYLATION_LABEL)
        labels[offset] = labels[offset + 1] = label

    segments: list[RenderSegment] = []
    block_start = 0
    for position in range(1, len(labels) + 1):
        if position < len(labels) and labels[position] == labels[block_start]:
            continue
        label = labels[block_start]
        segments.append(
            RenderSegment(
                motif="",
                label=label,
                color=pattern_colors.get(label, (0.5, 0.5, 0.5)),
                x_start=block_start,
                x_end=position,
                motif_length=position - block_start,
            )
        )
        block_start = position
    return tuple(segments), ()


def _resolve_motif_display_policy(
    *,
    observed_motifs: Sequence[str],
    expected_motifs: Sequence[str],
    motif_repeat_counts: Mapping[str, int],
    motif_read_support_counts: Mapping[str, int],
) -> tuple[list[str], bool]:
    observed_set = set(observed_motifs)
    selected = [motif for motif in _dedupe_preserving_order(expected_motifs) if motif in observed_set]
    seen = set(selected)
    for motif in sorted(observed_set, key=lambda motif: (len(motif), motif)):
        if len(selected) >= MAX_COLORED_MOTIFS:
            break
        if motif in seen:
            continue
        if motif_repeat_counts.get(motif, 0) < MOTIF_MIN_REPEAT_COUNT:
            continue
        if motif_read_support_counts.get(motif, 0) < MOTIF_MIN_READ_SUPPORT:
            continue
        selected.append(motif)
        seen.add(motif)
    if selected or not observed_set:
        return selected, len(selected) < len(observed_set)

    top_motif = sorted(observed_set, key=lambda motif: (-motif_repeat_counts.get(motif, 0), len(motif), motif))[0]
    return [top_motif], len(observed_set) > 1


def _build_pattern_colors(selected_motifs: Sequence[str], options: RenderOptions) -> dict[str, RGBColor]:
    if not selected_motifs:
        return {}
    if len(selected_motifs) > MAX_COLORED_MOTIFS:
        raise ValueError(f"At most {MAX_COLORED_MOTIFS} motifs can be assigned unique colors.")
    colors = palette("colorblind", len(selected_motifs)) if len(selected_motifs) <= 10 else palette("tab20", len(selected_motifs))
    pattern_colors = dict(zip(selected_motifs, colors, strict=True))
    reserved_colors = (options.motif_other_color, (0.5, 0.5, 0.5))
    replacement_pool = [
        color
        for palette_name, palette_size in (("tab20", 20), ("Set1", 9), ("Dark2", 8), ("Paired", 12))
        for color in palette(palette_name, palette_size)
        if not _is_approximately_gray(color) and all(_color_distance(color, reserved) >= 0.09 for reserved in reserved_colors)
    ]
    used_colors: list[RGBColor] = []
    replacement_index = 0
    for motif, color in list(pattern_colors.items()):
        if all(_color_distance(color, reserved) >= 0.09 for reserved in reserved_colors):
            used_colors.append(color)
            continue
        while replacement_index < len(replacement_pool):
            candidate = replacement_pool[replacement_index]
            replacement_index += 1
            if any(_color_distance(candidate, used) < 0.03 for used in used_colors):
                continue
            pattern_colors[motif] = candidate
            used_colors.append(candidate)
            break
    return pattern_colors


def _compute_motif_repeat_counts(reads: Sequence[PlotRead]) -> dict[str, int]:
    repeat_counts: dict[str, int] = {}
    for read in reads:
        for motif, count in read.patterns:
            if int(count) <= 0 or not motif:
                continue
            repeat_counts[motif] = repeat_counts.get(motif, 0) + int(count)
    return repeat_counts


def _compute_motif_read_support_counts(reads: Sequence[PlotRead]) -> dict[str, int]:
    read_support_counts: dict[str, int] = {}
    for read in reads:
        for motif in {motif for motif, count in read.patterns if int(count) > 0 and motif}:
            read_support_counts[motif] = read_support_counts.get(motif, 0) + 1
    return read_support_counts


def read_is_spanning(read: PlotRead) -> bool:
    raw = read.extra.get("is_spanning", 1)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false"}
    return raw != 0


def _read_is_suspicious(read: PlotRead) -> bool:
    raw = read.extra.get("is_suspicious", 0)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true"}
    return raw != 0


def _read_is_used_for_allele_size(read: PlotRead) -> bool | None:
    if "included_in_allele_size_estimation" not in read.extra:
        return None
    raw = read.extra["included_in_allele_size_estimation"]
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true"}
    return raw != 0


def _read_non_spanning_end(read: PlotRead) -> Literal["left", "right"] | None:
    if read_is_spanning(read):
        return None
    raw = read.extra.get("read_start")
    if raw is None:
        return None
    return "left" if _to_int(raw) == 0 else "right"


def _read_section_boundaries(read: PlotRead) -> tuple[int, ...]:
    raw_boundaries = read.extra.get("section_boundaries")
    if not isinstance(raw_boundaries, list | tuple):
        return ()
    return tuple(_to_int(boundary) for boundary in raw_boundaries)


def _to_int(value: object) -> int:
    if isinstance(value, bool | int | float | str):
        return int(value)
    raise TypeError(f"Expected an int-compatible value, got {type(value).__name__}")


def _read_haplotype(read: PlotRead) -> str:
    value = str(read.extra.get("original_hap", "0")).strip()
    return value if value in {"1", "2"} else "0"


def _read_allele(read: PlotRead) -> str:
    value = str(read.extra.get("allele_id", "0")).strip()
    return value if value in {"1", "2"} else "0"


def _normalize_strand(strand: str) -> str:
    normalized = str(strand).strip().lower()
    if normalized in {"+", "positive", "pos", "forward", "fwd", "1"}:
        return "+"
    if normalized in {"-", "negative", "neg", "reverse", "rev", "-1"}:
        return "-"
    return "unknown"


def patterns_length(patterns: Sequence[tuple[str, int]]) -> int:
    return sum(len(motif) * int(count) for motif, count in patterns if int(count) > 0 and motif)


def _dedupe_preserving_order(motifs: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for motif in motifs:
        if motif in seen:
            continue
        seen.add(motif)
        result.append(motif)
    return result


def nice_tick_step(n: int, target_tick_count: int = 7) -> int:
    if n <= 0 or target_tick_count <= 0:
        return 1
    raw_step = max(float(n) / float(target_tick_count), 1.0)
    magnitude = 10 ** int(np.floor(np.log10(raw_step)))
    for base in (1, 2, 5):
        step = int(base * magnitude)
        if step >= raw_step:
            return max(step, 1)
    return max(int(10 * magnitude), 1)


def _color_distance(a: RGBColor, b: RGBColor) -> float:
    return float(np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2)))


def _is_approximately_gray(color: RGBColor, channel_delta: float = 0.12) -> bool:
    return (max(color) - min(color)) < channel_delta

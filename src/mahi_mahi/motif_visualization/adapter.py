from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from mahi_mahi.io_utils.bed_utils import Locus, is_primary_output_locus
from mahi_mahi.pipeline_records import ReadCall, RegionCallSet
from mahi_mahi.utils import expand_motif_representation

from .core import patterns_length, read_is_spanning
from .types import PlotPanel, PlotRead


def build_plot_panels(regions: Sequence[RegionCallSet]) -> list[PlotPanel]:
    panels: list[PlotPanel] = []
    for region in regions:
        if not region.reads:
            continue
        expected_motifs = tuple(_dedupe_preserving_order(expand_motif_representation(region.repeat_unit)))
        repeat_unit_length = len(expected_motifs[0]) if expected_motifs else None
        title = _panel_title(region.repeat_id, region.repeat_unit, region.chrom, region.start, region.end)
        panels.append(
            PlotPanel(
                key=_build_panel_key(region.repeat_id, region.chrom, region.start, region.end),
                title=title,
                reads=tuple(_plot_read_from_record(record) for record in region.reads),
                expected_motifs=expected_motifs,
                repeat_unit_length=repeat_unit_length,
                repeat_id=region.repeat_id,
                chrom=region.chrom,
                start=region.start,
                end=region.end,
                variant_id=region.variant_id,
                repeat_unit=region.repeat_unit,
            )
        )
    return _merge_composite_panels(panels)


def _plot_read_from_record(record: ReadCall) -> PlotRead:
    extra: dict[str, object] = {
        "original_hap": record.original_hap,
        "allele_id": record.allele_id or "0",
        "analysis_exclusion_reason": record.analysis_exclusion_reason,
        "is_spanning": int(record.is_spanning),
        "is_suspicious": int(record.is_suspicious),
        "read_start": record.read_start,
        "included_in_allele_size_estimation": int(record.included_in_allele_size_estimation),
        "mc_sites": record.mc_sites,
        "_section_variant_id": record.variant_id,
        "_section_repeat_unit": record.repeat_unit,
    }
    return PlotRead(
        patterns=_patterns_from_lst_dec(record.lst_dec),
        strand=record.strand,
        read_id=record.read,
        extra=extra,
    )


def _patterns_from_lst_dec(lst_dec: str) -> tuple[tuple[str, int], ...]:
    motifs = [motif for motif in lst_dec.split(",") if motif]
    return tuple((motif, sum(1 for _ in group)) for motif, group in itertools.groupby(motifs))


def _merge_composite_panels(panels: Sequence[PlotPanel]) -> list[PlotPanel]:
    panels_by_repeat_id: dict[str, list[PlotPanel]] = defaultdict(list)
    for panel in panels:
        panels_by_repeat_id[panel.repeat_id].append(panel)

    merged_by_original_key: dict[str, PlotPanel] = {}
    consumed_keys: set[str] = set()
    for repeat_id, repeat_panels in panels_by_repeat_id.items():
        section_coords = {(panel.chrom, panel.start, panel.end) for panel in repeat_panels}
        if len(repeat_panels) < 2 or len(section_coords) < 2:
            continue
        merged_panel = _merge_one_composite(repeat_id, repeat_panels)
        merged_by_original_key[repeat_panels[0].key] = merged_panel
        consumed_keys.update(panel.key for panel in repeat_panels[1:])

    result: list[PlotPanel] = []
    for panel in panels:
        if panel.key in consumed_keys:
            continue
        result.append(merged_by_original_key.get(panel.key, panel))
    return result


def _merge_one_composite(repeat_id: str, section_panels: Sequence[PlotPanel]) -> PlotPanel:
    sorted_sections = sorted(section_panels, key=lambda panel: (panel.chrom, panel.start, panel.end))
    main_section_idx = next((idx for idx, panel in enumerate(sorted_sections) if _is_primary_panel(panel)), None)
    chrom = sorted_sections[0].chrom
    overall_start = min(panel.start for panel in sorted_sections)
    overall_end = max(panel.end for panel in sorted_sections)
    section_labels = [panel.repeat_unit for panel in sorted_sections]
    label_part = f" ({' + '.join(label for label in section_labels if label)})" if any(section_labels) else ""
    title = f"{repeat_id}{label_part} {chrom}:{overall_start}-{overall_end}"
    expected_order = (
        [main_section_idx, *(idx for idx in range(len(sorted_sections)) if idx != main_section_idx)]
        if main_section_idx is not None
        else list(range(len(sorted_sections)))
    )
    expected_motifs = tuple(
        _dedupe_preserving_order(
            motif for section_idx in expected_order for motif in expand_motif_representation(sorted_sections[section_idx].repeat_unit)
        )
    )
    repeat_unit_length = (
        sorted_sections[main_section_idx].repeat_unit_length if main_section_idx is not None else sorted_sections[0].repeat_unit_length
    )
    section_reads = [{read.read_id: read for read in panel.reads} for panel in sorted_sections]
    read_ids = _composite_read_ids(section_reads)
    return PlotPanel(
        key=_build_panel_key(repeat_id, chrom, overall_start, overall_end),
        title=title,
        reads=tuple(
            _merge_read_sections(
                read_id=read_id,
                section_reads=section_reads,
                section_labels=section_labels,
                title=title,
                main_section_idx=main_section_idx,
            )
            for read_id in read_ids
        ),
        expected_motifs=expected_motifs,
        repeat_unit_length=repeat_unit_length,
        repeat_id=repeat_id,
        chrom=chrom,
        start=overall_start,
        end=overall_end,
        variant_id=repeat_id,
        repeat_unit=", ".join(label for label in section_labels if label),
    )


def _merge_read_sections(  # noqa: C901
    *,
    read_id: str,
    section_reads: Sequence[Mapping[str, PlotRead]],
    section_labels: Sequence[str],
    title: str,
    main_section_idx: int | None,
) -> PlotRead:
    source_read = section_reads[main_section_idx].get(read_id) if main_section_idx is not None else None
    first_present_read: PlotRead | None = None
    patterns: list[tuple[str, int]] = []
    section_boundaries: list[int] = []
    cursor = 0
    is_spanning = True
    strand = "unknown"
    read_start: int | None = None
    mc_sites: str | None = None

    for section_idx, section_by_read in enumerate(section_reads):
        section_read = section_by_read.get(read_id)
        if section_read is None:
            continue
        if section_idx > 0:
            section_boundaries.append(cursor)
        if first_present_read is None:
            first_present_read = section_read
            strand = section_read.strand
        section_read_start = section_read.extra.get("read_start")
        if isinstance(section_read_start, int):
            read_start = section_read_start if read_start is None else min(read_start, section_read_start)
        section_mc_sites = section_read.extra.get("mc_sites")
        if isinstance(section_mc_sites, str):
            mc_sites = section_mc_sites if mc_sites is None else mc_sites + section_mc_sites
        patterns.extend(section_read.patterns)
        cursor += patterns_length(section_read.patterns)
        if not read_is_spanning(section_read):
            is_spanning = False

    whole_read_source = source_read or first_present_read
    extra: dict[str, object] = {
        "is_spanning": int(is_spanning),
        "section_boundaries": section_boundaries,
        "section_labels": list(section_labels),
        "composite_title": title,
        "mc_sites": mc_sites,
    }
    if whole_read_source is not None:
        source_extra = whole_read_source.extra
        for key in ("allele_id", "original_hap", "is_suspicious", "included_in_allele_size_estimation"):
            if key in source_extra:
                extra[key] = source_extra[key]
        if read_start is not None:
            extra["read_start"] = read_start
    return PlotRead(patterns=tuple(patterns), strand=strand, read_id=read_id, extra=extra)


def _composite_read_ids(section_reads: Sequence[Mapping[str, PlotRead]]) -> list[str]:
    read_order: dict[str, None] = {}
    for reads_by_id in section_reads:
        for read_id in reads_by_id:
            read_order.setdefault(read_id, None)
    return list(read_order)


def _build_panel_key(repeat_id: str, chrom: str, start: int, end: int) -> str:
    return f"{repeat_id}|{chrom}:{start}-{end}"


def _is_primary_panel(panel: PlotPanel) -> bool:
    return is_primary_output_locus(
        Locus(
            chrom=panel.chrom,
            start=panel.start,
            end=panel.end,
            motif=panel.repeat_unit,
            repeat_id=panel.repeat_id,
            variant_id=panel.variant_id,
        )
    )


def _panel_title(repeat_id: str, repeat_unit: str, chrom: str, start: int, end: int) -> str:
    motif_part = f" ({repeat_unit})" if repeat_unit else ""
    return f"{repeat_id}{motif_part} {chrom}:{start}-{end}"


def _dedupe_preserving_order(motifs: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for motif in motifs:
        if motif in seen:
            continue
        seen.add(motif)
        result.append(motif)
    return result

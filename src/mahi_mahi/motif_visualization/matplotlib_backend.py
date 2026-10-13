from __future__ import annotations

import atexit
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np

from mahi_mahi.pipeline_records import RegionCallSet
from mahi_mahi.utils import sanitize_plot_filename

from .adapter import build_plot_panels
from .core import nice_tick_step, prepare_panel
from .types import NonSpanningReadInfo, PlotPanel, PreparedPanel, RenderOptions, RenderReadRow, RenderRow, RGBColor

_PLOT_CACHE_TEMP_DIR: tempfile.TemporaryDirectory[str] | None = None
SECTION_BOUNDARY_MARKER_SIZE_POINTS = 3.0


@dataclass(frozen=True)
class _PlotLibs:
    plt: Any
    Patch: Any
    Path: Any
    Rectangle: Any


def write_motif_plot_pdfs(
    regions: Sequence[RegionCallSet],
    *,
    sample_name: str,
    output_dir: Path,
    options: RenderOptions | None = None,
    include_methylation: bool = False,
) -> list[Path]:
    panels = build_plot_panels(regions)
    if not panels:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    used_filenames: set[str] = set()
    for panel in panels:
        filename = _unique_pdf_filename(panel, sample_name, used_filenames)
        output_path = output_dir / filename
        _write_panel_pdf(
            panel,
            sample_name=sample_name,
            output_path=output_path,
            options=options or RenderOptions(),
            include_methylation=include_methylation and _panel_has_methylation(panel),
        )
        paths.append(output_path)
    return paths


def _panel_has_methylation(panel: PlotPanel) -> bool:
    return any(isinstance(sites := read.extra.get("mc_sites"), str) and sites for read in panel.reads)


def _write_panel_pdf(panel: PlotPanel, *, sample_name: str, output_path: Path, options: RenderOptions, include_methylation: bool) -> None:
    libs = _plot_libs()
    row_count = max(len(panel.reads), 15)
    panel_count = 2 if include_methylation else 1
    panel_height = min(11.69 / panel_count, max(3.5, 1.4 + row_count * 0.12))
    fig, axes = libs.plt.subplots(
        panel_count,
        1,
        figsize=(8.27, panel_height * panel_count),
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    _plot_panel(axes[0][0], panel, options)
    if include_methylation:
        _plot_panel(axes[1][0], panel, replace(options, segment_mode="methylation"), show_repeat_unit_axis=False)
    guide_text = "; black diamonds show composite TR section boundaries" if _is_composite_panel(panel) else ""
    description = "motif composition and CpG methylation" if include_methylation else "motif composition"
    fig.suptitle(
        f"{sample_name} - {description}. Each row is a read; red markers show non-spanning read ends{guide_text}.",
        fontsize=8,
        wrap=True,
    )
    fig.savefig(output_path, format="pdf")
    libs.plt.close(fig)


def _plot_panel(ax: object, panel: PlotPanel, options: RenderOptions, *, show_repeat_unit_axis: bool = True) -> None:
    ax_obj = cast(Any, ax)
    libs = _plot_libs()
    prepared = prepare_panel(panel, options)
    if prepared.status != "ok":
        assert prepared.message is not None
        _render_placeholder(ax_obj, panel.title, prepared.message)
        return

    num_rows = len(prepared.rows)
    motif_cols = prepared.axis.motif_cols
    xtick_step = nice_tick_step(motif_cols)
    ytick_step = nice_tick_step(num_rows)
    unknown_color = (0.5, 0.5, 0.5)

    _draw_render_segments(ax, prepared.rows, mask_suspicious=prepared.options.mask_suspicious, unknown_color=unknown_color)
    ax_obj.set_ylim(num_rows - 0.5, -0.5)

    x_left, x_right = prepared.axis.default_x_axis_limits
    x_range = x_right - x_left if x_right > x_left else 1.0
    sidebar_count = _sidebar_count(prepared.options)
    sidebar_width_bp = max(2.0, 0.01 * x_range)
    sidebar_gap_bp = max(5.0, 0.015 * x_range)
    left_edge_bp = x_left - sidebar_count * (sidebar_width_bp + sidebar_gap_bp)
    right_edge_bp = x_right + 0.1 * x_range
    ax_obj.set_xlim(left_edge_bp, right_edge_bp)

    x_ticks_pos = np.arange(0, right_edge_bp, xtick_step)
    x_ticks = np.unique(x_ticks_pos)
    ax_obj.set_xticks(x_ticks)
    ax_obj.set_xticklabels([str(int(v)) for v in x_ticks], fontsize=8)
    # undo the tick label hiding that sharex applies to all but the bottom panel
    ax_obj.tick_params(labelbottom=True)
    if prepared.axis.repeat_unit_length is not None and show_repeat_unit_axis:
        repeat_unit_length = prepared.axis.repeat_unit_length
        top_ax = ax_obj.secondary_xaxis(
            "top",
            functions=(
                lambda bp: bp / float(repeat_unit_length),
                lambda units: units * float(repeat_unit_length),
            ),
        )
        unit_ticks = _integer_repeat_unit_ticks(max(0.0, left_edge_bp), right_edge_bp, repeat_unit_length)
        top_ax.set_xticks(unit_ticks)
        top_ax.set_xticklabels([str(int(v)) for v in unit_ticks], fontsize=8)
        top_ax.set_xlabel("Approx. repeat units", fontsize=7, labelpad=6)

    ax_obj.set_yticks(np.arange(0, num_rows, ytick_step))
    ax_obj.set_yticklabels(np.arange(0, num_rows, ytick_step), fontsize=8)
    ax_obj.set_xlabel("Sequence length (bp)", fontsize=7)
    ax_obj.set_ylabel("Read rows", fontsize=7)
    panel_description = "CpG Methylation" if prepared.options.segment_mode == "methylation" else "Pattern Visualization"
    ax_obj.set_title(f"{panel.title} - {panel_description}", fontsize=7)

    _draw_sidebar_patches(ax, libs, prepared.rows, prepared.options, unknown_color, sidebar_width_bp=sidebar_width_bp, sidebar_gap_bp=sidebar_gap_bp)
    _draw_row_separators(ax, num_rows)
    if prepared.options.show_motif_boundaries:
        visible_span = prepared.axis.default_x_axis_limits[1] - prepared.axis.default_x_axis_limits[0]
        if visible_span <= 1000:
            _draw_motif_boundary_guides(ax, prepared.rows)
    _draw_section_boundaries(ax, prepared.rows)
    _draw_non_spanning_markers(ax, prepared.non_spanning_reads)
    _draw_split_separators(ax, prepared.separator_starts, prepared.options)
    _draw_legends(ax, libs, prepared, unknown_color)


def _draw_render_segments(ax: object, rows: Sequence[RenderRow], *, mask_suspicious: bool, unknown_color: RGBColor) -> None:
    ax_obj = cast(Any, ax)
    for row in rows:
        if not isinstance(row, RenderReadRow):
            continue
        bars: list[tuple[int, int]] = []
        colors: list[RGBColor] = []
        for segment in row.segments:
            width = segment.x_end - segment.x_start
            if width <= 0:
                continue
            bars.append((segment.x_start + row.display_offset, width))
            colors.append(unknown_color if mask_suspicious and row.is_suspicious else segment.color)
        if bars:
            ax_obj.broken_barh(bars, (row.row_index - 0.5, 1), facecolors=colors, edgecolors="none", linewidths=0)


def _draw_sidebar_patches(
    ax: object,
    libs: _PlotLibs,
    rows: Sequence[RenderRow],
    options: RenderOptions,
    unknown_color: RGBColor,
    sidebar_width_bp: float,
    sidebar_gap_bp: float,
) -> None:
    ax_obj = cast(Any, ax)
    sidebars: tuple[tuple[bool, Mapping[str, RGBColor], Any, str], ...] = (
        (options.show_haplotype_sidebar, options.haplotype_colors, lambda row: row.haplotype, "haplotype"),
        (options.show_allele_sidebar, options.allele_colors, lambda row: row.allele, "allele"),
        (options.show_strand_sidebar, options.strand_colors, lambda row: row.strand, "strand"),
    )
    x_cursor = -sum(1 for visible, _, _, _ in sidebars if visible) * (sidebar_width_bp + sidebar_gap_bp)
    for visible, colors, key_fn, label in sidebars:
        if not visible:
            continue
        ax_obj.text(
            x_cursor + sidebar_width_bp / 2.0,
            -0.012,
            label,
            transform=ax_obj.get_xaxis_transform(),
            rotation=90,
            ha="center",
            va="top",
            fontsize=5,
            clip_on=False,
        )
        for row in rows:
            if not isinstance(row, RenderReadRow):
                continue
            key = str(key_fn(row))
            color = colors.get(key, colors.get("0", unknown_color))
            rect = libs.Rectangle(
                (x_cursor, row.row_index - 0.5),
                sidebar_width_bp,
                1.0,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
            ax_obj.add_patch(rect)
        x_cursor += sidebar_width_bp + sidebar_gap_bp


def _sidebar_count(options: RenderOptions) -> int:
    return sum((options.show_strand_sidebar, options.show_haplotype_sidebar, options.show_allele_sidebar))


def _draw_row_separators(ax: object, num_rows: int) -> None:
    ax_obj = cast(Any, ax)
    if num_rows <= 1:
        return
    for row_index in range(1, num_rows):
        ax_obj.axhline(y=row_index - 0.5, color="white", linewidth=0.35, alpha=0.25)


def _draw_motif_boundary_guides(ax: object, rows: Sequence[RenderRow]) -> None:
    ax_obj = cast(Any, ax)
    x_left, x_right = ax_obj.get_xlim()
    for row in rows:
        if not isinstance(row, RenderReadRow):
            continue
        boundaries = [boundary + row.display_offset for boundary in row.motif_boundaries if x_left <= boundary + row.display_offset <= x_right]
        if boundaries:
            ax_obj.vlines(
                x=boundaries,
                ymin=row.row_index - 0.5,
                ymax=row.row_index + 0.5,
                colors="#111111",
                linewidth=0.45,
                alpha=0.35,
            )


def _draw_section_boundaries(ax: object, rows: Sequence[RenderRow]) -> None:
    ax_obj = cast(Any, ax)
    libs = _plot_libs()
    marker = libs.Path([(-0.3, 0.0), (0.0, -0.5), (0.3, 0.0), (0.0, 0.5), (-0.3, 0.0)])
    for row in rows:
        if not isinstance(row, RenderReadRow):
            continue
        for boundary in row.section_boundaries:
            x = float(int(boundary) + row.display_offset)
            ax_obj.plot(
                [x],
                [row.row_index],
                linestyle="None",
                marker=marker,
                markersize=SECTION_BOUNDARY_MARKER_SIZE_POINTS,
                markerfacecolor="black",
                markeredgewidth=0,
                alpha=0.75,
                zorder=5,
                clip_on=False,
            )


def _is_composite_panel(panel: PlotPanel) -> bool:
    return any(read.extra.get("section_boundaries") for read in panel.reads)


def _draw_non_spanning_markers(ax: object, non_spanning_reads: Sequence[NonSpanningReadInfo]) -> None:
    ax_obj = cast(Any, ax)
    if not non_spanning_reads:
        return
    half_h = 0.4
    x_lo, x_hi = ax_obj.get_xlim()
    chevron_w = max(1.0, 0.01 * (x_hi - x_lo))
    for info in non_spanning_reads:
        cy = float(info.row_index)
        if info.end == "left":
            base_x = float(info.display_offset)
            xs = [base_x, base_x - chevron_w, base_x]
        else:
            base_x = float(info.display_offset + info.read_length)
            xs = [base_x, base_x + chevron_w, base_x]
        ax_obj.fill(xs, [cy - half_h, cy, cy + half_h], color="#c23b22", alpha=0.85, zorder=3)


def _draw_split_separators(ax: object, separator_starts: Sequence[int], options: RenderOptions) -> None:
    ax_obj = cast(Any, ax)
    if not separator_starts:
        return
    separator_rows = {
        "haplotype": options.haplotype_separator_rows,
        "strand": options.strand_separator_rows,
        "allele": options.allele_separator_rows,
        "none": 0,
    }[options.split_by]
    if separator_rows <= 0:
        return
    for separator_start in separator_starts:
        center_y = separator_start + (separator_rows / 2.0) - 0.5
        ax_obj.axhline(y=center_y, color="black", linestyle="--", linewidth=0.6, alpha=0.7)


def _draw_legends(ax: object, libs: _PlotLibs, prepared: PreparedPanel, unknown_color: RGBColor) -> None:
    ax_obj = cast(Any, ax)
    if prepared.options.show_motif_legend and (
        prepared.legends.pattern_colors
        or (prepared.legends.motif_filter_active and prepared.options.show_other_in_legend)
        or prepared.legends.masked_suspicious_present
    ):
        motif_handles = [libs.Patch(facecolor=color, label=motif) for motif, color in prepared.legends.pattern_colors.items()]
        if prepared.legends.motif_filter_active and prepared.options.show_other_in_legend:
            motif_handles.append(libs.Patch(facecolor=prepared.options.motif_other_color, label=prepared.options.motif_other_label))
        if prepared.legends.masked_suspicious_present:
            motif_handles.append(libs.Patch(facecolor=unknown_color, label="Masked suspicious"))
        legend_title = "Methylation" if prepared.options.segment_mode == "methylation" else "Pattern\n(main plot)"
        motif_legend = ax_obj.legend(
            handles=motif_handles,
            title=legend_title,
            loc="upper right",
            ncol=1,
            frameon=True,
            fontsize=4,
            title_fontsize=5,
        )
        ax_obj.add_artist(motif_legend)

    sidebar_handles = _sidebar_legend_handles(libs, prepared, unknown_color)
    if sidebar_handles:
        sidebar_legend = ax_obj.legend(
            handles=sidebar_handles,
            title="Sidebar(s)",
            loc="lower right",
            ncol=1,
            frameon=True,
            fontsize=4,
            title_fontsize=5,
        )
        ax_obj.add_artist(sidebar_legend)


def _sidebar_legend_handles(libs: _PlotLibs, prepared: PreparedPanel, unknown_color: RGBColor) -> list[Any]:
    handles: list[Any] = []
    if prepared.options.show_strand_sidebar:
        handles.extend(
            _legend_section(
                libs,
                "Strand",
                prepared.options.strand_colors,
                prepared.legends.present_strands,
                {"+": "Positive (+)", "-": "Negative (-)", "unknown": "Unknown"},
                unknown_color,
            )
        )
    if prepared.options.show_haplotype_sidebar:
        handles.extend(
            _legend_section(
                libs,
                "Haplotype",
                prepared.options.haplotype_colors,
                prepared.legends.present_haplotypes,
                {"1": "Haplotype 1", "2": "Haplotype 2", "0": "Unknown"},
                unknown_color,
            )
        )
    if prepared.options.show_allele_sidebar:
        handles.extend(
            _legend_section(
                libs,
                "Allele",
                prepared.options.allele_colors,
                prepared.legends.present_alleles,
                {"1": "Allele 1", "2": "Allele 2", "0": "Unknown"},
                unknown_color,
            )
        )
    return handles


def _legend_section(
    libs: _PlotLibs,
    title: str,
    colors: Mapping[str, RGBColor],
    present_keys: set[str],
    labels: Mapping[str, str],
    unknown_color: RGBColor,
) -> list[Any]:
    if not present_keys:
        return []
    handles = [libs.Patch(facecolor="none", edgecolor="none", label=title)]
    for key, label in labels.items():
        if key not in present_keys:
            continue
        handles.append(libs.Patch(facecolor=colors.get(key, unknown_color), label=f"  {label}"))
    return handles


def _integer_repeat_unit_ticks(left_bp: float, right_bp: float, repeat_unit_length: int, target_tick_count: int = 7) -> np.ndarray[Any, Any]:
    if repeat_unit_length <= 0 or right_bp <= left_bp:
        return np.array([], dtype=float)
    left_units = left_bp / float(repeat_unit_length)
    right_units = right_bp / float(repeat_unit_length)
    unit_span = max(int(np.ceil(right_units - left_units)), 1)
    unit_step = nice_tick_step(unit_span, target_tick_count)
    first_tick = int(np.ceil(left_units / unit_step) * unit_step)
    last_tick = int(np.floor(right_units / unit_step) * unit_step)
    if last_tick < first_tick:
        return np.array([], dtype=float)
    return np.arange(first_tick, last_tick + unit_step, unit_step, dtype=float)


def _render_placeholder(ax: object, title: str, message: str) -> None:
    ax_obj = cast(Any, ax)
    ax_obj.text(0.5, 0.5, message, fontsize=10, ha="center", va="center")
    ax_obj.set_title(f"{title} - Pattern Visualization", fontsize=7)
    ax_obj.axis("off")


def _unique_pdf_filename(panel: PlotPanel, sample_name: str, used_filenames: set[str]) -> str:
    base = sanitize_plot_filename(f"{panel.repeat_id}_motifs_{sample_name}") or "motif_plot"
    filename = f"{base}.pdf"
    if filename not in used_filenames:
        used_filenames.add(filename)
        return filename
    coord_base = sanitize_plot_filename(f"{panel.repeat_id}_{panel.chrom}_{panel.start}_{panel.end}_motifs_{sample_name}") or "motif_plot"
    filename = f"{coord_base}.pdf"
    suffix = 2
    while filename in used_filenames:
        filename = f"{coord_base}_{suffix}.pdf"
        suffix += 1
    used_filenames.add(filename)
    return filename


@lru_cache(maxsize=1)
def _plot_libs() -> _PlotLibs:
    _configure_plot_cache()
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.path import Path as MplPath

    return _PlotLibs(plt=plt, Patch=Patch, Path=MplPath, Rectangle=Rectangle)


def _configure_plot_cache() -> None:
    global _PLOT_CACHE_TEMP_DIR

    if "MPLCONFIGDIR" in os.environ and "XDG_CACHE_HOME" in os.environ:
        return
    if _PLOT_CACHE_TEMP_DIR is None:
        _PLOT_CACHE_TEMP_DIR = tempfile.TemporaryDirectory(prefix="mahi-mahi-motif-plot-cache-")
        atexit.register(_cleanup_plot_cache_temp_dir)

    cache_root = Path(_PLOT_CACHE_TEMP_DIR.name)
    mpl_cache_dir = cache_root / "matplotlib"
    xdg_cache_home = cache_root / "xdg"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    (xdg_cache_home / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.fspath(mpl_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", os.fspath(xdg_cache_home))


def _cleanup_plot_cache_temp_dir() -> None:
    global _PLOT_CACHE_TEMP_DIR

    if _PLOT_CACHE_TEMP_DIR is None:
        return
    _PLOT_CACHE_TEMP_DIR.cleanup()
    _PLOT_CACHE_TEMP_DIR = None

import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from mahi_mahi.kde_peak_clusterer import KDEPeakClusterer

# Suppress DEBUG logs from matplotlib.font_manager
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)


class SwarmPlotPoints:
    """
    A helper class to calculate non-overlapping y positions for points in a swarm plot style.

    This is O(n^2) but we only have ~30 points so we don't care.

    This is used to plot individual reads on top of the KDE, without them overlapping each other too much.
    """

    def __init__(self, spacing_x: float, spacing_y: float, max_x_offset: float = 0.3, max_y: float = np.inf) -> None:
        """Initialize the SwarmPlotPoints instance.
        Args:
            spacing_x (float): Minimum horizontal spacing between points.
            spacing_y (float): Minimum vertical spacing between points.
            max_x_offset (float): Maximum horizontal offset to try when vertical space is exceeded.
               The default of 0.3 is chosen to allow integer data to still be read exactly.
            max_y (float): Maximum y value to assign before giving up and allowing overlap."""
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.max_x_offset = max_x_offset
        self.last_x_offset = 0
        self.max_y = max_y
        self.points: list[tuple[float, float]] = []

    def _find_vacant_y(self, x: float) -> float:
        """
        Find a vacant y position for a point at a given x to avoid overlap.

        Args:
            x (float): The x-coordinate of the new point.

        Returns:
            float: The y-coordinate for the new point.
        """
        neighbors_ymax = max((existing_y for existing_x, existing_y in self.points if abs(existing_x - x) <= self.spacing_x), default=-self.spacing_y)
        return neighbors_ymax + self.spacing_y

    def add_point(self, x: float) -> tuple[float, float]:
        """
        Add a point at a given x, calculating a non-overlapping y position.

        Args:
            x (float): The x-coordinate of the new point.
        Returns:
            tuple: The (x, y) coordinates of the new point.
        """
        y = self._find_vacant_y(x)
        if y > self.max_y and abs(1.01 * self.spacing_x) < self.max_x_offset:
            # out of vertical space, try shifting left/right to find more room
            # alternate left and right sides
            x_offset = -1.01 * self.spacing_x if self.last_x_offset > 0 else 1.01 * self.spacing_x
            self.last_x_offset = x_offset
            x += x_offset
        y = self._find_vacant_y(x)
        if y > self.max_y:
            # If we still have too many points, give up and let them overlap
            y = self.max_y * np.random.rand()
        self.points.append((x, y))
        return x, y


RGBAColor = tuple[float, float, float, float]

MARKER_STYLES = ["d", "s", "^", "v", "D", "o", "<", ">", "*", "P", "p", "h", "8"]


def _plot_density_and_peaks(clusterer: "KDEPeakClusterer", *, plot_first_attempt_density: bool = False) -> None:
    """Plot the final density estimate, detected peaks and cluster centres."""
    plt.plot(clusterer.x_vals, clusterer.density, label="Density estimate", color="blue")
    if plot_first_attempt_density and clusterer.first_attempt_density.size > 0:
        plt.plot(clusterer.first_attempt_x_vals, clusterer.first_attempt_density, label="First-run (rejected) density estimate", color="orange")

    for peak_idx in clusterer.peaks_positions:
        peak_x = clusterer.x_vals[peak_idx]
        peak_y = clusterer.density[peak_idx]
        plt.plot(peak_x, peak_y, "mx", zorder=3)
        plt.axvline(peak_x, linestyle="--", color="m", zorder=3)  # type: ignore
    # legend
    plt.plot([], [], "--mx", label="Detected peaks", zorder=3)

    for i, cluster in enumerate(clusterer.clusters):
        # Only add the label for the first final cluster to avoid duplicate legend entries
        plt.axvline(cluster.center, linestyle="-", color="k", label="Final clusters" if i == 0 else None, zorder=2.5)

    plt.xlabel("TR length (bases)")
    plt.ylabel("Density")
    plt.grid(True)


def _build_marker_map(unique_haps: NDArray[np.str_]) -> dict[str, str]:
    """Assign a marker style to each haplotype label.

    The shifted start is to have consistent markers between regions with and without unphased reads.
    """
    return {hap: MARKER_STYLES[i % len(MARKER_STYLES)] for i, hap in enumerate(unique_haps, start=0 if "ps0_0" in unique_haps else 1)}


def _build_color_map(unique_labels: NDArray[np.int32]) -> dict[int, RGBAColor]:
    """Assign a colour to each cluster label.

    The shifted start is to have consistent colours between regions with and without unclustered reads.
    """
    color_map = plt.get_cmap("tab10")
    return {lab: color_map(i % 10) for i, lab in enumerate(unique_labels, start=0 if -1 in unique_labels else 1)}


def _plot_reads(
    clusterer: "KDEPeakClusterer",
    haps: list[str],
    is_filled: NDArray[np.bool_],
    marker_map: dict[str, str],
    color_dict: dict[int, RGBAColor],
    plot_first_attempt_density: bool,
) -> SwarmPlotPoints:
    """Plot one point per read, coloured by cluster and shaped by haplotype, and return their placement."""
    max_y = np.max(clusterer.density) if clusterer.density.size > 0 else 1
    if plot_first_attempt_density and clusterer.first_attempt_density.size > 0:
        max_y = max(max_y, np.max(clusterer.first_attempt_density))
    swarm_points = SwarmPlotPoints(
        spacing_x=(np.max(clusterer.tr_lengths) - np.min(clusterer.tr_lengths)) / 50 if len(clusterer.tr_lengths) > 1 else 1,
        spacing_y=max_y / 25,
        max_x_offset=0.3,
        max_y=max_y,
    )

    for idx in np.arange(len(clusterer.tr_lengths)):
        this_x, this_y = swarm_points.add_point(clusterer.tr_lengths[idx])
        plt.plot(
            this_x,
            this_y,
            label=None,  # We'll handle legends manually
            marker=marker_map[haps[idx]],
            color=color_dict[clusterer.labels[idx]],
            fillstyle="full" if is_filled[idx] else "none",
        )
    return swarm_points


def _plot_legend_entries(
    unique_labels: NDArray[np.int32],
    unique_haps: NDArray[np.str_],
    marker_map: dict[str, str],
    color_dict: dict[int, RGBAColor],
    is_filled_labels: tuple[str, str],
) -> None:
    """Add the (otherwise missing) legend entries for clusters, haplotypes and fill state."""
    for lab in unique_labels:
        # the +1 is because internal labels are 0-based and user-facing alleles are 1-based
        plt.plot([], [], "o", color=color_dict[lab], label="Unclustered reads" if lab == -1 else f"Cluster {lab + 1} reads")
    for hap in unique_haps:
        plt.plot([], [], marker_map[hap], color="gray", label=f"Haplotype {hap}")
    plt.plot([], [], "o", color="gray", fillstyle="none", label=is_filled_labels[0])
    plt.plot([], [], "o", color="gray", fillstyle="full", label=is_filled_labels[1])


def _build_title(shape: str, sample_name: str | None, region_name: str | None, points_shifted: bool) -> str:
    """Build the plot title describing the clustering and how to read the points."""
    title = f"Shape detected {shape.capitalize()} Clusters"
    if sample_name:
        title += f", Sample: {sample_name}"
    if region_name:
        title += f", Region: {region_name}"
    title += "\nPoints are reads used in clustering, colour = cluster, shape = haplotype"
    if points_shifted:
        title += "\nFor clarity, some points are slightly shifted left/right to avoid overlap"
    return title


class KDEPeakClustererWithPlotting(KDEPeakClusterer):
    def plot_KDE(
        self: "KDEPeakClusterer",
        *,
        tr_ps_hap_labels: list[str],
        is_filled: NDArray[np.bool_],
        is_filled_labels: tuple[str, str],
        filename: Path | None = None,
        sample_name: str | None = None,
        region_name: str | None = None,
        plot_first_attempt_density: bool = False,
    ) -> None:
        """
        Plot the KDE and detected peaks.

        Args:
            tr_ps_hap_labels (list of str): List of string labels (e.g., "ps1_1").
                These determine the marker shape.
            is_filled (array of bool): whether to fill this read's point.
                May be used to represent e.g. spanning vs non-spanning or as-expected vs not-as-expected.
            is_filled_labels (tuple of str): Legend entries for unfilled and filled points (unfilled first).
            filename (Path | None): Filename to save the plot.
            sample_name (str | None): Sample name.
            region_name (str | None): Region name.
            plot_first_attempt_density (bool): Include the density estimate superseded by a fallback fit.
        """
        # Ensure region_name is a string or None
        region_name = str(region_name) if region_name is not None else None
        plt.figure(figsize=(8, 4), layout="constrained")
        _plot_density_and_peaks(self, plot_first_attempt_density=plot_first_attempt_density)

        # Plot points for individual reads, colored by cluster and shaped by ps_hap

        haps = [str(lbl) for lbl in tr_ps_hap_labels]  # make sure we don't have NaNs
        unique_haps = np.unique(haps)
        unique_labels = np.unique(self.labels)
        marker_map = _build_marker_map(unique_haps)
        color_dict = _build_color_map(unique_labels)

        swarm_points = _plot_reads(self, haps, is_filled, marker_map, color_dict, plot_first_attempt_density)

        _plot_legend_entries(unique_labels, unique_haps, marker_map, color_dict, is_filled_labels)
        plt.title(_build_title(self.shape, sample_name, region_name, swarm_points.last_x_offset != 0), fontsize="medium")

        plt.legend()

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

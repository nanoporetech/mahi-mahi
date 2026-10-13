import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.stats
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.signal import find_peaks


class FitResult(Enum):
    """How one bandwidth's fit went.

    SUCCESS: the clusters can be trusted over any haplotags.
    SOFT_FAIL: clusters were found but are not trustworthy; they are only used if there are no usable haplotags.
    FAIL: no reads could be assigned at all.
    """

    SUCCESS = "success"
    SOFT_FAIL = "soft_fail"
    FAIL = "fail"


class ClusterInfo(BaseModel):
    """
    Represents information about a cluster.

    Attributes:
        center (int): The center of the cluster.
        range (Tuple[int, int]): The range of the cluster (min, max).
        is_unstable (bool): Indicates whether the cluster is unstable.
    """

    center: int
    range: tuple[int, int]
    is_unstable: bool

    model_config = ConfigDict(frozen=True)


@dataclass
class _InitialPeaksResult:
    """Return type of KDEPeakClusterer._find_initial_peaks."""

    tr_lengths: NDArray[np.float64]
    x_vals: NDArray[np.float64]
    density: NDArray[np.float64]
    peaks_positions: NDArray[np.int32]
    peaks_values: NDArray[np.float64]
    # if you add new fields to this class, they may need copying to the main KDEPeakClusterer (e.g. in fit())


@dataclass
class _CombineResult:
    """Return type of KDEPeakClusterer._combine_into_labels."""

    labels: NDArray[np.int32]
    shape: str
    soft_failed: bool


class KDEPeakClusterer:
    """
    Class for performing KDE-based peak detection and clustering on read length data.

    Consider running pytest -m "extensive_clustering_testing" after modifying this class.

    Methods:
        fit(read_lengths): Perform KDE, detect peaks, and assign clusters.

    Description:
        The algorithm fits the read lengths using a KDE (Kernel Density Estimation) to create a density
        function and detects its peaks. Each read is assigned to its closest peak if it is within
        ``typical_peak_width(length) * peak_assignment_width_factor`` of it, otherwise it is left
        unassigned for now.

        The peaks with enough reads to be valid peaks are then counted.  If there are two or more,
        the two best-supported peaks (i.e. those with the most reads assigned) are retained;
        if fewer than two are valid, the single best-supported peak is used.

        Reads are then reassigned to the selected peaks.  If there were exactly two valid peaks,
        reads outside the normal assignment width may also be assigned to a peak, using the smaller
        of a multiple of the normal assignment width and a fraction of the spacing between the peaks.

        If there are two peaks
        that are very close together and also look suspicious (very unbalanced and/or many reads fairly
        close by but not very close to either of the peaks), the clustering fails.

        If many reads are still unassigned and clearly longer than the lower ("bottom") peak, they are
        gathered into a single "wide peak".  This is intended to handle both 3-peaked cases, and
        2-peaked cases where the second peak is wider than the main assignment step will accept.
        - If there are two main peaks well apart, the wide peak is merged into the upper peak (i.e. is
        assumed to be a somatic expansion of the longer allele).
        - If there are two main peaks close together (i.e. too close for that to be a safe assumption)
        and a wide peak, clustering soft-fails: the bottom peak and the wide peak are kept as the two
        alleles and the reads of the upper close peak, which could belong to either, are left unassigned.
        - If there is only one main peak and a wide peak, the wide peak is kept as a separate cluster
        (i.e. is assumed to be the other allele).

        If there are still many reads unassigned after this, clustering fails: hard if that leaves a single
        allele, soft if it leaves two.  (The reasoning behind this is that two dubious peaks are more likely
        to be a substantial improvement over using all the reads (as happens after a hard fail), while
        either of them can be worse than that (if there are enough unassigned reads that the chosen peak(s)
        aren't really representative), but this is currently a guess.)

        A soft failure keeps its clusters but clears ``is_confident_clustering``, so the caller retries at
        the fallback bandwidth and prefers haplotags where there are any, and only falls back on these
        clusters when there are none."""

    def __init__(
        self: "KDEPeakClusterer",
        bandwidth: float = 0.5,
        prominence_factor: float = 0.02,
        dominance_threshold: float = 0.85,
        min_peak_separation: int = 3,
        min_unchecked_peak_separation: int = 15,
        check_min_minority_fraction: float = 0.25,
        check_min_total_fraction: float = 0.0,  # disable by default
        min_small_cluster_size: int = 3,
        expanded_reads_ratio_threshold_one_allele: float = 0.1,
        peak_assignment_width_factor: float = 7.0,
        widening_assignment_max_factor: float = 3.0,
        widening_assignment_max_fraction_of_spacing: float = 1 / 3,
        fallback_bandwidth: float = 3,
        wide_peak_min_absolute_offset: float = 100.0,
        wide_peak_relative_threshold: float = 2.0,
        well_apart_absolute_offset: float = 50.0,
        well_apart_relative: float = 1.5,
        min_assigned_fraction: float = 0.85,
        clearly_separated_min_distance_over_halfwidth: float = 4.0,
    ) -> None:
        """
        Initializes the clustering object with configurable parameters for density-based clustering and peak detection.

        Parameters:
            bandwidth (float): Bandwidth parameter for density estimation, as a multiple of typical_peak_width(length).
            prominence_factor (float): Minimum prominence of peaks, as a fraction of the maximum density.
            dominance_threshold (float): Peaks must contain at least (1 - dominance_threshold) of the reads.
            min_peak_separation (int): Two peaks closer than this (in bases) are always treated as a single
              allele; peaks at least this far apart but closer than ``min_unchecked_peak_separation`` are only
              accepted if they pass the ``_is_two_clearly_separated_alleles`` check.
            min_unchecked_peak_separation (int): Two peaks at least this far apart (in bases) are accepted as
              separate alleles without the ``_is_two_clearly_separated_alleles`` check.
            check_min_minority_fraction (float): In the ``_is_two_clearly_separated_alleles`` check, each peak
              must hold at least this fraction of the reads near the pair when the peaks are exactly
              ``min_peak_separation`` apart, so very unbalanced peaks are rejected. For peaks further apart the
              required fraction falls towards ``1 - dominance_threshold`` (see ``min_fraction_per_peak``).
            check_min_total_fraction (float): In the ``_is_two_clearly_separated_alleles`` check, the two peaks
              together must hold at least this fraction of the reads near the pair, so pairs with many reads
              between or around them (rather than at the peaks) are rejected.
            min_small_cluster_size (int): Minimum number of reads in each peak.
            expanded_reads_ratio_threshold_one_allele (float): Minimum fraction of reads that must fall in the
              wide peak for it to be created.
            peak_assignment_width_factor (float): A read is assigned to its closest peak only if it is within
              ``typical_peak_width(length) * peak_assignment_width_factor`` of that peak.
            widening_assignment_max_factor (float): For two peaks, the maximum distance for assigning an
              otherwise-unassigned read is at most the normal per-read assignment width multiplied by this value.
            widening_assignment_max_fraction_of_spacing (float): For two peaks, the maximum distance for assigning an
              otherwise-unassigned read is also at most this fraction of the distance between the peaks.
            fallback_bandwidth (float): If clustering fails, it is re-attempted with this bandwidth.
              If this is smaller than the main bandwidth, it is also re-attempted if it found a single or single-plus-wide peaks.
            wide_peak_min_absolute_offset (float): Absolute part of the wide-peak length cutoff
              ``max(bottom_peak + offset, bottom_peak * wide_peak_relative_threshold)``.
            wide_peak_relative_threshold (float): Relative part of the wide-peak length cutoff (see above).
            well_apart_absolute_offset (float): Absolute part of the "well apart" test
              ``upper_peak > max(bottom_peak + offset, bottom_peak * well_apart_relative)``.
            well_apart_relative (float): Relative part of the "well apart" test (see above).
            min_assigned_fraction (float): Clustering fails if fewer than this fraction of reads end up assigned.
            clearly_separated_min_distance_over_halfwidth (float): In the ``_is_two_clearly_separated_alleles``
              check, a read counts as "near" a peak if it is within ``distance_between_peaks`` divided by this of
              the peak. Larger values make "near" narrower.
        """
        self.bandwidth = bandwidth
        self.pad_points = 5
        self.prominence_factor = prominence_factor
        self.dominance_threshold = dominance_threshold
        self.min_peak_separation = min_peak_separation
        self.min_unchecked_peak_separation = min_unchecked_peak_separation
        self.check_min_minority_fraction = check_min_minority_fraction
        self.check_min_total_fraction = check_min_total_fraction
        self.min_small_cluster_size = min_small_cluster_size
        self.expanded_reads_ratio_threshold_one_allele = expanded_reads_ratio_threshold_one_allele
        self.peak_assignment_width_factor = peak_assignment_width_factor
        self.fallback_bandwidth = fallback_bandwidth
        self.wide_peak_min_absolute_offset = wide_peak_min_absolute_offset
        self.wide_peak_relative_threshold = wide_peak_relative_threshold
        self.well_apart_absolute_offset = well_apart_absolute_offset
        self.well_apart_relative = well_apart_relative
        self.min_assigned_fraction = min_assigned_fraction
        self.clearly_separated_min_distance_over_halfwidth = clearly_separated_min_distance_over_halfwidth
        self.widening_assignment_max_factor = widening_assignment_max_factor
        self.widening_assignment_max_fraction_of_spacing = widening_assignment_max_fraction_of_spacing
        self.x_vals = np.array([])
        self.density = np.array([])
        self.split_x_vals = np.array([])
        self.first_attempt_x_vals = np.array([])
        self.first_attempt_density = np.array([])
        self.peaks_positions = np.array([])
        self.peaks_values = np.array([])
        self.plot = False
        self.shape = "unknown"

        ## output
        self.labels = np.array([], dtype=np.int32)
        self.clusters: list[ClusterInfo] = []
        self.is_confident_clustering = True

    def typical_peak_width(self: "KDEPeakClusterer", length: "float | NDArray[np.float64]") -> "np.float64 | NDArray[np.float64]":
        """Standard deviation expected of a typical non-expanded peak of this length.

        Accepts a scalar or an array of lengths.
        """
        # must be non-zero everywhere to avoid division by zero
        return np.maximum(0.5, np.asarray(length, dtype=np.float64) * 0.03)

    def _find_initial_peaks(
        self: "KDEPeakClusterer",
        read_lengths: NDArray[np.float64],
        bandwidth_factor: float,
    ) -> _InitialPeaksResult:
        """
        Perform KDE and initial peak detection.

        This does not modify the object's state: it is a pure helper that returns its results, which the caller
        (``fit`` via ``_fit_this_bandwidth``) assigns to the object.

        Args:
            read_lengths (NDArray[np.float64]): Array of read lengths.
            bandwidth_factor (float): Multiplier applied to the per-read kernel width.

        Returns:
            _InitialPeaksResult: The detected peaks and closest peak indices.
        """
        # Ensure read_lengths is a numpy array
        read_lengths = np.array(read_lengths, dtype=np.float64)
        tr_lengths = read_lengths.flatten()

        data_min, data_max = tr_lengths.min(), tr_lengths.max()
        # If all values are the same, don't run KDE (it would crash)
        if len(set(tr_lengths)) == 1:
            return _InitialPeaksResult(
                tr_lengths=tr_lengths,
                x_vals=np.array([tr_lengths[0]], dtype=np.float64),
                density=np.array([len(tr_lengths)], dtype=np.float64),
                peaks_positions=np.array([0]),
                peaks_values=np.array([tr_lengths[0]], dtype=np.float64),
            )
        # using integer x_vals means we don't see peaks 1 base apart as separate,
        # which is intentional because these are probably sequencing error
        x_vals = np.arange(data_min - self.pad_points, data_max + self.pad_points + 1)
        # we don't use scipy.stats.gaussian_kde because we want bandwidth to be set by the prior, not the data's SD
        # this isn't quite a density because the extra factor of self.typical_peak_width(length) is included
        # to make peaks with the same number of reads but very different lengths about the same height
        # (needed for the prominence check in find_peaks)
        density = sum(
            (
                scipy.stats.norm.pdf(x_vals, loc=length, scale=self.typical_peak_width(length) * bandwidth_factor) * self.typical_peak_width(length)
                for length in tr_lengths
            ),
            start=np.zeros_like(x_vals, dtype=np.float64),
        )

        # Pad density for peak detection at edges
        density_padded = np.pad(density, self.pad_points, mode="constant", constant_values=0)
        prominence = self.prominence_factor * np.max(density_padded)
        peaks, _ = find_peaks(density_padded, prominence=prominence)
        peaks = peaks - self.pad_points  # Remove padding offset

        peaks_positions = peaks
        peaks_values = x_vals[peaks]
        return _InitialPeaksResult(
            tr_lengths=tr_lengths,
            x_vals=x_vals,
            density=density,
            peaks_positions=peaks_positions,
            peaks_values=peaks_values,
        )

    def _assign_reads_to_peaks(
        self: "KDEPeakClusterer",
        reads: NDArray[np.float64],
        peaks: NDArray[np.float64],
        widths: NDArray[np.float64],
    ) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
        """Assign each read to its closest peak, and report which reads are within the assignment width.

        Args:
            reads (NDArray[np.float64]): Read lengths.
            peaks (NDArray[np.float64]): Peak positions.
            widths (NDArray[np.float64]): Per-read ``typical_peak_width``.

        Returns:
            tuple: (closest_peak_index_per_read, within_assignment_width_mask).
        """
        distances = np.abs(reads[:, None] - peaks[None, :])
        closest = np.argmin(distances, axis=1).astype(np.int32)
        closest_distance = distances[np.arange(len(reads)), closest]
        within = closest_distance <= widths * self.peak_assignment_width_factor
        return closest, within

    def _assign_additional_reads_to_two_peaks(
        self: "KDEPeakClusterer",
        reads: NDArray[np.float64],
        peaks: NDArray[np.float64],
        widths: NDArray[np.float64],
        peak_of_read: NDArray[np.int32],
        assigned: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
        """Assign currently unassigned reads to one of two peaks using an expanded width.

        Existing assignments are never removed, even when an expanded width is narrower than the normal assignment width.
        """
        distances = np.abs(reads[:, None] - peaks[None, :])
        closest = np.argmin(distances, axis=1).astype(np.int32)
        closest_distance = distances[np.arange(len(reads)), closest]
        distance_between_peaks = abs(float(peaks[1] - peaks[0]))
        max_distance = np.minimum(
            widths * self.peak_assignment_width_factor * self.widening_assignment_max_factor,
            distance_between_peaks * self.widening_assignment_max_fraction_of_spacing,
        )
        additional = ~assigned & (closest_distance <= max_distance)
        logging.debug(f"widening assignment width to {max_distance} assigned {np.sum(additional)} additional reads")

        peak_of_read = peak_of_read.copy()
        assigned = assigned.copy()
        peak_of_read[additional] = closest[additional]
        assigned[additional] = True
        return peak_of_read, assigned

    def _fail(self: "KDEPeakClusterer", shape: str) -> None:
        """Record a failed clustering: every read unclustered, no clusters."""
        self.labels = np.full(len(self.tr_lengths), -1, dtype=np.int32)
        self.clusters = []
        self.shape = shape
        self.is_confident_clustering = False

    def fit(
        self: "KDEPeakClusterer",
        read_lengths: NDArray[np.float64],
    ) -> NDArray[np.int32]:
        """
        Perform KDE, detect peaks, and assign cluster labels.

        Args:
            read_lengths (NDArray[np.float64]): Array of read lengths.

        Returns:
            NDArray[np.int32]: Cluster labels for each read.

        Notes:
            Cluster labels are 0 and 1 for the two clusters, and -1 for unclustered reads.
            Label 0 is always the lower-length cluster.
            A soft-failed clustering also has labels: see ``is_confident_clustering``.
        """
        if len(self.labels) != 0:
            # this function does not clear internal state, so we don't allow re-fitting the same instance
            raise RuntimeError("fit() has already been called. Please create a new instance to fit again.")
        result = self._fit_this_bandwidth(read_lengths, bandwidth_factor=self.bandwidth)
        try_fallback = result is not FitResult.SUCCESS
        if self.fallback_bandwidth < self.bandwidth:
            # also attempt splitting single or single-plus-wide peaks
            try_fallback = try_fallback or len(self.clusters) < 2 or self.shape == "one_peak_one_wide"
        if try_fallback:
            description = {FitResult.FAIL: "failed", FitResult.SOFT_FAIL: "soft-failed", FitResult.SUCCESS: "gave single main peak"}[result]
            logging.debug(f"clustering {description} at bandwidth {self.bandwidth}, retrying with fallback bandwidth {self.fallback_bandwidth}")
            self.first_attempt_x_vals = self.x_vals
            self.first_attempt_density = self.density
            self._fit_this_bandwidth(read_lengths, bandwidth_factor=self.fallback_bandwidth)
        return self.labels

    def _fit_this_bandwidth(
        self: "KDEPeakClusterer",
        read_lengths: NDArray[np.float64],
        bandwidth_factor: float,
    ) -> FitResult:
        peaks_result = self._find_initial_peaks(read_lengths, bandwidth_factor=bandwidth_factor)
        self.tr_lengths = peaks_result.tr_lengths
        self.x_vals = peaks_result.x_vals
        self.density = peaks_result.density
        self.peaks_positions = peaks_result.peaks_positions
        self.peaks_values = peaks_result.peaks_values

        reads = self.tr_lengths.flatten()
        total_reads = len(reads)
        peaks = np.asarray(self.peaks_values, dtype=np.float64)
        if total_reads == 0 or peaks.size == 0:
            self._fail("no_peaks")
            self._log_clustering_result()
            return FitResult.FAIL
        # per-read expected peak width, used for every assignment threshold below
        widths = np.asarray(self.typical_peak_width(reads), dtype=np.float64)

        # Step 1: assign each read to its closest peak if it is within widths * factor of it
        closest_peak, assigned = self._assign_reads_to_peaks(reads, peaks, widths)
        counts = np.array([int(np.sum(assigned & (closest_peak == i))) for i in range(peaks.size)])

        # Step 2: choose the main peak(s)
        min_count_per_peak = max(self.min_small_cluster_size, (1 - self.dominance_threshold) * total_reads)
        valid_peak_indices = np.flatnonzero(counts >= min_count_per_peak)
        if len(valid_peak_indices) <= 1:
            top_peak = int(np.argmax(counts))
            main_peaks = [peaks[top_peak]]
        else:
            top_two = np.argsort(counts)[::-1][:2]
            main_peaks = sorted(float(peaks[i]) for i in top_two)

        # Step 3: (re)assign reads to the chosen main peak(s)
        main_arr = np.asarray(main_peaks, dtype=np.float64)
        main_peak_of_read, main_assigned = self._assign_reads_to_peaks(reads, main_arr, widths)
        if len(valid_peak_indices) == 2:
            # extra width only allowed if there were only 2 >15% peaks to start with
            main_peak_of_read, main_assigned = self._assign_additional_reads_to_two_peaks(
                reads, np.asarray(main_peaks, dtype=np.float64), widths, main_peak_of_read, main_assigned
            )

        if len(main_peaks) == 2 and not self._is_two_clearly_separated_alleles(main_peaks[0], main_peaks[1], reads):
            # two peaks are not clearly separated - fail
            self._fail("failed_not_clearly_separated")
            self._log_clustering_result()
            return FitResult.FAIL

        bottom_peak = main_peaks[0]

        # Step 4: gather unassigned long reads into a wide peak
        unassigned = ~main_assigned
        wide_threshold = max(bottom_peak + self.wide_peak_min_absolute_offset, bottom_peak * self.wide_peak_relative_threshold)
        wide_mask = unassigned & (reads > wide_threshold)
        wide_count = int(np.sum(wide_mask))
        has_wide_peak = wide_count >= self.min_small_cluster_size and wide_count >= self.expanded_reads_ratio_threshold_one_allele * total_reads
        if not has_wide_peak:
            wide_mask = np.zeros(total_reads, dtype=bool)

        # Step 5: combine into final cluster labels (label 0 is always the lower cluster)
        combined = self._combine_into_labels(main_peaks, main_peak_of_read, main_assigned, wide_mask, has_wide_peak)
        labels = combined.labels
        self.shape = combined.shape
        result = FitResult.SOFT_FAIL if combined.soft_failed else FitResult.SUCCESS

        # Step 6: require enough of the reads to be assigned
        if result is FitResult.SUCCESS and float(np.mean(labels != -1)) < self.min_assigned_fraction:
            self.shape = "failed_insufficient_assigned"
            if len(set(labels[labels != -1].tolist())) < 2:
                self._fail(self.shape)
                self._log_clustering_result()
                return FitResult.FAIL
            result = FitResult.SOFT_FAIL

        self.labels = labels
        self.clusters = self._create_clusters_from_labels(self.tr_lengths, self.labels, unstable_labels={1} if has_wide_peak else set())
        self.is_confident_clustering = result is FitResult.SUCCESS
        self._log_clustering_result()
        return result

    def _combine_into_labels(
        self: "KDEPeakClusterer",
        main_peaks: list[float],
        main_peak_of_read: NDArray[np.int32],
        main_assigned: NDArray[np.bool_],
        wide_mask: NDArray[np.bool_],
        has_wide_peak: bool,
    ) -> _CombineResult:
        """Build final cluster labels (label 0 is always the lower cluster).

        A soft-failed combination still has usable labels, but they are not to be preferred over haplotags.
        """
        labels = np.full(len(main_assigned), -1, dtype=np.int32)
        if len(main_peaks) == 2:
            bottom_peak, second_peak = main_peaks
            well_apart = second_peak > max(bottom_peak + self.well_apart_absolute_offset, bottom_peak * self.well_apart_relative)
            labels[main_assigned & (main_peak_of_read == 0)] = 0
            if has_wide_peak and not well_apart:
                # the upper of the two close peaks may belong to either allele, so leave those reads unassigned
                labels[wide_mask] = 1
                return _CombineResult(labels, "failed_close_peaks_plus_wide", soft_failed=True)
            labels[main_assigned & (main_peak_of_read == 1)] = 1
            if has_wide_peak:
                # two peaks well apart: merge the wide peak into the upper peak
                labels[wide_mask] = 1
                return _CombineResult(labels, "two_peaks_one_wide", soft_failed=False)
            return _CombineResult(labels, "two_peaks", soft_failed=False)
        labels[main_assigned] = 0
        if has_wide_peak:
            # a single peak plus a wide peak: the wide peak stays a separate cluster
            labels[wide_mask] = 1
            return _CombineResult(labels, "one_peak_one_wide", soft_failed=False)
        return _CombineResult(labels, "single_peak", soft_failed=False)

    def _log_clustering_result(self: "KDEPeakClusterer") -> None:
        """Describe the clusters for debugging purposes."""
        logging.debug(f"{len(self.peaks_values)} initial peaks (before any splitting): {self.peaks_values}")
        logging.debug(f"Cluster shape determined: {self.shape}")
        for i, cluster in enumerate(self.clusters):
            logging.debug(f"Cluster {i} center: {cluster.center}, range: {cluster.range}, unstable: {cluster.is_unstable}")
            logging.debug(f"Cluster {i} has {sum(self.labels == i)} reads assigned: {sorted([int(x) for x in self.tr_lengths[self.labels == i]])}")

    def min_fraction_per_peak(self: "KDEPeakClusterer", distance_between_peaks: float) -> float:
        """Minimum fraction of the reads that each of two peaks this far apart must hold to be believed.

        This is ``check_min_minority_fraction`` at ``min_peak_separation``, falling smoothly towards
        ``1 - dominance_threshold`` as the peaks get further apart, because the closer two peaks are the more
        likely it is that the smaller one is only sequencing error of the larger. Peaks closer than
        ``min_peak_separation`` are *not* extrapolated to an even higher fraction, because this method is
        also used for assessing haplotag validity, where peaks 0 apart are allowed.

        Args:
            distance_between_peaks (float): Distance between the two peaks, in bases.

        Returns:
            float: The minimum fraction of the reads required in each peak.
        """
        floor = 1 - self.dominance_threshold
        if distance_between_peaks <= self.min_peak_separation:
            return self.check_min_minority_fraction
        return floor + (self.check_min_minority_fraction - floor) * (self.min_peak_separation / distance_between_peaks)

    def _is_two_clearly_separated_alleles(self: "KDEPeakClusterer", peak1: float, peak2: float, reads: NDArray[np.float64]) -> bool:
        """
        Check if the peaks are clearly separated.

        The decision uses several parameters set in ``__init__``:

        - Peaks closer than ``min_peak_separation`` bases are never accepted (returns False).
        - Peaks at least ``min_unchecked_peak_separation`` bases apart are always accepted (returns True).
          (We don't just rely on min_fraction_per_peak falling to near ``1 - dominance_threshold``
          for very far-apart peaks, because far-apart peaks may be wide enough that their reads aren't
          close to the peak as here defined.)
        - In between, the peaks are checked against the surrounding reads. A read counts as "near" a peak if it is
          within ``distance_between_peaks / clearly_separated_min_distance_over_halfwidth`` of it, and "nearby" the
          pair if it is within ``2 * distance_between_peaks`` of their midpoint. The pair is accepted only if each
          peak holds at least a minimum fraction of the nearby reads (rejecting very unbalanced pairs) and the two
          peaks together hold at least ``check_min_total_fraction`` of them (rejecting pairs with too many reads
          sitting between or around the peaks rather than at them). The minimum fraction per peak depends on the
          distance between the peaks: see :meth:`min_fraction_per_peak`.

        Args:
            peak1 (float): Position of the first peak.
            peak2 (float): Position of the second peak.
            reads (NDArray[np.float64]): Array of read lengths.

        Returns:
            bool: True if the peaks are clearly separated, False otherwise.
        """
        reads = np.array(reads, dtype=np.float64)
        distance_between_peaks = abs(peak1 - peak2)
        if distance_between_peaks < self.min_peak_separation:
            return False
        if distance_between_peaks >= self.min_unchecked_peak_separation:
            return True
        min_fraction_per_peak = self.min_fraction_per_peak(distance_between_peaks)
        halfwidth_to_count = distance_between_peaks / self.clearly_separated_min_distance_over_halfwidth
        count_near_peak1 = int(np.sum(np.abs(reads - peak1) < halfwidth_to_count))
        count_near_peak2 = int(np.sum(np.abs(reads - peak2) < halfwidth_to_count))
        # > 1.5 so if these are 2 of a 3-peak set (e.g. errors of +/- one motif), 'nearby' includes the third one
        halfwidth_nearby = distance_between_peaks * 2.0
        count_nearby = int(np.sum(np.abs(reads - (peak1 + peak2) / 2) < halfwidth_nearby))
        min_count_per_peak = count_nearby * min_fraction_per_peak
        if count_near_peak1 < min_count_per_peak or count_near_peak2 < min_count_per_peak:
            return False
        return (count_near_peak1 + count_near_peak2) / count_nearby >= self.check_min_total_fraction

    def _create_clusters_from_labels(
        self: "KDEPeakClusterer",
        tr_lengths: NDArray[np.float64],
        labels: NDArray[np.int32],
        unstable_labels: set[int] | None = None,
    ) -> list[ClusterInfo]:
        """
        Create clusters based on unique labels in `labels`.
        Calculate cluster properties such as center, range, and instability.

        Args:
            tr_lengths (NDArray[np.float64]): Array of read lengths.
            labels (NDArray[np.int32]): Array of cluster labels.
            unstable_labels (set[int] | None): Labels that are considered unstable.

        Returns:
            List[ClusterInfo]: List of ClusterInfo objects representing the clusters.
        """
        # Get unique cluster labels excluding -1 (outliers)
        unique_labels = np.unique(labels[labels != -1])

        # Initialize clusters
        clusters = []

        for label in unique_labels:
            # Get lengths corresponding to the current label
            cluster_lengths = tr_lengths[labels == label]

            # Calculate cluster properties
            cluster_center = int(np.median(cluster_lengths))
            cluster_range = (int(cluster_lengths.min()), int(cluster_lengths.max()))

            # Create and append the cluster
            clusters.append(
                ClusterInfo(
                    center=cluster_center,
                    range=cluster_range,
                    is_unstable=unstable_labels is not None and label in unstable_labels,
                )
            )
        return clusters

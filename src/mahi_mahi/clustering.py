import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .cluster_visualization import KDEPeakClustererWithPlotting as KDEPeakClusterer
from .pipeline_records import AlleleAssignmentContext, ClusteringState, MergedLocusRead
from .utils import is_sex_chr, sanitize_plot_filename


def check_consistency(
    tr_ps_hap_labels: list[str],
    tr_labels: list[int],
    *,
    max_mismatch_fraction: float,
) -> bool:
    """
    Checks the consistency of clustering labels with haplotags by ensuring that for each
    phaseset (ps), the haplotypes and their associated cluster IDs
    are grouped into one or two distinct groups excluding untagged reads.

    Args:
        tr_ps_hap_labels (list of str): A list of initial labels in the format
            "ps_hap", where `ps` is the phaseset identifier and `hap`
            indicates the haplotype (e.g., "1", "2", or "0" for reads without haplotags).
        tr_labels (list of int): A list of cluster IDs corresponding to the
            initial labels. A cluster ID of -1 indicates an unclustered item.
        max_mismatch_fraction (float): Maximum allowed fraction of tagged reads per phaseset
            (excluding untagged reads, i.e., labels "0" and cluster ID -1)
            that can fall outside the two largest haplotype-cluster groups.

    Returns:
        bool: Returns `True` if the clustering is consistent (i.e., there are
        one or two dominant haplotype-cluster ID groups per phaseset, allowing up to
        `max_mismatch_fraction` mismatches among tagged reads). Returns `False` otherwise.
    """
    ps_to_pairs = defaultdict(list)

    for label, cluster_id in zip(tr_ps_hap_labels, tr_labels, strict=False):
        ps, hap = label.split("_")
        if hap != "0" and cluster_id != -1:
            ps_to_pairs[ps].append((hap, cluster_id))

    for pairs in ps_to_pairs.values():
        counter = Counter(pairs)
        total = sum(counter.values())
        top_two_counts_sum = sum(count for _, count in counter.most_common(2))
        mismatches = total - top_two_counts_sum
        # Check if the fraction of mismatched reads exceeds the allowed maximum mismatch ratio.
        if mismatches / total > max_mismatch_fraction:
            logging.debug(f"Haplotag consistency check failed: {mismatches} mismatches out of {total} reads, ratio {mismatches / total}.")
            return False
    return True


def motif_mode(reads: Sequence[MergedLocusRead]) -> tuple[set[tuple[str, ...]], int]:
    """
    Finds the most common motif decomposition(s) of a group of reads.

    Decompositions are compared exactly, i.e. two reads count as having the same decomposition
    only if their motif sequences are identical.

    Args:
        reads (Sequence[MergedLocusRead]): The reads assigned to one allele.

    Returns:
        tuple[set[tuple[str, ...]], int]: The set of motif decompositions tied for most common
        (a single-element set if there is one clear mode) and the number of reads having each of
        them, or (set(), 0) if there are no reads.
    """
    counts = Counter(record.motif_dec for record in reads)
    if not counts:
        return set(), 0
    max_count = max(counts.values())
    return {motif_dec for motif_dec, count in counts.items() if count == max_count}, max_count


def check_distinct_motif_modes(
    reads1: Sequence[MergedLocusRead],
    reads2: Sequence[MergedLocusRead],
    *,
    name_of_group: str,
) -> bool:
    """
    Checks whether two candidate alleles look like genuinely different sequences.

    Candidate alleles at least ``KDEPeakClusterer.min_unchecked_peak_separation`` bases apart are
    accepted without further checks. Otherwise, reads of the same allele usually share one exact
    motif decomposition, so if one candidate allele has a single most common decomposition shared
    by at least ``KDEPeakClusterer.min_small_cluster_size`` reads, the split is rejected if that
    decomposition is also (tied for) most common in the other allele, or if the other allele has no
    repeated decomposition at all (i.e. it is too noisy to tell the two apart). Otherwise the split
    is accepted, as having no clear mode on either side is normal for long (i.e. nearly always
    containing a sequencing error) alleles.

    Args:
        reads1 (Sequence[MergedLocusRead]): The reads assigned to the first allele.
        reads2 (Sequence[MergedLocusRead]): The reads assigned to the second allele.
        name_of_group (str): Name of the region group, used in log messages.

    Returns:
        bool: True if the two alleles may be kept separate, False if the split should be rejected.
    """
    if not reads1 or not reads2:
        return True
    median1 = float(np.median([record.size for record in reads1]))
    median2 = float(np.median([record.size for record in reads2]))
    clusterer = KDEPeakClusterer()
    if abs(median1 - median2) >= clusterer.min_unchecked_peak_separation:
        return True
    modes1, mode1_count = motif_mode(reads1)
    modes2, mode2_count = motif_mode(reads2)
    for modes, mode_count, other_modes, other_mode_count in ((modes1, mode1_count, modes2, mode2_count), (modes2, mode2_count, modes1, mode1_count)):
        if len(modes) != 1 or mode_count < clusterer.min_small_cluster_size:
            continue
        (mode,) = modes
        if mode in other_modes:
            if median1 != median2:
                logging.warning(
                    f"{name_of_group}: the two candidate alleles have the same most common repeat sequence but different median lengths"
                    f" ({median1:g} and {median2:g}): treating them as one allele."
                )
            else:
                logging.debug(f"{name_of_group}: the two candidate alleles have the same most common repeat sequence: treating them as one allele.")
            return False
        if other_mode_count == 1:
            logging.warning(
                f"{name_of_group}: one candidate allele has a most common repeat sequence but no two reads of the other share a repeat"
                f" sequence, and the two are only {abs(median1 - median2):g} bases apart (less than"
                f" {clusterer.min_unchecked_peak_separation}), so we cannot tell whether they differ: treating them as one allele."
            )
            return False
    logging.debug(f"{name_of_group}: the two candidate alleles have different repeat sequences, so the split is accepted")
    return True


def handle_haploid_regions(
    reads_in_region: list[MergedLocusRead],
    region_name: str,
    non_spanning_read_policy: Literal["default", "exclude"] = "default",
) -> None:
    """
    Handles haploid regions by assigning default haplotype and clustering values.

    Args:
        reads_in_region (list[MergedLocusRead]): Read information for the region.
        region_name: Locus name used in log messages.
        non_spanning_read_policy: Whether to include non-spanning reads. "default" uses the
            standard logic; "exclude" always assigns non-spanning reads to allele 0.
    """
    if not reads_in_region:
        return
    logging.info(f"Region {region_name} is treated as haploid.")
    for record in reads_in_region:
        record.clustered = False
        record.allele_id = "1"
    if non_spanning_read_policy == "exclude":
        for record in reads_in_region:
            if not record.is_spanning:
                record.allele_id = "0"
                record.included_in_allele_size_estimation = False


def get_non_spanning_min_included_length(reads_in_region: Sequence[MergedLocusRead], non_spanning_min_extra_length: int) -> int:
    """
    Determines the minimum length for non-spanning reads to be included in allele size estimation.

    This is calculated as the longest allele that can be found using only
    spanning reads (which may not be the actual longest allele), plus
    an extra length to exclude barely-spanning reads that have been misclassified as non-spanning.

    This is intended to include non-spanning reads only when they are needed,
    to find very long alleles that would otherwise be missed (or have their
    size greatly under-estimated) because they are rarely or never spanned,
    while excluding other non-spanning reads to not bias sizes downwards.

    Args:
        reads_in_region (Sequence[MergedLocusRead]): Read information for the region.
        non_spanning_min_extra_length (int): Extra length to add to the maximum spanning allele length.

    Returns:
        int: The minimum length for non-spanning reads to be included in allele size estimation.
    """
    spanning_tr_lengths = np.array([record.size for record in reads_in_region if record.is_spanning], dtype=float)
    spanning_only_clusterer = KDEPeakClusterer()
    if len(spanning_tr_lengths) >= 2:
        spanning_only_clusterer.fit(spanning_tr_lengths)
        # 'default' applies if clustering failed or is not to be trusted, and treats all spanning reads as one allele
        clusters = spanning_only_clusterer.clusters if spanning_only_clusterer.is_confident_clustering else []
        longest_spanned_allele_length = max((cluster.center for cluster in clusters), default=int(np.median(spanning_tr_lengths)))
        non_spanning_min_included_length = longest_spanned_allele_length + non_spanning_min_extra_length
    else:  # one or no spanning reads - include all the non-spanning reads
        non_spanning_min_included_length = 0
    return non_spanning_min_included_length


def exclude_non_spanning_reads(reads_in_region: Iterable[MergedLocusRead]) -> None:
    """
    Assigns every non-spanning read to allele 0 and excludes it from allele size estimation.

    Args:
        reads_in_region (Iterable[MergedLocusRead]): Reads to update in place.
    """
    for record in reads_in_region:
        if not record.is_spanning:
            record.allele_id = "0"
            record.included_in_allele_size_estimation = False


def apply_non_spanning_read_policy(
    reads_in_region: list[MergedLocusRead],
    *,
    non_spanning_read_policy: Literal["default", "exclude"],
    non_spanning_min_extra_length: int,
) -> list[MergedLocusRead]:
    """
    Applies the configured non-spanning read policy to already-clustered reads.

    Args:
        reads_in_region (list[MergedLocusRead]): All reads in the region after clustering.
        non_spanning_read_policy: "exclude" drops every non-spanning read (see ``exclude_non_spanning_reads``),
            "default" keeps them and assigns the between-allele ones (see ``assign_non_spanning_reads``).
        non_spanning_min_extra_length (int): Extra length threshold used by ``assign_non_spanning_reads``.

    Returns:
        list[MergedLocusRead]: The updated reads.
    """
    if non_spanning_read_policy == "exclude":
        exclude_non_spanning_reads(reads_in_region)
        return reads_in_region
    return assign_non_spanning_reads(reads_in_region, non_spanning_min_extra_length)


def drop_haplotag_inconsistent_reads(reads_in_region: Iterable[MergedLocusRead]) -> None:
    """
    Assigns every haplotagged read whose allele disagrees with its haplotag to allele 0.

    Such reads are also excluded from allele size estimation. Untagged reads and reads that are
    already unassigned are left alone.

    Args:
        reads_in_region (Iterable[MergedLocusRead]): Reads to update in place.
    """
    for record in reads_in_region:
        if record.allele_id != record.original_hap and record.original_hap != "0" and record.allele_id != "0":
            record.allele_id = "0"
            record.included_in_allele_size_estimation = False


@dataclass(frozen=True)
class ClusteringOutcome:
    """
    Result of the length-based clustering step (see ``run_kde_clustering``).

    Attributes:
        labels (NDArray[np.int32]): Cluster label per selected read; -1 means unassigned.
        failed (bool): True if no usable clusters were found.
        homogeneous (bool): True if at most one cluster was found (there may still be unassigned outliers).
        confident (bool): Whether the clusters are trustworthy enough to take precedence over haplotags.
            Clusters found despite a soft failure are not, and are only used where there are no usable haplotags.
    """

    labels: NDArray[np.int32]
    failed: bool
    homogeneous: bool
    confident: bool


def mark_ambiguous_if_homozygous_with_many_non_spanning_reads(
    reads_in_region: Sequence[MergedLocusRead],
    clustering_state: ClusteringState,
    *,
    is_haploid_locus: bool = False,
) -> ClusteringState:
    """Mark a non-haploid homozygous call ambiguous when most reads are non-spanning.

    This is an additional check after allele assignment.  It deliberately counts every
    read in ``reads_in_region``, including non-spanning reads that were excluded from
    clustering or allele-size estimation.
    """
    if not reads_in_region or is_haploid_locus or clustering_state is ClusteringState.ASSUMED_HAPLOID:
        return clustering_state

    non_spanning_fraction = sum(not record.is_spanning for record in reads_in_region) / len(reads_in_region)
    dominance_threshold = KDEPeakClusterer().dominance_threshold
    if non_spanning_fraction <= 1 - dominance_threshold:
        return clustering_state

    allele_ids = {record.allele_id for record in reads_in_region if record.allele_id not in (None, "", "0")}
    if len(allele_ids) <= 1:
        return ClusteringState.AMBIGUOUS
    return clustering_state


def run_kde_clustering(
    selected_reads: Sequence[MergedLocusRead],
    tr_lengths: NDArray[np.float64],
    tr_ps_hap_labels: list[str],
    *,
    min_read_support_per_allele: int,
    context: AlleleAssignmentContext,
    plots_basename: str | Path | None,
    plot_first_attempt_density: bool = False,
) -> ClusteringOutcome:
    """
    Clusters the selected reads by TR length, and optionally writes a debug plot of the result.

    Args:
        selected_reads (Sequence[MergedLocusRead]): The reads considered for clustering.
        tr_lengths (NDArray[np.float64]): TR length of each selected read.
        tr_ps_hap_labels (list[str]): Initial "ps_hap" label of each selected read, used for logging and plotting.
        min_read_support_per_allele (int): Clustering is not attempted with fewer reads than this.
        context: Locus identity used for logging and debug plots.
        plots_basename: If not None, a KDE plot is written using this as the filename prefix.
        plot_first_attempt_density: Include the first density estimate when a fallback fit was run.

    Returns:
        ClusteringOutcome: The cluster labels and what they tell us about the region.
    """
    region_name = context.region_name
    # Debug information for clustering trace logs.
    sorted_indices = np.argsort(tr_lengths)
    print_tr_lengths = ",".join([str(int(x)) for x in tr_lengths[sorted_indices]])
    print_tr_ps_hap_labels = ",".join([tr_ps_hap_labels[i] for i in sorted_indices])
    logging.debug(f"{region_name}: clustering {print_tr_lengths} with initial labels {print_tr_ps_hap_labels}")
    clusterer = KDEPeakClusterer()
    if len(selected_reads) >= min_read_support_per_allele:
        # KDE-based clustering and peak detection.
        tr_labels = clusterer.fit(tr_lengths)
        if plots_basename is not None:
            region_name_str = context.display_name
            plots_path = Path(plots_basename)
            repeat_id = sanitize_plot_filename(context.clustering_id)
            sample_name = plots_path.name
            plot_filename = plots_path.parent / f"{repeat_id}_clustering_{sample_name}.pdf"
            plot_filename.parent.mkdir(parents=True, exist_ok=True)
            clusterer.plot_KDE(
                tr_ps_hap_labels=tr_ps_hap_labels,
                is_filled=np.array([record.is_spanning for record in selected_reads], dtype=bool),
                is_filled_labels=("Non-spanning read", "Spanning read"),
                region_name=region_name_str,
                filename=Path(plot_filename),
                plot_first_attempt_density=plot_first_attempt_density,
            )
        clustering_failed = not np.any(tr_labels != -1)
        homogeneous_cluster = set(tr_labels) <= {0, -1}  # single peak but may have unclassified outliers
        if not clustering_failed and not homogeneous_cluster:
            grouped_reads: defaultdict[int, list[MergedLocusRead]] = defaultdict(list)
            for record, label in zip(selected_reads, tr_labels.tolist(), strict=True):
                if label != -1:
                    grouped_reads[label].append(record)
            groups = [grouped_reads[label] for label in sorted(grouped_reads)]
            if len(groups) == 2 and not check_distinct_motif_modes(groups[0], groups[1], name_of_group=str(region_name)):
                # Treat an allele split that we cannot tell apart by sequence as a failed clustering.
                clustering_failed = True
                homogeneous_cluster = True
    else:
        logging.debug(f"{region_name}: not enough spanning reads for clustering")
        tr_labels = np.array([-1] * len(selected_reads), dtype=np.int32)
        clustering_failed = True
        homogeneous_cluster = True
    return ClusteringOutcome(
        labels=tr_labels,
        failed=bool(clustering_failed),
        homogeneous=homogeneous_cluster,
        confident=clusterer.is_confident_clustering and not clustering_failed,
    )


def cluster_reads(
    tr_records: Sequence[MergedLocusRead],
    sex: str,
    min_read_support_per_allele: int,
    *,
    non_spanning_read_policy: Literal["default", "exclude"],
    exclude_haplotag_inconsistent_reads: bool,
    hap_max_mismatch_fraction: float,
    non_spanning_min_extra_length: int = 75,
    selected_ps: str,
    context: AlleleAssignmentContext,
    plots_basename: str | Path | None = None,
    plot_first_attempt_density: bool = False,
) -> tuple[list[MergedLocusRead], bool, ClusteringState]:
    """
    Cluster reads based on input parameters.

    A two-allele split, whether found by clustering or taken from haplotags, is rejected if the two
    alleles cannot be told apart by their repeat sequence (see ``check_distinct_motif_modes``); a
    rejected clustering is treated like a failed one.

    Args:
        tr_records (Sequence[MergedLocusRead]): Input records containing read information.
        sex (str): Sex information, either "XY" or "XX".
        min_read_support_per_allele (int): Minimum number of reads required to support clustering in a region.
        selected_ps: Dominant phase set selected for this locus.
        context: Locus identity used for ploidy decisions, logging, and debug plots.
        plot_first_attempt_density: Include the first density estimate in debug plots when a fallback fit was run.
        non_spanning_min_extra_length (int): Minimum extra length (past the longest spanning length) for non-spanning reads,
            to skip barely-spanning reads that have been mistakenly marked as non-spanning and hence have the wrong length.
        non_spanning_read_policy: Whether to include non-spanning reads in clustering and allele size estimation.
            "default" uses the standard logic (include long non-spanning reads to find very long alleles);
            "exclude" excludes all non-spanning reads and assigns them to allele 0.

    Returns:
        tuple[list[MergedLocusRead], bool, ClusteringState]: Clustered records, region phased state,
        and how the alleles were assigned.
    """
    if not tr_records:
        logging.debug("No regions to process, returning empty record list.")
        return [], False, ClusteringState.AMBIGUOUS

    phase_sets = {record.ps for record in tr_records}
    expected_phase_sets = {selected_ps, "ps0"}
    assert phase_sets <= expected_phase_sets, f"Expected reads from selected phase set {selected_ps} or ps0, found phase sets {sorted(phase_sets)}"

    reads_in_region = [record.clone() for record in tr_records]
    region_name = context.region_name
    non_spanning_min_included_length = get_non_spanning_min_included_length(reads_in_region, non_spanning_min_extra_length)
    logging.debug(f"{region_name}: any non-spanning reads shorter than {non_spanning_min_included_length} will be excluded from clustering")
    selected_reads: list[MergedLocusRead] = []
    non_spanning_short_reads: list[MergedLocusRead] = []
    for record in reads_in_region:
        if record.is_spanning or (record.size >= non_spanning_min_included_length and non_spanning_read_policy != "exclude"):
            record.included_in_allele_size_estimation = True
            selected_reads.append(record)
        else:
            record.included_in_allele_size_estimation = False
            non_spanning_short_reads.append(record)
        record.allele_size_estimation_quantile = "allele_q50"
    tr_ps_hap_labels = [f"{record.ps}_{record.original_hap}" for record in selected_reads]
    tr_lengths = np.array([record.size for record in selected_reads], dtype=float)
    tagged = is_haplotagged(selected_reads, min_depth_per_allele=min_read_support_per_allele, selected_ps=selected_ps, name_of_group=str(region_name))
    is_haploid_locus = sex == "XY" and is_sex_chr(context.chrom)
    if len(reads_in_region) < 2 or is_haploid_locus:
        logging.info(f"{region_name}: not considered for clustering")
        handle_haploid_regions(reads_in_region, region_name, non_spanning_read_policy)
        return reads_in_region, tagged, ClusteringState.ASSUMED_HAPLOID if is_haploid_locus else ClusteringState.AMBIGUOUS

    outcome = run_kde_clustering(
        selected_reads,
        tr_lengths,
        tr_ps_hap_labels,
        min_read_support_per_allele=min_read_support_per_allele,
        context=context,
        plots_basename=plots_basename,
        plot_first_attempt_density=plot_first_attempt_density,
    )
    if tagged:
        reads, phased, clustering_state = cluster_tagged_reads(
            reads_in_region,
            selected_reads,
            non_spanning_short_reads,
            outcome,
            tr_ps_hap_labels,
            region_name=region_name,
            selected_ps=selected_ps,
            non_spanning_read_policy=non_spanning_read_policy,
            exclude_inconsistent_reads=exclude_haplotag_inconsistent_reads,
            hap_max_mismatch_fraction=hap_max_mismatch_fraction,
            non_spanning_min_extra_length=non_spanning_min_extra_length,
        )
    else:
        reads, clustering_state = cluster_untagged_reads(
            reads_in_region,
            selected_reads,
            non_spanning_short_reads,
            outcome,
            region_name=region_name,
            non_spanning_read_policy=non_spanning_read_policy,
            non_spanning_min_extra_length=non_spanning_min_extra_length,
        )
        phased = False
    return reads, phased, mark_ambiguous_if_homozygous_with_many_non_spanning_reads(reads, clustering_state)


def cluster_tagged_reads(
    reads_in_region: list[MergedLocusRead],
    selected_reads: Sequence[MergedLocusRead],
    non_spanning_short_reads: Sequence[MergedLocusRead],
    outcome: ClusteringOutcome,
    tr_ps_hap_labels: list[str],
    *,
    region_name: str,
    selected_ps: str,
    non_spanning_read_policy: Literal["default", "exclude"],
    exclude_inconsistent_reads: bool,
    hap_max_mismatch_fraction: float,
    non_spanning_min_extra_length: int,
) -> tuple[list[MergedLocusRead], bool, ClusteringState]:
    """
    Assigns alleles in a region that has usable haplotags.

    If clustering found at most one allele, or found alleles it is not confident in, the haplotags are used
    as-is; otherwise the clusters and the haplotags are reconciled by ``handle_tagged_region``.

    Args:
        reads_in_region (list[MergedLocusRead]): All reads in the region, updated in place.
        selected_reads (Sequence[MergedLocusRead]): The reads that were considered for clustering.
        non_spanning_short_reads (Sequence[MergedLocusRead]): The reads that were not.
        outcome: The result of ``run_kde_clustering``.
        tr_ps_hap_labels (list[str]): Initial "ps_hap" label of each selected read.
        region_name (str): Name of the region, used in log messages.
        selected_ps (str): Dominant phase set chosen for this region.
        non_spanning_read_policy: Whether to keep non-spanning reads (see ``apply_non_spanning_read_policy``).
        exclude_inconsistent_reads (bool): Whether to drop reads whose allele disagrees with their haplotag.
        hap_max_mismatch_fraction (float): Passed to the haplotag consistency check.
        non_spanning_min_extra_length (int): Extra length threshold for assigning non-spanning reads.

    Returns:
        tuple[list[MergedLocusRead], bool, ClusteringState]: The updated records, region phased state,
        and how the alleles were assigned.
    """
    if outcome.failed or outcome.homogeneous or not outcome.confident:
        description = "clustering failed" if outcome.failed else "homogeneous cluster" if outcome.homogeneous else "clustering is not confident"
        logging.debug(f"{region_name}: {description} and haplotags are used")
        for record in reads_in_region:
            record.allele_id = record.original_hap
        if non_spanning_read_policy == "exclude":
            exclude_non_spanning_reads(reads_in_region)
        return reads_in_region, True, ClusteringState.HAPLOTAG_FALLBACK
    # only clusters that take precedence over the haplotags get here, which assign_non_spanning_reads relies on
    result, phased, clustering_state = handle_tagged_region(
        region_name,
        selected_reads,
        non_spanning_short_reads,
        outcome.labels,
        tr_ps_hap_labels,
        selected_ps,
        do_check_consistency=not outcome.confident,
        hap_max_mismatch_fraction=hap_max_mismatch_fraction,
    )
    result = apply_non_spanning_read_policy(
        result,
        non_spanning_read_policy=non_spanning_read_policy,
        non_spanning_min_extra_length=non_spanning_min_extra_length,
    )
    if phased and exclude_inconsistent_reads:
        drop_haplotag_inconsistent_reads(result)
    return result, phased, clustering_state


def cluster_untagged_reads(
    reads_in_region: list[MergedLocusRead],
    selected_reads: Sequence[MergedLocusRead],
    non_spanning_short_reads: Sequence[MergedLocusRead],
    outcome: ClusteringOutcome,
    *,
    region_name: str,
    non_spanning_read_policy: Literal["default", "exclude"],
    non_spanning_min_extra_length: int,
) -> tuple[list[MergedLocusRead], ClusteringState]:
    """
    Assigns alleles in a region that has no usable haplotags, using the clusters alone.

    A region whose reads could not be clustered is treated as haploid.  Clusters the clusterer is not
    confident in are used here, as there are no haplotags to prefer to them, but leave the region AMBIGUOUS.

    Args:
        reads_in_region (list[MergedLocusRead]): All reads in the region, updated in place.
        selected_reads (Sequence[MergedLocusRead]): The reads that were considered for clustering.
        non_spanning_short_reads (Sequence[MergedLocusRead]): The reads that were not.
        outcome: The result of ``run_kde_clustering``.
        region_name (str): Name of the region, used in log messages.
        non_spanning_read_policy: Whether to keep non-spanning reads (see ``apply_non_spanning_read_policy``).
        non_spanning_min_extra_length (int): Extra length threshold for assigning non-spanning reads.

    Returns:
        tuple[list[MergedLocusRead], ClusteringState]: The updated records and how the alleles were assigned.
    """
    if outcome.failed:
        logging.debug(f"{region_name}: failing to cluster, no haplotags")
        handle_haploid_regions(reads_in_region, region_name, non_spanning_read_policy)
        return reads_in_region, ClusteringState.AMBIGUOUS
    # the E2E tests and the clustering reports check for these messages, so if you reword them, update those checks
    if outcome.confident:
        logging.debug(f"{region_name}: performed clustering and no haplotags")
    else:
        logging.debug(f"{region_name}: performed unconfident clustering and no haplotags")
    result = handle_non_tagged_region(region_name, selected_reads, non_spanning_short_reads, outcome.labels)
    return (
        apply_non_spanning_read_policy(
            result,
            non_spanning_read_policy=non_spanning_read_policy,
            non_spanning_min_extra_length=non_spanning_min_extra_length,
        ),
        ClusteringState.SUCCESS if outcome.confident else ClusteringState.AMBIGUOUS,
    )


def handle_tagged_region(
    name_of_group: str,
    selected_reads: Sequence[MergedLocusRead],
    non_spanning_short_reads: Sequence[MergedLocusRead],
    tr_labels: NDArray[np.int32],
    tr_ps_hap_labels: list[str],
    selected_ps: str,
    *,
    hap_max_mismatch_fraction: float,
    do_check_consistency: bool = True,
) -> tuple[list[MergedLocusRead], bool, ClusteringState]:
    """
    Handles clustering for tagged regions.

    This function ensures the consistency of clustering labels with whatshap tagging information.
    If the consistency check succeeds, it carries over the tagging information to the clustering labels.

    Args:
        name_of_group (str): Name of the region group.
        selected_reads (Sequence[MergedLocusRead]): Selected read information for the region.
        non_spanning_short_reads (Sequence[MergedLocusRead]): Non-spanning read information for the region
            that are shorter than the maximum spanning read.
        tr_labels (NDArray[np.int32]): Cluster labels.
        tr_ps_hap_labels (List[str]): Initial labels.
        selected_ps (str): Dominant phase set chosen for this region.
        do_check_consistency (bool): Whether to perform consistency check with haplotags.

    Returns:
        tuple[list[MergedLocusRead], bool, ClusteringState]: The updated records, region phased state,
        and how the alleles were assigned.
    """
    selected_records = [record.clone() for record in selected_reads]
    non_spanning_records = [record.clone() for record in non_spanning_short_reads]
    proper_clustering = check_consistency(tr_ps_hap_labels, tr_labels.tolist(), max_mismatch_fraction=hap_max_mismatch_fraction)
    for record in non_spanning_records:
        record.allele_id = record.original_hap
    if proper_clustering or not do_check_consistency:
        new_labels = map_labels_to_initial(tr_ps_hap_labels, tr_labels.tolist(), selected_ps)
        num_reads, num_clustered_reads = (
            len(selected_records),
            len(tr_labels),
        )

        if num_reads != num_clustered_reads:
            error_message = f"Number of clustered sequences is not equal to number of initial ones: {num_reads} != {num_clustered_reads}"
            logging.error(error_message)
            raise ValueError(error_message)

        for record, new_label in zip(selected_records, new_labels, strict=True):
            record.allele_id = str(new_label)
            record.ps = selected_ps
            record.clustered = True
        logging.debug(f"{name_of_group}: performed clustering")
        if not proper_clustering:
            for record in non_spanning_records:
                record.allele_id = "0"
            logging.warning(
                # the E2E tests check for this message, so if you reword it, update that check
                f"{name_of_group}: TR length clusters are inconsistent with haplotags, which implies that at least one of them is wrong:"
                " consider inspecting the .tagged.bam output with e.g. IGV.  Using clusters."
            )
            return selected_records + non_spanning_records, False, ClusteringState.SUCCESS
    else:
        logging.warning(
            # the E2E tests check for this message, so if you reword it, update that check
            f"{name_of_group}: TR length clusters are inconsistent with haplotags, which implies that at least one of them is wrong:"
            " consider inspecting the .tagged.bam output with e.g. IGV.  Using haplotags."
        )
        for record in selected_records:
            record.allele_id = record.original_hap
        return selected_records + non_spanning_records, True, ClusteringState.HAPLOTAG_FALLBACK
    return selected_records + non_spanning_records, True, ClusteringState.SUCCESS


def handle_non_tagged_region(
    name_of_group: str,
    selected_reads: Sequence[MergedLocusRead],
    non_spanning_short_reads: Sequence[MergedLocusRead],
    tr_labels: NDArray[np.int32],
) -> list[MergedLocusRead]:
    """
    Handles clustering for non-tagged regions.

    Args:
        name_of_group (str): Name of the region group.
        selected_reads (Sequence[MergedLocusRead]): Selected read information for the region.
        non_spanning_short_reads (Sequence[MergedLocusRead]): Non-spanning read information for the region.
        tr_labels (NDArray[np.int32]): Cluster labels.
    Returns:
        list[MergedLocusRead]: The updated records with clustering information.
    """
    selected_records = [record.clone() for record in selected_reads]
    non_spanning_records = [record.clone() for record in non_spanning_short_reads]
    logging.debug(f"{name_of_group}: assigning alleles from the clusters")
    for record, label in zip(selected_records, tr_labels, strict=True):
        record.allele_id = str(label + 1)
        record.clustered = True
    for record in non_spanning_records:
        record.allele_id = "0"
    reads_in_region = selected_records + non_spanning_records

    return reads_in_region


def assign_non_spanning_reads(
    reads_in_region: list[MergedLocusRead],
    non_spanning_min_extra_length: int,
) -> list[MergedLocusRead]:
    """
    Assigns unused non-spanning reads that fall between two alleles to the longer allele.

    Non-spanning reads excluded from allele size estimation whose size is between
    (75th percentile of shorter allele + non_spanning_min_extra_length) and
    (75th percentile of longer allele) are assigned to the longer allele
    (overriding any existing allele, such as a haplotag),
    but remain excluded from allele size estimation.

    Args:
        reads_in_region (list[MergedLocusRead]): All reads in the region after clustering.
        non_spanning_min_extra_length (int): Extra length threshold to avoid assigning
            barely-spanning misclassified reads.

    Returns:
        The updated reads with between-allele non-spanning reads assigned.
    """
    # Collect sizes per allele from reads included in size estimation
    allele_sizes: dict[str, list[float]] = defaultdict(list)
    for record in reads_in_region:
        if record.allele_id and record.allele_id != "0" and record.included_in_allele_size_estimation:
            allele_sizes[record.allele_id].append(record.size)

    if len(allele_sizes) < 2:
        return reads_in_region

    # Compute 75th percentile for each allele
    allele_q75: dict[str, float] = {}
    for allele_id, sizes in allele_sizes.items():
        allele_q75[allele_id] = float(np.percentile(sizes, 75))

    # Sort alleles by q75 to identify shorter and longer
    sorted_alleles = sorted(allele_q75.items(), key=lambda x: x[1])
    shorter_allele_q75 = sorted_alleles[0][1]
    longer_allele_id = sorted_alleles[-1][0]
    longer_allele_q75 = sorted_alleles[-1][1]

    lower_bound = shorter_allele_q75 + non_spanning_min_extra_length
    upper_bound = longer_allele_q75

    if lower_bound >= upper_bound:
        return reads_in_region

    for record in reads_in_region:
        if (not record.included_in_allele_size_estimation) and (not record.is_spanning) and lower_bound < record.size <= upper_bound:
            record.allele_id = longer_allele_id

    return reads_in_region


def is_haplotagged(reads_df: Sequence[MergedLocusRead], min_depth_per_allele: int, selected_ps: str, name_of_group: str = "") -> bool:
    """
    Determines if a region's haplotags may be used.

    Haplotags may be rejected if:
      - there are not enough of them
      - they are very unbalanced (one haplotype has many more reads than the other)
      - the TR allele appears to be the same in both haplotypes (see ``check_distinct_motif_modes``)
    The last causes homozygous alleles to be reported as single alleles, not split by haplotag.

    Args:
        reads_df (Sequence[MergedLocusRead]): Read information for a region.
        min_depth_per_allele (int): Minimum number of reads required per allele to consider the region haplotagged.
        selected_ps (str): The phaseset to evaluate.
        name_of_group (str): Name of the region group, used in log messages.

    Returns:
        bool: True if the region's haplotags may be used, False otherwise.
    """
    reads1 = [record for record in reads_df if record.ps == selected_ps and record.original_hap == "1"]
    reads2 = [record for record in reads_df if record.ps == selected_ps and record.original_hap == "2"]
    clusterer_for_params = KDEPeakClusterer()
    min_count_per_allele = max(min_depth_per_allele, (len(reads1) + len(reads2)) * (1 - clusterer_for_params.dominance_threshold))
    if not (len(reads1) >= min_count_per_allele and len(reads2) >= min_count_per_allele):
        return False
    return check_distinct_motif_modes(reads1, reads2, name_of_group=name_of_group)


def get_pairs_by_phaseset(tr_ps_hap_labels: list[str], tr_labels: list[int], selected_ps: str) -> list[tuple[int, int]]:
    """
    Retrieves (cluster_id, hap) pairs for a specific phaseset.

    Args:
        tr_ps_hap_labels (list of str): Initial labels in the format "ps_hap".
        tr_labels (list of int): Cluster labels.
        selected_ps (str): The phaseset to retrieve pairs for.

    Returns:
        list of tuple: A list of (cluster_id, hap) pairs for the selected phaseset.
    """
    pairs = []
    for label, cluster_id in zip(tr_ps_hap_labels, tr_labels, strict=False):
        ps, hap = label.split("_")
        if ps == selected_ps:
            hap = int(hap)
            if cluster_id != -1:  # Exclude unclustered items.
                pairs.append((cluster_id, hap))
    return pairs


def generate_label_mapping(pairs: list[tuple[int, int]]) -> dict[int, int]:
    """
    Creates a mapping of cluster IDs to the most common haplotype.

    Args:
        pairs (list of tuple): A list of (cluster_id, hap) pairs.

    Returns:
        dict: A dictionary mapping cluster IDs to the most common haplotype.
    """
    # 1. Collect haplotypes per cluster.
    cluster_to_haps: defaultdict[int, list[int]] = defaultdict(list)
    for cluster_id, hap in pairs:
        cluster_to_haps[cluster_id].append(hap)

    mappings = {cid: Counter(haps).most_common(1)[0][0] for cid, haps in cluster_to_haps.items() if haps}
    first_cluster_id = mappings.get(0, 1)
    second_cluster_id = mappings.get(1, 2)
    if first_cluster_id == second_cluster_id:
        first_cluster_id = 2
        second_cluster_id = 1
    return {0: first_cluster_id, 1: second_cluster_id}


def map_labels_to_initial(tr_ps_hap_labels: list[str], tr_labels: list[int], selected_ps: str) -> list[int]:
    """
    Maps cluster labels in `tr_labels` to match the equivalent initial labels in `tr_ps_hap_labels`.

    Uses the provided phaseset (ps) for mapping.

    Args:
        tr_ps_hap_labels (list of str): Initial labels in the format "ps_hap".
            The haplotype part must be one of "0", "1", or "2".
        tr_labels (list of int): Cluster labels, where only values 0, 1, or -1 are allowed.
        selected_ps (str): The phaseset to use for mapping.

    Returns:
        list of int: Updated cluster labels matching the initial labels.
    """
    assert all(label in (0, 1, -1) for label in tr_labels), "tr_labels must only contain 0, 1, or -1"
    haps = [hap.split("_")[1] for hap in tr_ps_hap_labels]
    assert all(hap in ("0", "1", "2") for hap in haps), "haplotype part of tr_ps_hap_labels must only contain '0', '1', or '2'"
    pairs = get_pairs_by_phaseset(tr_ps_hap_labels, tr_labels, selected_ps)
    label_mapping = generate_label_mapping(pairs)
    initial_haps = [int(hap[-1]) for hap in tr_ps_hap_labels]
    return [label_mapping.get(cluster_id, hap) for cluster_id, hap in zip(tr_labels, initial_haps, strict=True)]

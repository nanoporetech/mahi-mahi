import logging
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from .basecaller_error_detection import check_strand_bias_state
from .config import MahiMahiConfig
from .pipeline_records import AlleleSummary, ClusteringState, ReadCall, RegionCallSet, format_genotype
from .utils import expand_motif_representation, is_sex_chr, motif_rotation_representatives, sum_dictionaries_in_column


def _append_diagnostic_reads(region: RegionCallSet, diagnostic_reads: Sequence[ReadCall]) -> RegionCallSet:
    """Add diagnostic-only reads after all analytical summaries have been calculated."""
    if not diagnostic_reads:
        return region
    if not region.reads:
        return RegionCallSet.from_reads(
            diagnostic_reads,
            phased=region.phased,
            genotype=region.genotype,
            region_is_suspicious=region.region_is_suspicious,
            clustering_state=region.clustering_state,
        )
    return region.model_copy(update={"reads": [*region.reads, *diagnostic_reads]})


def _compute_motif_medians(group: Sequence[ReadCall]) -> dict[str, int]:
    """Return the median per-read count of each motif in this allele.

    Cyclic rotations are collapsed (see motif_rotation_representatives). Besides the most common motif,
    unexpected motifs are reported only when their counts agree across reads on both strands.
    """
    if not group:
        return {}
    expected_motifs = expand_motif_representation(group[0].repeat_unit)
    representatives = motif_rotation_representatives(sum_dictionaries_in_column(group, "motif_counts"), expected_motifs)
    motif_lists: list[Counter[str]] = []
    for record in group:
        normalized: Counter[str] = Counter()
        for motif, count in record.motif_counts.items():
            normalized[representatives[motif]] += count
        motif_lists.append(normalized)

    motifs_to_report = set(expected_motifs)

    # Build motif -> list of values (missing counts as 0)
    motif_to_counts = {motif: [motifs.get(motif, 0) for motifs in motif_lists] for motif in sum(motif_lists, Counter())}

    medians = {motif: int(np.median(counts)) for motif, counts in motif_to_counts.items()}
    if medians:
        # Choose the motif with the highest count that would be reported, rather than the highest total across reads.
        most_common_motif = min(medians, key=lambda motif: (-medians[motif], motif))
        motifs_to_report.add(most_common_motif)

    # Decide which additional motifs to report based on agreement across reads
    for motif, counts in motif_to_counts.items():
        median = medians[motif]
        if motif in motifs_to_report or median == 0:
            continue
        # Require at least 80% of the reads to have approximately the count we will report...
        tolerance = median / 4
        agrees = [abs(count - median) <= tolerance for count in counts]
        if 5 * sum(agrees) >= 4 * len(group) and all(
            # ... and independent support from both sequencing strands
            sum(agrees[i] for i, read in enumerate(group) if read.strand == strand) >= 2
            for strand in ("+", "-")
        ):
            motifs_to_report.add(motif)

    return {motif: median for motif, median in sorted(medians.items()) if motif in motifs_to_report and median > 0}


def _quantile_as_int(values: Sequence[int], quantile: float) -> int:
    return int(np.quantile(np.asarray(values, dtype=float), q=quantile))


def _compute_pooled_5mc_summary(group: Sequence[ReadCall]) -> tuple[float | None, int | None]:
    """Return the pooled methylated fraction and number of callable CpG sites.

    Both values are None if no read of the group has a callable CpG site.
    """
    counts = [record.mc_counts for record in group]
    called = sum(called for _, called in counts)
    if not called:
        return None, None
    return sum(methylated for methylated, _ in counts) / called, called


def _validate_single_region(records: Sequence[ReadCall]) -> None:
    region_names = {record.region_name for record in records}
    if len(region_names) > 1:
        raise ValueError(f"Expected records from a single region. Found: {sorted(region_names)}")


def add_per_allele_stats(  # noqa: C901
    records: Sequence[ReadCall],
    cfg: MahiMahiConfig,
    *,
    phased: bool = False,
    clustering_state: ClusteringState = ClusteringState.AMBIGUOUS,
) -> RegionCallSet:
    """Calculate grouped per-allele statistics for one region."""
    if not records:
        return RegionCallSet.empty()

    _validate_single_region(records)
    selected_reads = [record.clone() for record in records if record.included_in_allele_size_estimation]
    not_selected_reads = [record.clone() for record in records if not record.included_in_allele_size_estimation]
    output_reads = selected_reads + not_selected_reads
    if not selected_reads:
        return RegionCallSet.from_reads(output_reads, phased=phased, clustering_state=clustering_state)

    # Copy number quantile statistics
    grouped_reads: dict[str, list[ReadCall]] = defaultdict(list)
    for record in selected_reads:
        grouped_reads[record.allele_id or ""].append(record)

    group_quantiles: dict[str, dict[int, int]] = {}
    top_q50 = 0
    for allele_id, group in grouped_reads.items():
        copy_numbers = [record.copy_number for record in group]
        quantiles = {
            10: _quantile_as_int(copy_numbers, 0.10),
            25: _quantile_as_int(copy_numbers, 0.25),
            50: _quantile_as_int(copy_numbers, 0.50),
            75: _quantile_as_int(copy_numbers, 0.75),
            90: _quantile_as_int(copy_numbers, 0.90),
        }
        group_quantiles[allele_id] = quantiles
        if allele_id != "0":
            top_q50 = max(top_q50, quantiles[50])

    allele_summaries: dict[str, AlleleSummary] = {}
    for allele_id, group in grouped_reads.items():
        allele_quantiles = group_quantiles[allele_id]
        allele_q50 = allele_quantiles[50]
        allele_q75 = allele_quantiles[75]
        # Use q75 where there is a high fraction of non-spanning reads, to partially allow for them
        # only being lower bounds on TR length.
        if sum(record.is_spanning for record in group) / len(group) <= 0.5:
            for record in group:
                record.allele_size_estimation_quantile = "allele_q75"
        # this exists to pick the longest rather than middle peak of 3-peaked (partial somatic expansion) cases
        # q75/q50 > 2 (default) = wider than a normal (single-peaked) long expansion's variability (q75/q50 <~ 1.5)
        # count >= 5 = allele_q75 is at or below the second-longest read, i.e. not influenced by a single outlier read
        is_very_wide_cluster = allele_q75 is not None and allele_q75 >= allele_q50 * cfg.very_wide_cluster_minimum_q75_q50_ratio and len(group) >= 5
        # don't apply this to the shorter cluster, because of the risk that q75 will be in misclassified
        # reads that should be in the longer cluster (allele 0 is unclassified reads, not a cluster)
        is_top_cluster = allele_id != "0" and allele_q50 == top_q50
        if is_very_wide_cluster and is_top_cluster:
            for record in group:
                record.allele_size_estimation_quantile = "allele_q75"

        quantiles_to_report = {record.allele_size_estimation_quantile for record in group}
        if len(quantiles_to_report) != 1:
            raise ValueError(f"Each allele group must have exactly one allele_size_estimation_quantile. Found: {sorted(quantiles_to_report)}")
        quantile_to_report = next(iter(quantiles_to_report))
        try:
            quantile_as_number = float(str(quantile_to_report).replace("allele_q", "")) / 100
        except ValueError as exc:
            raise ValueError(f"Invalid allele_size_estimation_quantile value: {quantile_to_report}") from exc
        if not 0 <= quantile_as_number <= 1:
            raise ValueError(f"Out-of-range allele_size_estimation_quantile value: {quantile_to_report}")

        # Get the copy number and size-in-bases values for the selected quantile
        copy_numbers = [record.copy_number for record in group]
        sizes = [record.size for record in group]
        allele_copy_number = _quantile_as_int(copy_numbers, quantile_as_number)
        allele_size = _quantile_as_int(sizes, quantile_as_number)
        allele_std = float("nan") if len(copy_numbers) < 2 else float(np.std(np.asarray(copy_numbers, dtype=float), ddof=1))
        allele_num_nonspanning_reads_used = sum(not record.is_spanning for record in group)
        motifs_median_counts = _compute_motif_medians(group)
        pooled_5mc_fraction, pooled_5mc_n = _compute_pooled_5mc_summary(group)
        group_quantiles_for_allele = group_quantiles[allele_id]
        allele_summary = AlleleSummary(
            allele_id=allele_id,
            copy_number=allele_copy_number,
            size=allele_size,
            stddev=allele_std,
            q10=group_quantiles_for_allele[10],
            q25=group_quantiles_for_allele[25],
            q50=group_quantiles_for_allele[50],
            q75=group_quantiles_for_allele[75],
            q90=group_quantiles_for_allele[90],
            num_reads_used=len(group),
            num_nonspanning_reads_used=allele_num_nonspanning_reads_used,
            motifs_median_counts=motifs_median_counts,
            pooled_5mc_fraction=pooled_5mc_fraction,
            pooled_5mc_n=pooled_5mc_n,
            quantile_used=quantile_to_report,
        )
        allele_summaries[allele_id] = allele_summary
    genotype = format_genotype(output_reads, allele_summaries)
    region = RegionCallSet.from_reads(
        output_reads,
        alleles=allele_summaries,
        phased=phased,
        genotype=genotype,
        clustering_state=clustering_state,
    )
    return region


def summarize_assigned_reads(
    records: Sequence[ReadCall],
    cfg: MahiMahiConfig,
    *,
    phased: bool,
    clustering_state: ClusteringState,
) -> RegionCallSet:
    """Calculate section-specific allele statistics and reconcile strand bias."""
    diagnostic_reads = [record.clone() for record in records if record.analysis_exclusion_reason is not None]
    analysis_records = [record for record in records if record.analysis_exclusion_reason is None]
    if not analysis_records:
        return RegionCallSet.from_reads(
            diagnostic_reads,
            phased=phased,
            clustering_state=clustering_state,
        )
    records_unfiltered = [record.clone() for record in analysis_records]
    records_filtered = [record.clone() for record in analysis_records]
    for record in records_filtered:
        if record.is_suspicious:
            record.included_in_allele_size_estimation = False
    region_unfiltered = add_per_allele_stats(
        records_unfiltered,
        cfg,
        phased=phased,
        clustering_state=clustering_state,
    )
    region_filtered = add_per_allele_stats(
        records_filtered,
        cfg,
        phased=phased,
        clustering_state=clustering_state,
    )
    return _append_diagnostic_reads(check_strand_bias_state(region_unfiltered, region_filtered, cfg), diagnostic_reads)


def is_haplotagged(records: Sequence[ReadCall], sex: str) -> bool:
    """
    Check if the input BAM is haplotagged by verifying if any record has a non-zero original haplotype.

    Args:
        records (Sequence[ReadCall]): Typed processed read statistics.

    Returns:
        bool: True if the BAM is haplotagged, False otherwise.
    """
    checkable_haps = [
        record.original_hap for record in records if record.analysis_exclusion_reason is None and (sex == "XX" or not is_sex_chr(record.chrom))
    ]
    if len(checkable_haps) == 0:
        logging.info("The input record set is empty, or is a male sample with only chrX/Y loci. Cannot determine if the BAM is haplotagged.")
        return True
    if any(hap != "0" for hap in checkable_haps):
        logging.info("The input BAM is haplotagged.")
        return True
    else:
        return False

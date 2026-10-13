"""Locus-level phase selection, filtering, and allele assignment."""

import logging
from collections import Counter
from collections.abc import Sequence

from .clustering import cluster_reads, mark_ambiguous_if_homozygous_with_many_non_spanning_reads
from .config import MahiMahiConfig
from .pipeline_records import AlleleAssignmentContext, ClusteringState, MergedLocusRead
from .utils import is_sex_chr


def select_phase_set(records: Sequence[MergedLocusRead]) -> str | None:
    """Return the best-supported phase set, deprioritizing ``ps0`` and breaking ties by input order."""
    if not records:
        return None
    ps_counts = Counter(record.ps for record in records)
    scored_counts = {ps: (-1 if ps == "ps0" else count) for ps, count in ps_counts.items()}
    max_support = max(scored_counts.values())
    return next(record.ps for record in records if scored_counts[record.ps] == max_support)


def drop_other_phasesets(
    records: Sequence[MergedLocusRead],
    context: AlleleAssignmentContext,
    selected_ps: str,
) -> list[MergedLocusRead]:
    """Drop reads outside the selected phase set while retaining unphased reads."""
    if not records:
        return []
    ps_filter = [record.ps in {selected_ps, "ps0"} for record in records]
    number_filtered_reads = sum(not keep for keep in ps_filter)
    if number_filtered_reads > 0:
        phase_sets = {record.ps for record in records if record.ps != "ps0"}
        logging.warning(
            f"Region {context.region_name} has multiple phasesets {phase_sets}: "
            f"discarding {number_filtered_reads} reads that are not in the majority phaseset {selected_ps} or ps0"
        )
    return [record for record, keep in zip(records, ps_filter, strict=True) if keep]


def assign_reads_to_alleles(
    records: Sequence[MergedLocusRead],
    cfg: MahiMahiConfig,
    sex: str,
    no_clustering: bool,
    *,
    context: AlleleAssignmentContext,
    debug_output_basename: str | None = None,
) -> tuple[list[MergedLocusRead], bool, ClusteringState]:
    """Select a phase set and assign one locus's reads to alleles."""
    if not records:
        return [], False, ClusteringState.AMBIGUOUS

    selected_ps = select_phase_set(records)
    assert selected_ps is not None
    assigned_reads = drop_other_phasesets(records, context, selected_ps)
    if not no_clustering:
        assigned_reads, phased, clustering_state = cluster_reads(
            assigned_reads,
            sex,
            min_read_support_per_allele=cfg.min_read_support_per_allele,
            non_spanning_min_extra_length=cfg.min_flank_length,
            non_spanning_read_policy=cfg.clustering.non_spanning_read_policy,
            exclude_haplotag_inconsistent_reads=cfg.clustering.exclude_haplotag_inconsistent_reads,
            hap_max_mismatch_fraction=cfg.clustering.hap_max_mismatch_fraction,
            selected_ps=selected_ps,
            context=context,
            plots_basename=debug_output_basename,
            plot_first_attempt_density=cfg.clustering.plot_first_attempt_density,
        )
    else:
        assigned_reads = [record.clone(allele_id=record.original_hap) for record in assigned_reads]
        phased = True
        clustering_state = ClusteringState.HAPLOTAG_FALLBACK
        if all(record.original_hap == "0" for record in assigned_reads):
            logging.warning(f"All reads in {context.region_name} lack haplotags and clustering is disabled: treating as homozygous")
            assigned_reads = [record.clone(allele_id="1") for record in assigned_reads]
            phased = False
            clustering_state = ClusteringState.AMBIGUOUS
        clustering_state = mark_ambiguous_if_homozygous_with_many_non_spanning_reads(
            assigned_reads,
            clustering_state,
            is_haploid_locus=sex == "XY" and is_sex_chr(context.chrom),
        )
    return assigned_reads, phased, clustering_state

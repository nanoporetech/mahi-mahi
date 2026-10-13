import logging

import pysam

from .allele_assignment import assign_reads_to_alleles, drop_other_phasesets, select_phase_set
from .basecaller_error_detection import annotate_basecaller_strand_errors
from .config import MahiMahiConfig
from .io_utils.bed_utils import MergedLocus
from .locus_with_flanks import LocusWithFlanks
from .pipeline_records import AlleleAssignmentContext, ClusteringState, ReadCall, RegionCallSet
from .read_processing import build_merged_locus_read, main_section_kmer_length, split_merged_locus_read
from .stats import summarize_assigned_reads


def warn_if_many_reads_skipped_or_failed(
    skipped_or_failed_reads: list[pysam.AlignedSegment], successful_read_ids: set[str], locus: MergedLocus, cfg: MahiMahiConfig
) -> int:
    """
    Log a warning if a large fraction of reads were skipped or failed processing, and return the count of such reads.

    Args:
        skipped_or_failed_reads (list[pysam.AlignedSegment]): List of reads that were skipped or failed processing.
        successful_read_names (set[str]): Set of read names that were successfully processed.
        locus (MergedLocus): The TR locus being processed.
        cfg (MahiMahiConfig): Configuration object containing parameters for processing.
    """
    skipped_or_failed_read_ids = set()
    for read_aln_seg in skipped_or_failed_reads:
        # don't count reads that align to just the TR region as failed:
        # repetitive sequence from elsewhere in the genome often does this (particularly at TBP)
        if (read_aln_seg.reference_start <= locus.start - cfg.min_flank_length) or (
            read_aln_seg.reference_end is not None and read_aln_seg.reference_end >= locus.end + cfg.min_flank_length
        ):
            skipped_or_failed_read_ids.add(read_aln_seg.query_name)
    # exclude reads_seen_before from skipped_or_failed_reads to avoid counting reads that have
    # both a successful and a failed alignment (e.g. primary + supplementary) as failed
    skipped_or_failed_read_ids = skipped_or_failed_read_ids - successful_read_ids
    # > means we don't warn when there are 0 reads of any kind - this is deliberate
    # to avoid warnings when running single-locus data
    if len(skipped_or_failed_read_ids) > 0.5 * len(successful_read_ids):
        logging.warning(
            f"More than 1/3 of the reads ({len(skipped_or_failed_read_ids)} out of "
            f"{len(successful_read_ids) + len(skipped_or_failed_read_ids)}) "
            f"for region {locus.repeat_id} ({locus.chrom}:{locus.start}-{locus.end}) "
            "are secondary alignments or failed processing, so were ignored.  "
            "This high number may mean this region has a paralog it is not reliably distinguished from: "
            "such regions cannot be genotyped accurately by mahi-mahi."
        )
    elif len(skipped_or_failed_read_ids) > 0:
        logging.debug(
            f"{len(skipped_or_failed_read_ids)} out of {len(successful_read_ids) + len(skipped_or_failed_read_ids)} "
            f"for region {locus.repeat_id} ({locus.chrom}:{locus.start}-{locus.end}) "
            "are secondary alignments or failed processing, so were ignored."
        )
    return len(skipped_or_failed_read_ids)


def process_locus(
    bam: pysam.AlignmentFile,
    locus: MergedLocus,
    sex: str,
    unsplit_2d_reads: set[str],
    cfg: MahiMahiConfig,
    fastaFile: pysam.FastaFile,
    no_clustering: bool,
    *,
    debug_output_basename: str | None = None,
) -> tuple[list[ReadCall], bool, ClusteringState]:
    """Process, annotate, and assign reads for one complete TR locus.

    Reads are decomposed and checked for strand bias at the merged-locus level, then assigned
    to alleles before composite loci are split into section-specific records. This preserves one
    phase and clustering decision across every section of a composite locus.

    Args:
        bam (pysam.AlignmentFile): BAM file containing aligned reads.
        locus (MergedLocus): The TR locus to process.
        sex (str): Sex of the sample ("XY" or "XX").
        unsplit_2d_reads (Set[str]): Set of unsplit 2D reads to exclude supplementary alignments.
        cfg: Typed configuration with parameters for processing.
        fastaFile (pysam.FastaFile): FASTA file object for fetching reference sequences.
        no_clustering: Assign alleles directly from haplotags instead of clustering read lengths.
        debug_output_basename: Optional basename for the locus-level clustering plot.

    Returns:
        Section reads, the shared locus-level phase-assignment decision, and how the alleles were assigned.
    """

    locus_with_flanks = LocusWithFlanks.from_locus(locus, cfg.extension, fastaFile, cfg.max_flank_length)
    context = AlleleAssignmentContext(
        chrom=locus.chrom,
        region_name=f"{locus.chrom}_{locus.start}_{locus.end}",
        clustering_id=locus.repeat_id,
    )
    # reads that were successfully processed for analysis
    merged_reads = []
    reads_seen_before = set()
    diagnostic_candidates = {}
    # reads that were excluded (e.g. secondary alignments, failed flank realignment, non-spanning and not repetitive)
    # but _not_ skipped duplicates
    # Used to warn when there are unusually many of these
    skipped_or_failed_reads = []
    for read_aln_seg in bam.fetch(locus.chrom, locus.start, locus.end):
        # reads_seen_before doesn't replace this check, as we want the primary alignment to be
        # processed even if a supplementary one was seen first
        if read_aln_seg.query_name in unsplit_2d_reads and read_aln_seg.is_supplementary:
            logging.debug(
                f"Skipping unsplit-2d supplementary alignment of read {read_aln_seg.query_name} in region {locus.chrom}_{locus.start}_{locus.end}"
            )
            continue
        if read_aln_seg.query_name in reads_seen_before:
            logging.debug(f"Skipping duplicate read {read_aln_seg.query_name} in region {locus.chrom}_{locus.start}_{locus.end}")
            continue
        merged_read = build_merged_locus_read(
            read_aln_seg,
            locus_with_flanks,
            sex,
            cfg,
        )
        # Only mark analytical reads as seen. A diagnostic candidate is retained separately so a
        # later analytical alignment for the same read can replace it.
        # (supplementary alignments may only have the part of the read sequence that was initially aligned,
        # so it's possible, though unlikely, that another alignment of the same read could be realignable
        # where this one wasn't)
        if merged_read is not None:
            if merged_read.analysis_exclusion_reason is None:
                reads_seen_before.add(read_aln_seg.query_name)
                merged_reads.append(merged_read)
                diagnostic_candidates.pop(read_aln_seg.query_name, None)
            else:
                diagnostic_candidates.setdefault(read_aln_seg.query_name, merged_read)
                skipped_or_failed_reads.append(read_aln_seg)
        else:
            skipped_or_failed_reads.append(read_aln_seg)
    merged_reads.extend(diagnostic_candidates.values())
    assert len(reads_seen_before) == sum(record.analysis_exclusion_reason is None for record in merged_reads)
    warn_if_many_reads_skipped_or_failed(skipped_or_failed_reads, reads_seen_before, locus, cfg)
    if not merged_reads:
        return [], False, ClusteringState.AMBIGUOUS
    analysis_reads = [record for record in merged_reads if record.analysis_exclusion_reason is None]
    diagnostic_reads = [record for record in merged_reads if record.analysis_exclusion_reason is not None]
    if analysis_reads:
        selected_ps = select_phase_set(analysis_reads)
        assert selected_ps is not None
        annotated_analysis_reads = annotate_basecaller_strand_errors(
            analysis_reads,
            kmer_coverage_threshold=cfg.strand_bias_kmer_coverage_threshold,
            kmer_length=main_section_kmer_length(locus),
            run_min_tr_length=cfg.strand_bias_run_min_tr_length,
            run_min_long_tr_fraction=cfg.strand_bias_run_min_long_tr_fraction,
            strand_bias_method=cfg.strand_bias_method,
            merged_locus_label=f"{locus.chrom}:{locus.start}-{locus.end}",
        )
        phase_filtered_reads = drop_other_phasesets(annotated_analysis_reads, context, selected_ps)
        phase_diagnostic_reads = [
            record.clone(
                analysis_exclusion_reason="non_selected_phaseset",
                clustered=False,
                allele_id=None,
                included_in_allele_size_estimation=False,
            )
            for record in annotated_analysis_reads
            if record.ps not in {selected_ps, "ps0"}
        ]
        diagnostic_reads.extend(phase_diagnostic_reads)
        assigned_reads, phased, clustering_state = assign_reads_to_alleles(
            phase_filtered_reads,
            cfg,
            sex,
            no_clustering,
            context=context,
            debug_output_basename=debug_output_basename,
        )
    else:
        assigned_reads = []
        phased = False
        clustering_state = ClusteringState.AMBIGUOUS
    section_reads: list[ReadCall] = []
    for merged_read in [*assigned_reads, *diagnostic_reads]:
        section_reads.extend(split_merged_locus_read(merged_read, locus_with_flanks, cfg))
    return section_reads, phased, clustering_state


def process_regions(
    loci: list[MergedLocus],
    bamfile: str,
    no_clustering: bool,
    sex: str,
    unsplit_2d_reads: set[str],
    cfg: MahiMahiConfig,
    fastaFile: pysam.FastaFile,
    *,
    debug_output_basename: str | None = None,
) -> list[RegionCallSet]:
    """
    Process reads aligned to every TR region.

    Returns:
        list[RegionCallSet]: Grouped region results for all TR regions.
    """

    processed_records: list[RegionCallSet] = []

    with pysam.AlignmentFile(bamfile, "rb") as bam:
        # Iterate over merged loci.
        for locus in loci:
            if locus.chrom not in bam.references:
                logging.warning(
                    f"Chromosome {locus.chrom} not found in BAM file (name format mismatch? expected "
                    f"{bam.references[0] if bam.references else ''} format). Skipping locus."
                )
                continue
            assigned_reads, phased, clustering_state = process_locus(
                bam,
                locus,
                sex,
                unsplit_2d_reads,
                cfg,
                fastaFile,
                no_clustering,
                debug_output_basename=debug_output_basename,
            )
            if not assigned_reads:
                logging.warning(f"No reads found for {locus.chrom}:{locus.start}-{locus.end} with motif {locus.motif}")
                continue
            # Unmerge typed records back into per-region groups while preserving input order.
            records_by_region: dict[str, list[ReadCall]] = {}
            for record in assigned_reads:
                records_by_region.setdefault(record.region_name, []).append(record)
            for cur_seg_records in records_by_region.values():
                if cur_seg_records:
                    processed_records.append(
                        summarize_assigned_reads(
                            cur_seg_records,
                            cfg,
                            phased=phased,
                            clustering_state=clustering_state,
                        )
                    )
    if not processed_records:
        logging.warning("No reads found in any of the specified TR regions.")
        return []
    return processed_records

#!/usr/bin/env python
"""
Basecaller Error Detection Module

This module provides functions for detecting basecaller errors through strand bias analysis
of k-mer patterns in sequencing data. It implements two complementary approaches:

1. Statistical Testing Method: Uses rigorous statistical filtering with Fisher's exact test,
   G-test for independence, and FDR correction to identify error-signature k-mers.

2. Counts Threshold Method: Uses simple k-mer filtering and presence/absence comparison
   between strands to identify strand-specific k-mers.

The module is specifically designed for detecting systematic basecaller errors that manifest
as strand-specific biases in k-mer distributions, which can affect the accuracy of STR
genotyping and other genomic analyses.
"""

import logging
from collections import Counter
from collections.abc import Sequence

from .config import MahiMahiConfig
from .pipeline_records import AlleleSummary, MergedLocusRead, ReadCall, RegionCallSet, format_genotype
from .strand_bias_analysis import error_kmer_sets_strict, g_test_region
from .utils import count_all_kmers, filter_top_kmers_by_percentage


def analyze_strand_specific_motifs_statistical_testing(positive_motif_counts: dict[str, int], negative_motif_counts: dict[str, int]) -> set[str]:
    """
    Analyze strand-specific motifs to identify error-signature k-mers using rigorous statistical methods.

    This function aggregates normalized motif counts separately for positive and negative strands,
    performs a G-test for independence to assess strand bias, and uses rigorous statistical
    filtering to identify k-mers that show systematic strand asymmetry indicative of
    sequencing errors or technical artifacts.

    The statistical approach includes:
    - G-test for independence between k-mer counts and strand orientation
    - Fisher's exact test for individual k-mer significance
    - Benjamini-Hochberg FDR correction for multiple testing
    - Effect size filtering and prevalence gates

    Returns:
        Set[str]: Set of error-signature k-mers that show significant strand bias
    """
    if not positive_motif_counts and not negative_motif_counts:
        return set()

    logging.debug(f"Positive motif counts: {positive_motif_counts}")
    logging.debug(f"Negative motif counts: {negative_motif_counts}")

    # Perform G-test for overall strand independence
    region = g_test_region(positive_motif_counts, negative_motif_counts, keep_mass=0.99, n_perm=1000)
    logging.debug(f"Region G: {region['G']}, df: {region['df']}, p_mc: {region['p_mc']}")

    # Identify error-signature k-mers using rigorous statistical filtering
    suspicious_kmers_plus, suspicious_kmers_minus = error_kmer_sets_strict(positive_motif_counts, negative_motif_counts)

    if suspicious_kmers_minus or suspicious_kmers_plus:
        logging.debug(f"Error-signature kmers - E_plus: {suspicious_kmers_plus}, E_minus: {suspicious_kmers_minus}")

    return suspicious_kmers_plus | suspicious_kmers_minus


def analyze_strand_specific_motifs_counts_threshold(
    read_kmer_counts: Sequence[tuple[str, dict[str, int]]], *, top_kmers_percentage: float = 90.0, min_kmer_count: int = 10
) -> set[str]:
    """
    Analyze strand-specific motifs using k-mer filtering and simple presence/absence comparison.

    This function first preprocesses normalized k-mer counts by filtering to keep only
    the most frequent k-mers that collectively explain the specified percentage of
    total k-mers. Then it aggregates these filtered motif counts separately for
    positive and negative strands, filters out k-mers below the minimum count threshold,
    and identifies suspicious k-mers by comparing the sets of k-mers present on each strand.

    This method is simpler and faster than statistical testing but may be less precise
    in identifying true error-signature k-mers.

    Parameters:
        read_kmer_counts: Per-read strand and k-mer-count pairs for one merged locus.
        top_kmers_percentage: Percentage threshold (0-100) for k-mer filtering (default: 90.0)
        min_kmer_count: Minimum count threshold for k-mers after aggregation (default: 10)

    Returns:
        Set[str]: Set of error-signature k-mers that show strand-specific presence
    """
    if not read_kmer_counts:
        return set()

    positive_motif_counts: Counter[str] = Counter()
    negative_motif_counts: Counter[str] = Counter()
    # Step 1 + 2: filter each read down to its top k-mers, then aggregate by strand
    for strand, kmer_counts in read_kmer_counts:
        filtered_kmer_counts = filter_top_kmers_by_percentage(kmer_counts, top_kmers_percentage) if kmer_counts else {}
        if strand == "+":
            positive_motif_counts.update(filtered_kmer_counts)
        elif strand == "-":
            negative_motif_counts.update(filtered_kmer_counts)

    # Step 2.5: Filter k-mers below minimum count threshold after aggregation
    positive_motif_counts = Counter({k: v for k, v in positive_motif_counts.items() if v >= min_kmer_count})
    negative_motif_counts = Counter({k: v for k, v in negative_motif_counts.items() if v >= min_kmer_count})

    # Step 3: Identify suspicious k-mers by comparing strand-specific k-mer sets
    positive_kmers = set(positive_motif_counts.keys())
    negative_kmers = set(negative_motif_counts.keys())

    if not positive_kmers or not negative_kmers:
        return set()

    # K-mers that appear only on one strand are suspicious
    positive_only_kmers = positive_kmers - negative_kmers
    negative_only_kmers = negative_kmers - positive_kmers

    # Log results for debugging
    logging.debug(f"Filtered positive motif counts (min_count >= {min_kmer_count}): {positive_motif_counts}")
    logging.debug(f"Filtered negative motif counts (min_count >= {min_kmer_count}): {negative_motif_counts}")
    logging.debug(f"Positive-only k-mers: {positive_only_kmers}")
    logging.debug(f"Negative-only k-mers: {negative_only_kmers}")

    # Return union of strand-specific k-mers
    suspicious_kmers = positive_only_kmers | negative_only_kmers

    if suspicious_kmers:
        logging.debug(f"Strand-specific error k-mers identified: {suspicious_kmers}")

    return suspicious_kmers


def _collect_read_kmer_counts(
    records: Sequence[MergedLocusRead], *, kmer_length: int
) -> tuple[list[tuple[MergedLocusRead, dict[str, int], int]], Counter[str], Counter[str]]:
    read_kmer_counts: list[tuple[MergedLocusRead, dict[str, int], int]] = []
    positive_motif_counts: Counter[str] = Counter()
    negative_motif_counts: Counter[str] = Counter()
    for record in records:
        kmer_counts = count_all_kmers("".join(record.motif_dec), k=kmer_length)
        total_kmers = sum(kmer_counts.values())
        read_kmer_counts.append((record, kmer_counts, total_kmers))
        if record.strand == "+":
            positive_motif_counts.update(kmer_counts)
        elif record.strand == "-":
            negative_motif_counts.update(kmer_counts)
    return read_kmer_counts, positive_motif_counts, negative_motif_counts


def _select_suspicious_motifs(
    read_kmer_counts: Sequence[tuple[MergedLocusRead, dict[str, int], int]],
    positive_motif_counts: Counter[str],
    negative_motif_counts: Counter[str],
    *,
    strand_bias_method: str,
    merged_locus_label: str,
) -> set[str]:
    logging.debug("Analyzing strand-specific motifs using method: %s for merged locus %s", strand_bias_method, merged_locus_label)
    if strand_bias_method == "statistical_testing":
        return analyze_strand_specific_motifs_statistical_testing(dict(positive_motif_counts), dict(negative_motif_counts))
    if strand_bias_method == "counts_threshold":
        return analyze_strand_specific_motifs_counts_threshold(
            [(record.strand, kmer_counts) for record, kmer_counts, _total_kmers in read_kmer_counts]
        )
    logging.warning("Unknown strand_bias_method '%s', defaulting to statistical_testing", strand_bias_method)
    return analyze_strand_specific_motifs_statistical_testing(dict(positive_motif_counts), dict(negative_motif_counts))


def annotate_basecaller_strand_errors(
    records: Sequence[MergedLocusRead],
    *,
    kmer_coverage_threshold: float,
    kmer_length: int,
    run_min_tr_length: int = 0,
    run_min_long_tr_fraction: float = 0.0,
    strand_bias_method: str = "statistical_testing",
    merged_locus_label: str | None = None,
) -> list[MergedLocusRead]:
    """
    Annotate reads with basecaller strand error information and error flags.

    This function uses one of two methods to identify error-signature k-mers that show
    systematic strand bias, then for each read calculates the percentage contribution
    of each suspicious k-mer and determines if the read contains significant basecaller
    strand errors.

    Args:
        records: Whole-read records describing a single merged/composite tandem-repeat locus.
        kmer_coverage_threshold: Minimum coverage percentage (0.0-1.0) a k-mer must have
                                in a read to flag that read as suspicious.
        kmer_length: K-mer length to use for strand-bias profiling.
        run_min_tr_length: Minimum analyzed TR span, in bases, for a read to count toward
                            running strand-bias analysis for this locus.
        run_min_long_tr_fraction: Minimum fraction of reads that must meet
                                   `run_min_tr_length` before the strand-bias analysis runs.
        strand_bias_method: Method to use for strand bias analysis:
                           - "statistical_testing": Uses rigorous statistical filtering with
                             Fisher's exact test and FDR correction (default)
                           - "counts_threshold": Uses simple k-mer filtering and presence/absence
                             comparison between strands
        merged_locus_label: Optional label to use in debug logging.

    Returns:
        list[MergedLocusRead]: Cloned records with `is_suspicious` set for the whole read.
    """
    cloned_records = [record.clone() for record in records]
    if not cloned_records:
        return cloned_records

    merged_locus = merged_locus_label or "merged locus"
    long_enough_tr_reads = sum(record.size >= run_min_tr_length for record in cloned_records)
    long_tr_fraction = long_enough_tr_reads / len(cloned_records)
    if long_tr_fraction < run_min_long_tr_fraction:
        logging.debug(
            "Skipping strand-bias analysis for merged locus %s: long_tr_fraction=%s (%s/%s) < threshold=%s at min_tr_length=%s",
            merged_locus,
            long_tr_fraction,
            long_enough_tr_reads,
            len(cloned_records),
            run_min_long_tr_fraction,
            run_min_tr_length,
        )
        for record in cloned_records:
            record.is_suspicious = False
        return cloned_records

    read_kmer_counts, positive_motif_counts, negative_motif_counts = _collect_read_kmer_counts(cloned_records, kmer_length=kmer_length)
    suspicious_motifs = _select_suspicious_motifs(
        read_kmer_counts,
        positive_motif_counts,
        negative_motif_counts,
        strand_bias_method=strand_bias_method,
        merged_locus_label=merged_locus,
    )
    for record, kmer_counts, total_kmers in read_kmer_counts:
        if total_kmers == 0:
            record.is_suspicious = False
            continue
        motif_error_breakdown = {motif: count / total_kmers for motif, count in kmer_counts.items() if motif in suspicious_motifs}
        total_bad_motif_count = sum(count for motif, count in kmer_counts.items() if motif in suspicious_motifs)
        record.is_suspicious = total_bad_motif_count / total_kmers > kmer_coverage_threshold
        if record.is_suspicious:
            logging.debug(
                "Merged locus %s read %s strand %s suspicious_motifs=%s bad_fraction=%s/%s threshold=%s final_decision=%s",
                merged_locus,
                record.read,
                record.strand,
                motif_error_breakdown,
                total_bad_motif_count,
                total_kmers,
                kmer_coverage_threshold,
                record.is_suspicious,
            )

    return cloned_records


def total_variation_distance(counts1: dict[str, int], counts2: dict[str, int]) -> float:
    """Calculate the total variation distance between two distributions represented as count dictionaries."""
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    if total1 == 0 or total2 == 0:
        return float("nan")  # avoid division by zero
    all_keys = set(counts1.keys()) | set(counts2.keys())
    distance = 0.5 * sum(abs((counts1.get(k, 0) / total1) - (counts2.get(k, 0) / total2)) for k in all_keys)
    return distance


def check_strand_bias_state(  # noqa: C901
    cur_seg_region_unfiltered: RegionCallSet,
    cur_seg_region_filtered: RegionCallSet,
    cfg: MahiMahiConfig,
) -> RegionCallSet:
    """Check how heavily this region is affected by strand bias

    Mark regions as suspicious if they show evidence of strand bias in any of the following ways:
    - they have few non-suspicious reads (either in absolute number or in percentage)
    - the filtered and unfiltered motif compositions differ by more than a certain percentage

    For alleles with particularly strong strand bias, blank the allele's motifs_median_counts entirely,
    as it is likely to be inaccurate

    Return filtered results, unfiltered results, or some of both, depending on config

    Args:
        cur_seg_region_unfiltered (RegionCallSet): Unfiltered grouped region results.
        cur_seg_region_filtered (RegionCallSet): Filtered grouped region results for the same region.
        cfg: Typed configuration with parameters for processing.
    """
    region_is_suspicious = False
    alleles_set: set[str] = {record.allele_id or "0" for record in cur_seg_region_unfiltered.reads}
    alleles_hide_motif_counts: set[str] = set()
    alleles_force_unfiltered: set[str] = set()
    if len(alleles_set) > 1:
        alleles_set = alleles_set - {"0"}  # ignore unclustered reads
    for allele_id in sorted(alleles_set):
        unfiltered_reads_used = [
            record for record in cur_seg_region_unfiltered.reads if record.allele_id == allele_id and record.included_in_allele_size_estimation
        ]
        filtered_reads_used = [
            record for record in cur_seg_region_filtered.reads if record.allele_id == allele_id and record.included_in_allele_size_estimation
        ]
        if len(unfiltered_reads_used) == 0:
            # no size estimate available
            continue
        if len(filtered_reads_used) < min(len(unfiltered_reads_used), cfg.strand_bias_min_remaining_reads_per_allele):
            # some suspicious reads, and not enough non-suspicious reads
            region_is_suspicious = True
            alleles_hide_motif_counts.add(allele_id)
            alleles_force_unfiltered.add(allele_id)
        if len(filtered_reads_used) / len(unfiltered_reads_used) < 1 - cfg.strand_bias_reads_fraction_threshold:
            region_is_suspicious = True
        if len(filtered_reads_used) / len(unfiltered_reads_used) < 1 - cfg.strand_bias_hide_motifs_reads_fraction_threshold:
            alleles_hide_motif_counts.add(allele_id)
        if len(filtered_reads_used) == 0:
            # no filtered size estimate available - avoid crashing on trying to use it
            # the previous sections handle marking such regions as suspicious
            continue
        unfiltered_summary = cur_seg_region_unfiltered.alleles.get(allele_id)
        filtered_summary = cur_seg_region_filtered.alleles.get(allele_id)
        if unfiltered_summary is None or filtered_summary is None:
            raise ValueError(f"Allele summary for allele {allele_id} must be populated before strand-bias reconciliation")
        # compare motif compositions
        unfiltered_motif_counts = dict(unfiltered_summary.motifs_median_counts or {})
        filtered_motif_counts = dict(filtered_summary.motifs_median_counts or {})
        unfiltered_motif_counts["other"] = int(unfiltered_summary.q50) - sum(unfiltered_motif_counts.values())
        filtered_motif_counts["other"] = int(filtered_summary.q50) - sum(filtered_motif_counts.values())
        if total_variation_distance(unfiltered_motif_counts, filtered_motif_counts) > cfg.strand_bias_motif_composition_difference_threshold:
            region_is_suspicious = True
    final_records: list[ReadCall]
    final_alleles: dict[str, AlleleSummary]
    if cfg.strand_bias_exclude_suspicious_reads == "size_and_motifs":
        # we only need to do this here and not in motifs_only because the motifs of force_unfiltered alleles are blanked anyway
        final_records = [record.clone() for record in cur_seg_region_filtered.reads if record.allele_id not in alleles_force_unfiltered] + [
            record.clone() for record in cur_seg_region_unfiltered.reads if record.allele_id in alleles_force_unfiltered
        ]
        final_alleles = {
            allele_id: summary for allele_id, summary in cur_seg_region_filtered.alleles.items() if allele_id not in alleles_force_unfiltered
        }
        final_alleles.update(
            {allele_id: summary for allele_id, summary in cur_seg_region_unfiltered.alleles.items() if allele_id in alleles_force_unfiltered}
        )
    elif cfg.strand_bias_exclude_suspicious_reads == "motifs_only":
        final_records = [record.clone() for record in cur_seg_region_unfiltered.reads]
        final_alleles = {}
        for allele_id, summary in cur_seg_region_unfiltered.alleles.items():
            filtered_summary = cur_seg_region_filtered.alleles.get(allele_id)
            motifs_median_counts = filtered_summary.motifs_median_counts if filtered_summary is not None else None
            pooled_5mc_fraction = filtered_summary.pooled_5mc_fraction if filtered_summary is not None else None
            pooled_5mc_n = filtered_summary.pooled_5mc_n if filtered_summary is not None else None
            final_alleles[allele_id] = summary.model_copy(
                update={
                    "motifs_median_counts": motifs_median_counts,
                    "pooled_5mc_fraction": pooled_5mc_fraction,
                    "pooled_5mc_n": pooled_5mc_n,
                }
            )
    elif cfg.strand_bias_exclude_suspicious_reads == "none":
        final_records = [record.clone() for record in cur_seg_region_unfiltered.reads]
        final_alleles = dict(cur_seg_region_unfiltered.alleles)
    else:
        raise ValueError(
            f"Invalid value for strand_bias_exclude_suspicious_reads: {cfg.strand_bias_exclude_suspicious_reads}. "
            "Expected 'size_and_motifs', 'motifs_only', or 'none'."
        )
    for allele_id in alleles_hide_motif_counts:
        summary = final_alleles.get(allele_id)
        if summary is not None:
            final_alleles[allele_id] = summary.model_copy(update={"motifs_median_counts": None, "pooled_5mc_fraction": None, "pooled_5mc_n": None})
    final_genotype = format_genotype(final_records, final_alleles)
    return RegionCallSet.from_reads(
        final_records,
        alleles=final_alleles,
        phased=cur_seg_region_unfiltered.phased,
        genotype=final_genotype,
        region_is_suspicious=region_is_suspicious,
        clustering_state=cur_seg_region_unfiltered.clustering_state,
    )

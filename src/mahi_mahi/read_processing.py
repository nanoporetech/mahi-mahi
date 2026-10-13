import itertools
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pysam

from .align_utils import align_all_star, compute_boundary, segment_exact_match_fraction
from .config import MahiMahiConfig
from .io_utils.bed_utils import Locus, MergedLocus, select_primary_output_locus
from .locus_validation import primary_section_motif_length
from .locus_with_flanks import LocusWithFlanks
from .methylation import read_cpg_states, split_cpg_states
from .pipeline_records import RETAINED_READ_EXCLUSION_REASONS, MergedLocusRead, ReadCall, ReadExclusionReason
from .utils import (
    expand_motif_representation,
    get_ps_hap,
    is_repeated_sequence,
    process_subread,
)


@dataclass(frozen=True)
class SubreadResult:
    """Result of extracting a locus subread, including retained diagnostic failures."""

    subread: str
    aln_start: int
    is_aligned: bool
    is_spanning: bool
    exclusion_reason: ReadExclusionReason | None = None


def get_subread(
    read_aln_seg: pysam.AlignedSegment,
    locus_with_flanks: LocusWithFlanks,
    extension: int,
    min_flank_length: int,
    flank_realignment_check_epsilon: int,
    repeated_sequence_entropy_threshold: float,
) -> SubreadResult:
    """
    Extract the portion of the query sequence that covers the target region on the reference.

    Alignment pairs derived from the read's CIGAR string anchor the initial boundaries. Each flank sequence
    (left/right) is then realigned to the read. If the initial flank alignment fails, the flank is iteratively
    shrunk by half and realignment is attempted until either a match is found or the flank length falls below
    the minimum threshold. A single longer-flank attempt is then made when possible. When successful, boundaries
    are tightened based on the flank alignment. If no flank can be placed, the read is deemed to be non-spanning
    for that boundary. Reads that fail to span both breakpoints or that produce invalid coordinates are discarded,
    returning an empty subread.

    Args:
        read_aln_seg (pysam.AlignedSegment): The aligned read segment.
        locus_with_flanks (LocusWithFlanks): Locus context containing region coordinates, motifs, and reference flank sequences.
        extension (int): Extension to consider when checking spanning.
        min_flank_length (int): Minimum length of the flank to consider.

    Returns:
        SubreadResult: The extracted sequence and alignment status. Selected hard-clipped and non-repeated
            reads have a usable subread and an exclusion reason so they can be retained for diagnostics.
    """
    logging.debug(f"Getting subread for read {read_aln_seg.query_name}")
    assert min_flank_length > 0, "min_flank_length must be positive"
    assert min_flank_length < min(len(locus_with_flanks.flank_seq_start), len(locus_with_flanks.flank_seq_end)), (
        "min_flank_length must be less than the length of the flank sequences"
    )
    start = locus_with_flanks.locus.start
    end = locus_with_flanks.locus.end
    alignment_pairs = [(cons, ref) for cons, ref in read_aln_seg.get_aligned_pairs() if cons is not None and ref is not None and start <= ref <= end]

    if len(alignment_pairs) == 0:
        return SubreadResult("", 0, False, False)
    if len(alignment_pairs) == 1:
        subread = read_aln_seg.query_sequence[alignment_pairs[0][0]]  # type: ignore
        return SubreadResult(subread, alignment_pairs[0][0], True, False)
    read_len = len(read_aln_seg.query_sequence)  # type: ignore
    aln_start, is_spanning_left, left_realigned, left_realignment_shift = compute_boundary(
        read_aln_seg,
        alignment_pairs,
        start,
        extension,
        locus_with_flanks.flank_seq_start,
        min_flank_length,
        read_len,
        of_left_side=True,
    )
    aln_end, is_spanning_right, right_realigned, right_realignment_shift = compute_boundary(
        read_aln_seg,
        alignment_pairs,
        end,
        extension,
        locus_with_flanks.flank_seq_end,
        min_flank_length,
        read_len,
        of_left_side=False,
    )

    # sometimes we have trouble finding the flank leading to wrong coordinates. I am going to skip these cases
    wrong_coordinates = aln_end <= aln_start or aln_start < 0 or aln_end > read_len

    # Ignore reads that are contained within the region
    if (not is_spanning_left and not is_spanning_right) or wrong_coordinates:
        logging.debug(
            f"Wrong coordinates for read {read_aln_seg.query_name} in region {read_aln_seg.reference_name}:{start}-{end}. "
            f"Start: {aln_start}, End: {aln_end}, Read length: {read_len}"
        ) if wrong_coordinates else None
        return SubreadResult("", 0, False, False)

    subread = read_aln_seg.query_sequence[aln_start:aln_end]  # type: ignore
    is_spanning = is_spanning_left and is_spanning_right
    significantly_realigned = (
        left_realigned and left_realignment_shift is not None and abs(left_realignment_shift) > flank_realignment_check_epsilon
    ) or (right_realigned and right_realignment_shift is not None and abs(right_realignment_shift) > flank_realignment_check_epsilon)
    # ignore reads that are hard-clipped inside the TR (supplementary alignments often are),
    # as they are likely to be hard-clipped to the TR length *in the reference* and would hence bias length estimates
    if (not is_spanning_left) and (read_aln_seg.cigartuples is not None) and (read_aln_seg.cigartuples[0][0] == pysam.CIGAR_OPS.CHARD_CLIP):
        logging.debug(f"Read {read_aln_seg.query_name} is hard-clipped inside the TR on the left side")
        return SubreadResult(subread, aln_start, True, False, "hard_clip_left")
    if (not is_spanning_right) and (read_aln_seg.cigartuples is not None) and (read_aln_seg.cigartuples[-1][0] == pysam.CIGAR_OPS.CHARD_CLIP):
        logging.debug(f"Read {read_aln_seg.query_name} is hard-clipped inside the TR on the right side")
        return SubreadResult(subread, aln_start, True, False, "hard_clip_right")

    # Extra check that flank aligner got the right coordinates
    if significantly_realigned or not is_spanning:
        motif_dec = process_subread(subread, locus_with_flanks.motifs)
        is_repeat_sequence = is_repeated_sequence(
            motif_dec,
            set(locus_with_flanks.motifs),
            threshold=repeated_sequence_entropy_threshold,
        )
        if not is_repeat_sequence:
            logging.debug(
                f"Read {read_aln_seg.query_name} is discard because of failed flank realignment and sequence is not repeat sequence; "
                f"left_shift={left_realignment_shift}, right_shift={right_realignment_shift}, "
                f"left_realigned={left_realigned}, right_realigned={right_realigned}, "
                f"is_spanning_left={is_spanning_left}, is_spanning_right={is_spanning_right}, "
                f"aln_start={aln_start}, aln_end={aln_end}, "
                f"realignment_epsilon={flank_realignment_check_epsilon}, entropy_threshold={repeated_sequence_entropy_threshold}"
            )
            return SubreadResult(subread, aln_start, True, is_spanning, "non_repeated_sequence")

    return SubreadResult(subread, aln_start, True, is_spanning)


def _build_subregion_read_call(
    *,
    merged_read: MergedLocusRead,
    subregion_motif_dec: list[str],
    subregion_read_start: int,
    subregion_mc_sites: str | None,
    unmerged_locus: Locus,
) -> ReadCall:
    subregion_length = sum(len(this_motif) for this_motif in subregion_motif_dec)
    subregion_copy_number = len(subregion_motif_dec)
    if "nonrepeat" in unmerged_locus.variant_id and subregion_copy_number > 1:
        logging.warning(
            f"Read {merged_read.read} has copy number {subregion_copy_number} >1 for locus {unmerged_locus.variant_id},"
            f" which is not expected to be repeated: this may indicate misclassification of repeat sequence that resembles it"
        )
    if "nonrepeat" in unmerged_locus.variant_id and subregion_copy_number < 1 and merged_read.is_spanning:
        logging.warning(
            f"Read {merged_read.read} is missing locus {unmerged_locus.variant_id},"
            f" which is expected to be single-copy: this may indicate misclassification of it as part of a repeat"
        )

    # rotations and rare unexpected motifs are collapsed per allele (stats._compute_motif_medians), not here
    motif_counts = dict(Counter(subregion_motif_dec))
    return ReadCall(
        chrom=unmerged_locus.chrom,
        start=unmerged_locus.start,
        end=unmerged_locus.end,
        repeat_id=unmerged_locus.repeat_id,
        variant_id=unmerged_locus.variant_id,
        repeat_unit=unmerged_locus.motif,
        read=merged_read.read,
        copy_number=subregion_copy_number,
        size=subregion_length,
        read_start=subregion_read_start,
        strand=merged_read.strand,
        ps=merged_read.ps,
        original_hap=merged_read.original_hap,
        region_name=f"{unmerged_locus.chrom}_{unmerged_locus.start}_{unmerged_locus.end}",
        dec="".join([f"{k}({len(list(group))})" for k, group in itertools.groupby(subregion_motif_dec)]),
        lst_dec=",".join(subregion_motif_dec),
        motif_counts=motif_counts,
        clustered=merged_read.clustered,
        is_spanning=merged_read.is_spanning,
        included_in_allele_size_estimation=merged_read.included_in_allele_size_estimation,
        allele_size_estimation_quantile=merged_read.allele_size_estimation_quantile,
        is_suspicious=merged_read.is_suspicious,
        allele_id=merged_read.allele_id,
        analysis_exclusion_reason=merged_read.analysis_exclusion_reason,
        mc_sites=subregion_mc_sites,
    )


def _composite_aba_middle_idx(unmerged_loci: Sequence[Locus]) -> int | None:
    """Return the middle index for eligible three-part ABA composites."""
    if len(unmerged_loci) != 3:
        return None
    if unmerged_loci[0].motif != unmerged_loci[2].motif:
        return None
    # Reuse the shared primary-output validation so ambiguous custom naming fails
    # consistently with other composite-locus checks, while keeping the existing
    # ABA rule that only the literal repeat_id middle segment is eligible.
    select_primary_output_locus(
        unmerged_loci,
        context=f"composite locus {unmerged_loci[0].repeat_id} for ABA exact-match check",
    )
    if unmerged_loci[1].variant_id != unmerged_loci[1].repeat_id:
        return None
    return 1


def _apply_aba_middle_exact_match_check(
    pattern: Sequence[Sequence[str]],
    motif_dec: list[str],
    segments: list[list[str]],
    middle_idx: int | None,
    aba_middle_exact_fraction_threshold: float,
) -> list[list[str]]:
    """Collapse eligible ABA composites to the left segment when the middle split is a poor exact match."""
    if middle_idx is None:
        return segments
    middle_exact_fraction = segment_exact_match_fraction(pattern[middle_idx], segments[middle_idx])
    if middle_exact_fraction < aba_middle_exact_fraction_threshold:
        return [motif_dec, [], []]
    return segments


def _hard_clip_length_start(read_aln_seg: pysam.AlignedSegment) -> int:
    if read_aln_seg.cigartuples is None:
        return 0
    if read_aln_seg.cigartuples[0][0] != pysam.CIGAR_OPS.CHARD_CLIP:
        return 0
    return read_aln_seg.cigartuples[0][1]


def _build_merged_locus_read_from_subread(
    read_aln_seg: pysam.AlignedSegment,
    subread_seq: str,
    aln_start: int,
    is_spanning: bool,
    locus_with_flanks: LocusWithFlanks,
    sex: str,
    methylation_state_confidence_threshold: float,
    exclusion_reason: ReadExclusionReason | None = None,
) -> MergedLocusRead:
    if exclusion_reason is not None and exclusion_reason not in RETAINED_READ_EXCLUSION_REASONS:
        raise ValueError(f"Unsupported retained read exclusion reason: {exclusion_reason}")
    read_name = read_aln_seg.query_name
    if read_name is None:
        raise ValueError("Aligned segment is missing query_name")
    cur_ps, cur_hap = get_ps_hap(read_aln_seg, locus_with_flanks.locus.chrom, sex)
    return MergedLocusRead(
        read=read_name,
        strand="+" if not read_aln_seg.is_reverse else "-",
        ps=cur_ps,
        original_hap=cur_hap,
        motif_dec=tuple(process_subread(subread_seq, locus_with_flanks.motifs)),
        read_start=_hard_clip_length_start(read_aln_seg) + aln_start,
        mc_sites=read_cpg_states(read_aln_seg, subread_seq, aln_start, methylation_state_confidence_threshold),
        is_spanning=is_spanning,
        included_in_allele_size_estimation=is_spanning and exclusion_reason is None,
        analysis_exclusion_reason=exclusion_reason,
    )


def build_merged_locus_read(
    read_aln_seg: pysam.AlignedSegment,
    locus_with_flanks: LocusWithFlanks,
    sex: str,
    cfg: MahiMahiConfig,
) -> MergedLocusRead | None:
    if read_aln_seg.is_secondary:
        return None
    if read_aln_seg.query_sequence is None and not read_aln_seg.is_unmapped:
        logging.warning(f"{read_aln_seg.query_name} is mapped, but contains no query_sequence and not secondary")
        return None
    # the above skips mapped reads with no query_sequence, but unmapped reads with no query_sequence are currently allowed to crash
    # (I think currently get_subread will raise "TypeError: 'NoneType' object is not subscriptable", but I haven't tried it)
    subread_result = get_subread(
        read_aln_seg,
        locus_with_flanks,
        cfg.extension,
        min_flank_length=cfg.min_flank_length,
        flank_realignment_check_epsilon=cfg.flank_realignment_check_epsilon,
        repeated_sequence_entropy_threshold=cfg.repeated_sequence_entropy_threshold,
    )
    subread_seq = subread_result.subread
    aln_start = subread_result.aln_start
    is_aligned = subread_result.is_aligned
    is_spanning = subread_result.is_spanning
    exclusion_reason = subread_result.exclusion_reason
    if not is_aligned:
        return None
    return _build_merged_locus_read_from_subread(
        read_aln_seg,
        subread_seq,
        aln_start,
        is_spanning,
        locus_with_flanks,
        sex,
        cfg.methylation_state_confidence_threshold,
        exclusion_reason,
    )


def _split_composite_motif_dec(
    motif_dec: Sequence[str],
    locus_with_flanks: LocusWithFlanks,
    cfg: MahiMahiConfig,
) -> list[list[str]]:
    motif_dec = list(motif_dec)
    if len(locus_with_flanks.locus.loci) == 1:
        # single-TR locus - splitting would break ambiguous motif support (and also waste time)
        return [motif_dec]
    main_locus = select_primary_output_locus(
        locus_with_flanks.locus.loci,
        context=f"composite locus {locus_with_flanks.locus.repeat_id} for split scoring",
    )
    main_idx = locus_with_flanks.locus.loci.index(main_locus)
    section_motif_groups = [tuple(expand_motif_representation(locus.motif)) for locus in locus_with_flanks.locus.loci]
    _, subregion_motif_decs = align_all_star(section_motif_groups, motif_dec, main_idx=main_idx)
    return _apply_aba_middle_exact_match_check(
        section_motif_groups,
        motif_dec,
        subregion_motif_decs,
        _composite_aba_middle_idx(locus_with_flanks.locus.loci),
        cfg.aba_middle_exact_fraction_threshold,
    )


def split_merged_locus_read(
    merged_read: MergedLocusRead,
    locus_with_flanks: LocusWithFlanks,
    cfg: MahiMahiConfig,
) -> list[ReadCall]:
    """Split one merged-locus read into per-subregion read calls."""
    result: list[ReadCall] = []
    subregion_motif_decs = _split_composite_motif_dec(merged_read.motif_dec, locus_with_flanks, cfg)
    subregion_mc_sites = split_cpg_states(
        merged_read.mc_sites,
        "".join(merged_read.motif_dec),
        [sum(len(this_motif) for this_motif in subregion_motif_dec) for subregion_motif_dec in subregion_motif_decs],
    )
    subregion_read_start = merged_read.read_start
    # iterate over the individual TRs in the merged locus
    for locus_idx, unmerged_locus in enumerate(locus_with_flanks.locus.loci):
        subregion_motif_dec = subregion_motif_decs[locus_idx]
        result.append(
            _build_subregion_read_call(
                merged_read=merged_read,
                subregion_motif_dec=subregion_motif_dec,
                subregion_read_start=subregion_read_start,
                subregion_mc_sites=subregion_mc_sites[locus_idx],
                unmerged_locus=unmerged_locus,
            )
        )
        subregion_read_start = subregion_read_start + sum(len(this_motif) for this_motif in subregion_motif_dec)
    # length consistency check
    if subregion_read_start != merged_read.read_start + sum(len(this_motif) for this_motif in merged_read.motif_dec):
        raise RuntimeError(
            f"Length mismatch for read {merged_read.read}: {subregion_read_start} "
            f"!= {merged_read.read_start} + {sum(len(this_motif) for this_motif in merged_read.motif_dec)}"
        )
    return result


def main_section_kmer_length(locus: MergedLocus) -> int:
    return primary_section_motif_length(locus, context="strand-bias annotation")

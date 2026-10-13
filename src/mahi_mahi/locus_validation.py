import itertools
import logging
from collections.abc import Sequence

from .errors import UserFacingError
from .io_utils.bed_utils import Locus, MergedLocus, select_primary_output_locus
from .utils import expand_motif_representation


def _format_motif_preview(motifs: set[str], max_motifs: int = 10) -> str:
    sorted_motifs = sorted(motifs)
    preview = ", ".join(sorted_motifs[:max_motifs])
    if len(sorted_motifs) > max_motifs:
        preview += f", ... ({len(sorted_motifs)} total)"
    return preview


def _warn_for_adjacent_composite_motif_overlap(repeat_id: str, loci: tuple[Locus, ...]) -> None:
    for left_locus, right_locus in itertools.pairwise(loci):
        shared_motifs = set(expand_motif_representation(left_locus.motif)) & set(expand_motif_representation(right_locus.motif))
        if shared_motifs:
            logging.warning(
                f"Composite TR {repeat_id} has overlapping expanded motifs between adjacent sections "
                f"{left_locus.variant_id} ({left_locus.motif}) and {right_locus.variant_id} ({right_locus.motif}): "
                f"{_format_motif_preview(shared_motifs)}. Reads containing shared motifs near this boundary may be assigned "
                "ambiguously during composite splitting."
            )


def primary_section_motif_length(locus: MergedLocus, *, context: str) -> int:
    if len(locus.loci) == 1:
        main_section_motifs = expand_motif_representation(locus.loci[0].motif)
        if not main_section_motifs:
            raise UserFacingError(f"Invalid loci file: Expected at least one motif in single-locus section {locus.loci[0].variant_id}")
        return len(main_section_motifs[0])

    main_locus = select_primary_output_locus(locus.loci, context=f"composite locus {locus.repeat_id} for {context}")
    main_section_motifs = expand_motif_representation(main_locus.motif)
    if not main_section_motifs:
        raise UserFacingError(f"Invalid loci file: Expected at least one motif in main section {main_locus.variant_id}")
    main_section_motif_lengths = {len(motif) for motif in main_section_motifs}
    if len(main_section_motif_lengths) != 1:
        raise UserFacingError(
            f"Invalid loci file: Composite TR primary output section {main_locus.variant_id} must expand to motifs of one length for {context}"
        )
    return len(main_section_motifs[0])


def _validate_locus(locus: MergedLocus) -> None:
    motifs = expand_motif_representation(locus.motif)
    if any(len(motif) == 1 for motif in motifs):
        # this is a trviz limitation
        raise UserFacingError(f"{locus.repeat_id}: Single-base motifs are not supported.")
    # these are align_all_star() limitations
    if len(locus.loci) > 1 and any(locus.loci[i].end != locus.loci[i + 1].start for i in range(len(locus.loci) - 1)):
        raise UserFacingError(
            f"The parts of composite TR {locus.repeat_id} do not exactly touch, which is not supported "
            "(it would count the intermediate bases as part of whichever TR they resemble more). "
            "Either add a _nonrepeat portion between them (for an example, see HTT), "
            "or if you intended to treat these as separate TRs, "
            "reduce region_merge_distance to less than the distance between them. "
            "Caution: short region_merge_distance can cause flank mis-mapping when the other TR is expanded, and hence incorrect results."
        )
    if len(locus.loci) > 1:
        _ = primary_section_motif_length(locus, context="strand-bias annotation")
        _warn_for_adjacent_composite_motif_overlap(locus.repeat_id, locus.loci)


def validate_merged_loci(merged_loci: Sequence[MergedLocus]) -> None:
    for locus in merged_loci:
        _validate_locus(locus)

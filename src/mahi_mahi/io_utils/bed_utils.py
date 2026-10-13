import collections
import logging
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..errors import UserFacingError

logger = logging.getLogger(__name__)
NON_PRIMARY_VARIANT_ID_RE = re.compile(r".*_[ACGTRYSWKMBDHVN]+[0-9]*(_[35]P)?|.*_nonrepeat")


class LocusKey(BaseModel):
    """
    A genomic region (chromosome, start position, and end position).

    Attributes:
        chrom (str): The chromosome name (e.g., 'chr1', 'chrX').
        start (int): The starting position of the locus (0-based).
        end (int): The ending position of the locus (exclusive).
    Config:
        frozen (bool): Indicates that the class is immutable and hashable.
    """

    chrom: str
    start: int
    end: int

    model_config = ConfigDict(frozen=True)


class Locus(LocusKey):
    """
    A single TR (Tandem Repeat) locus definition from the BED/catalog.

    Attributes:
        chrom (str): The chromosome name (e.g., 'chr1', 'chrX').
        start (int): The starting position of the locus (0-based).
        end (int): The ending position of the locus (exclusive).
        motif (str): The repeat motif sequence (e.g., "ACGT").
        repeat_id (str): The identifier for the repeat region.
        variant_id (str): The identifier for the specific variant of the repeat.
    Config:
        frozen (bool): Indicates that the class is immutable and hashable.
    """

    motif: str
    repeat_id: str
    variant_id: str

    model_config = ConfigDict(frozen=True)


class MergedLocus(LocusKey):
    """
    A set of adjacent TRs (Tandem Repeats) merged into a single locus.

    Attributes:
        chrom (str): The chromosome name (e.g., 'chr1', 'chrX').
        start (int): The starting position of the merged locus (0-based).
        end (int): The ending position of the merged locus (exclusive).
        loci: tuple(Locus, ...): The individual loci making up this merged locus, sorted by start position.
        motif (str): A comma-separated list of the locis' repeat motifs, in the same order as the loci.
    Config:
        frozen (bool): Indicates that the class is immutable and hashable.
    """

    # tuple not list for immutability
    loci: tuple[Locus, ...]
    motif: str
    repeat_id: str

    model_config = ConfigDict(frozen=True)


def is_primary_output_locus(locus: Locus) -> bool:
    """Return whether a locus should be treated as the primary output/main section."""
    return (locus.variant_id == locus.repeat_id) or not NON_PRIMARY_VARIANT_ID_RE.fullmatch(locus.variant_id)


def select_primary_output_locus(loci: Sequence[Locus], *, context: str) -> Locus:
    """Select the single primary output locus, or raise if naming is ambiguous."""
    primary_loci = [locus for locus in loci if is_primary_output_locus(locus)]
    if len(primary_loci) != 1:
        primary_variant_ids = [locus.variant_id for locus in primary_loci]
        raise UserFacingError(f"Invalid loci file: Expected exactly one primary output section in {context}. Found candidates: {primary_variant_ids}")
    return primary_loci[0]


def merge_all_loci(loci_list: list[Locus]) -> MergedLocus:
    """Create a MergedLocus from a list of loci."""
    if len({locus.chrom for locus in loci_list}) != 1:
        raise ValueError("All loci must be on the same chromosome to merge.")
    if len({locus.repeat_id for locus in loci_list}) != 1:
        raise UserFacingError(
            f"Invalid loci file: Loci {[locus.repeat_id for locus in loci_list]} are within region_merge_distance bases"
            " but have different repeat IDs.  If you intended these loci to be merged into a composite TR,"
            " give them all the same repeat ID; if you intended them to stay separate, reduce region_merge_distance"
            " to less than the distance between them."
            " Caution: short region_merge_distance can cause flank mis-mapping when the other TR is expanded, and hence incorrect results."
        )
    start = min(locus.start for locus in loci_list)
    end = max(locus.end for locus in loci_list)
    if len(loci_list) > 1:
        logger.debug(f"Merged {len(loci_list)} loci into {loci_list[0].chrom}:{start}-{end}")
    return MergedLocus(
        chrom=loci_list[0].chrom,
        start=start,
        end=end,
        loci=tuple(loci_list),
        motif=",".join(locus.motif for locus in loci_list),
        repeat_id=loci_list[0].repeat_id,
    )


def merge_nearby_loci(loci_list: list[Locus], merge_distance: int) -> list[MergedLocus]:
    """
    Merge loci if gap between them is <= merge_distance.
    """
    # Sort regions by chromosome and start position before merging
    loci_list = sorted(loci_list, key=lambda x: (x.chrom, x.start, x.end))
    merged_loci_list: list[MergedLocus] = []
    current_group: list[Locus] = []
    current_group_end: int = -1
    for locus in loci_list:
        chrom, start, end = locus.chrom, locus.start, locus.end
        if current_group and chrom == current_group[-1].chrom and start - current_group_end <= merge_distance:
            # merge when within merge_distance - note that this can include overlaps (start < current_group_end)
            current_group_end = max(current_group_end, end)
            current_group.append(locus)
        else:
            # not close - start a new group
            if current_group:
                merged_loci_list.append(merge_all_loci(current_group))
            current_group = [locus]
            current_group_end = end
    # handle the last group
    if current_group:
        merged_loci_list.append(merge_all_loci(current_group))
    repeat_ids = [locus.repeat_id for locus in merged_loci_list]
    if len(repeat_ids) != len(set(repeat_ids)):
        duplicates = [k for k, v in collections.Counter(repeat_ids).items() if v > 1]
        raise UserFacingError(
            f"Invalid loci file: Duplicate repeat IDs found (that are not within the same composite TR locus): {duplicates}, which are not allowed"
        )
    return merged_loci_list


def load_loci_list(loci_bed: Path, repeat_id: str | None = None) -> list[Locus]:
    """
    Create a list of loci from the loci bed file.
    Args:
        loci_bed (Path): Path to the loci bed file.
        repeat_id (str | None): If provided, only load loci with this repeat ID (case-insensitive).
    Returns:
        List[Locus]: List of loci.
    """
    # Read all loci entries
    loci_list: list[Locus] = []
    with loci_bed.open() as ff:
        for line in ff:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split("\t")
            if len(cols) < 6:
                logger.warning(f"Skipping malformed line: {line.strip()}")
                continue
            chrom, start, end, motif, this_repeat_id, variant_id, *_ = cols
            if repeat_id is not None and this_repeat_id.upper() != repeat_id.upper():
                continue
            loci_list.append(
                Locus(chrom=chrom, start=int(start), end=int(end), motif=motif, repeat_id=this_repeat_id, variant_id=variant_id),
            )
    if repeat_id and len(loci_list) == 0:
        raise UserFacingError(f"Requested repeat ID '{repeat_id}' not found in {loci_bed}.")
    variant_ids = [locus.variant_id for locus in loci_list]
    if len(variant_ids) != len(set(variant_ids)):
        duplicates = [k for k, v in collections.Counter(variant_ids).items() if v > 1]
        raise UserFacingError(f"Invalid loci file: Duplicate variant IDs found: {duplicates}, which are not allowed")
    return loci_list

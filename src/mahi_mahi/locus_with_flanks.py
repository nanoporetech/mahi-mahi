from __future__ import annotations

import logging

import pysam
from pydantic import BaseModel, ConfigDict

from .io_utils.bed_utils import MergedLocus
from .utils import expand_motif_representation, normalize_cyclic_shifts, process_subread


class LocusWithFlanks(BaseModel):
    """Locus with flanking sequences and non-ambiguous motif list.

    Allows these to be loaded or computed once, rather than for every read."""

    locus: MergedLocus
    motifs: list[str]
    flank_seq_start: str
    flank_seq_end: str

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_locus(
        cls: type[LocusWithFlanks],
        locus: MergedLocus,
        extension: int,
        fasta_file: pysam.FastaFile,
        max_flank_length: int,
    ) -> LocusWithFlanks:
        """Create LocusWithFlanks from MergedLocus, flank length, and reference genome.

        Args:
            locus (MergedLocus): the locus.
            extension (int): length (in bases) of flanking sequences to fetch.
            fasta_file (pysam.FastaFile): reference genome for fetching sequences.
            max_flank_length (int): maximum length of flanking sequences to fetch for fallback realignment.
        Returns:
            LocusWithFlanks object with flanking sequences."""
        motifs = expand_motif_representation(locus.motif)
        flank_length = max(extension, max_flank_length)
        start_pos_with_flank = max(0, locus.start - flank_length)
        end_pos_with_flank = min(fasta_file.lengths[fasta_file.references.index(locus.chrom)], locus.end + flank_length)
        flank_seq_start = fasta_file.fetch(locus.chrom, start_pos_with_flank, locus.start)
        flank_seq_end = fasta_file.fetch(locus.chrom, locus.end, end_pos_with_flank)
        reference_allele = fasta_file.fetch(locus.chrom, locus.start, locus.end).upper()
        if len(reference_allele) > 0:
            reference_decomposition = process_subread(reference_allele, motifs)
            # Note that while this allows cyclic-shifted motifs, we do not recommend them, as they are likely to cause an off-by-1 count
            # (because the partial motifs at both ends will count)
            # We allow them partly because of the possibility that trviz will use them in imperfect TRs
            normalized_motifs = set(normalize_cyclic_shifts(motifs))
            normalized_decomposition = normalize_cyclic_shifts(reference_decomposition, set(motifs))
            reference_matching_fraction = sum(
                len(motif)
                for motif, normalized_motif in zip(reference_decomposition, normalized_decomposition, strict=True)
                if normalized_motif in normalized_motifs
            ) / len(reference_allele)
            if reference_matching_fraction < 0.5:
                logging.warning(
                    f"Locus {locus.repeat_id} motif(s) {locus.motif} match less than half of reference sequence {reference_allele}.\n\n"
                    "If this was intended to exclude a non-pathogenic reference motif from the total count, it will not work: "
                    "mahi-mahi's total count is everything between the given coordinates not just everything that "
                    "matches one of the given motifs.  Consider using a 3-part composite TR instead (see README.md).\n\n"
                    "If you expected a reference allele that did mostly match, check that the coordinates "
                    f"{locus.chrom}:{locus.start}-{locus.end} are for the version "
                    "of the reference genome you are using, and that the motif is given in the reference orientation.\n\n"
                    "If this locus is a very imperfect TR (i.e. most of its repeats are non-exact repeats), "
                    "be aware that mahi-mahi is not designed to handle these and may give incorrect results."
                )
        return cls(
            locus=locus,
            motifs=motifs,
            flank_seq_start=flank_seq_start,
            flank_seq_end=flank_seq_end,
        )

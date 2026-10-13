import itertools
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pysam
from trviz.decomposer import Decomposer

from .errors import UserFacingError
from .io_utils.bed_utils import MergedLocus
from .pipeline_records import ReadCall

iupac_codes: dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}


def iupac_generate_all(seq: str) -> list[str]:
    """Generates all ACGT-strings from the given IUPAC template"""
    # Create a list of lists where each inner list contains possible characters for the given position
    try:
        possible_chars = [iupac_codes[char] for char in seq]
    except KeyError as exc:
        raise UserFacingError(f"Invalid loci file: Unsupported IUPAC character {exc.args[0]!r} in motif {seq!r}") from exc

    # Generate the Cartesian product of all possible characters
    all_combinations = itertools.product(*possible_chars)

    # Join the tuples into strings and convert to a list
    return ["".join(combination) for combination in all_combinations]


def get_chr_id(x: str) -> int:
    x = x.split("_")[0]
    for chr_x_name in get_chrX_names():
        if x.startswith(chr_x_name):
            return 23
    for chr_y_name in get_chrY_names():
        if x.startswith(chr_y_name):
            return 24
    if x.startswith("chr"):
        return int(x[len("chr") :])
    else:
        return int(x)


def get_chrX_names() -> set[str]:
    return {"chrX", "X"}


def get_chrY_names() -> set[str]:
    return {"chrY", "Y"}


def is_sex_chr(chr_name: str) -> bool:
    """
    Check if the given chromosome name corresponds to a sex chromosome (X or Y).

    Args:
        chr_name (str): Chromosome name (e.g., 'chr1', 'X', 'chrY', 'Y').

    Returns:
        bool: True if the chromosome is a sex chromosome, False otherwise.
    """
    chrXY_names = get_chrX_names() | get_chrY_names()
    return any(chr_name.startswith(name) for name in chrXY_names)


def expand_motif_representation(motifs_str: str) -> list[str]:
    """For each IUPAC motif representation generate set of ACGT-strings"""
    motifs = motifs_str.split(",")
    res_motifs = []
    for m in motifs:
        res_motifs.extend(iupac_generate_all(m))
    return res_motifs


def sum_dictionaries_in_column(rows: Iterable[ReadCall | Mapping[str, object]], column_name: str) -> Counter[str]:
    """
    Aggregate dictionary-like entries from the specified record field.

    Args:
        rows: Records or mapping-like rows containing a dictionary-valued field.
        column_name: Name of the field to aggregate.

    Returns:
        Counter[str]: Summed counts for each motif across all rows.
    """
    aggregated_counts: Counter[str] = Counter()

    for row in rows:
        motif_counts = row.get(column_name) if isinstance(row, Mapping) else getattr(row, column_name, None)

        if not isinstance(motif_counts, dict):
            continue

        for motif, count in motif_counts.items():
            aggregated_counts[str(motif)] += int(count)

    return aggregated_counts


def process_subread(read: str, motifs: list[str]) -> list[str]:
    """Convert TR sequence alignment into a list of motifs"""
    if not read:
        return []
    tr_decomposer = Decomposer()
    dec = tr_decomposer.decompose(read, sorted(set(motifs)))
    assert all(motif for motif in dec), f"Decomposition contained empty motif token: {dec}"
    return dec


def is_cram(file_path: str) -> bool:
    with pysam.AlignmentFile(file_path, "r") as samfile:
        return samfile.is_cram


def is_bam(file_path: str) -> bool:
    with pysam.AlignmentFile(file_path, "r") as samfile:
        return samfile.is_bam


def convert_cram_to_bam(cram_file: str, output_dir: str, reference_fasta: str | None = None, bed_file: Path | None = None) -> str:
    """
    Convert a CRAM file to a BAM file using pysam, optionally restricting to
    regions in a BED file.

    Parameters:
    cram_file (str): Path to the input CRAM file.
    output_dir (str): Path to the output directory for the temporary BAM and BAI files.
    reference_fasta (str, optional): Path to the reference genome FASTA file
                                     (required for CRAM input if needed).
    bed_file (str, optional): Path to a BED file specifying regions to include
                              in the conversion.

    Returns:
    bam_file(str): Path to the output BAM file.
    """
    bam_file = str(Path(output_dir) / "file.bam")
    # Open output BAM file

    convert_parameters = ["-T", reference_fasta] if reference_fasta else []
    if bed_file:
        convert_parameters.extend(["--regions-file", str(bed_file)])
    convert_parameters.extend(["-b", "-o", bam_file, cram_file])
    pysam.view(*convert_parameters, catch_stdout=False)
    # Create BAM index
    pysam.index(bam_file)

    logging.info(f"Conversion complete: {cram_file} -> {bam_file}")
    return bam_file


def get_mahi_mahi_version() -> str:
    try:
        return version("mahi_mahi")
    except PackageNotFoundError:
        return "unknown"


def get_ps_hap(read_aln_seg: pysam.AlignedSegment, chr_name: str, sex: str) -> tuple[str, str]:
    """
    Determine phase set (PS) and haplotype (HAP) tags for an aligned read segment based on PS/HP tags,
    taking into account sex and chromosome type.

    Parameters:
        read_aln_seg (pysam.AlignedSegment): The aligned read segment with optional PS/HP tags.
        chr_name (str): Chromosome name (e.g., 'chr1', 'chrY').
        sex (str): Sample sex, either 'XY' or 'XX'.
    Returns:
        Tuple[str, str]: A tuple (ps_tag, hap_tag):
            - If sex is 'XY' and chr_name is a sex chromosome, returns ('ps0', '1').
            - Otherwise, extracts PS and HP tags from read_aln_seg (prefixed with 'ps'),
              defaulting to 'ps0' and '0' if both tags are absent. Reads must contain
              both PS and HP tags, or neither.
    """
    has_ps = read_aln_seg.has_tag("PS")
    has_hp = read_aln_seg.has_tag("HP")
    if has_ps != has_hp:
        raise UserFacingError(
            f"Read {read_aln_seg.query_name!r} has mismatched phasing tags: PS is {'present' if has_ps else 'absent'} and "
            f"HP is {'present' if has_hp else 'absent'}. Each read must contain both PS and HP tags, or neither."
        )
    if sex == "XY" and is_sex_chr(chr_name):
        return "ps0", "1"
    ps_tag = "ps" + str(read_aln_seg.get_tag("PS")) if has_ps else "ps0"
    hap_tag = str(read_aln_seg.get_tag("HP")) if has_hp else "0"
    return ps_tag, hap_tag


def ensure_folder_exists(folder_path: str) -> None:
    """
    Ensures that the specified folder exists. If it does not exist, it creates the folder.

    Args:
        folder_path (str): Path to the folder to check or create.
    """
    path = Path(folder_path)
    if not path.exists():
        path.mkdir(parents=True)


def detect_unsplit_2d_reads(
    bamfile: str,
    loci: Iterable[MergedLocus],
    extension: int,
) -> set[str]:
    """
    Identify unsplit 2D reads (reads with both primary and supplementary alignments) aligned to given loci in a BAM file.

    This function expands each locus by `extension` bases on both sides and fetches aligned segments from the BAM file.
    Reads that have both a primary and supplementary alignment within the expanded region are considered unsplit 2D reads.
    This function match behaviour of `process_aligned_segment` in main.py to get same set of alignments.

    Parameters:
        bamfile (str): Path to the input BAM file.
        loci (Iterable[MergedLocus]): Iterable of MergedLocus objects defining regions to search.
        extension (int): Number of bases to expand each locus on both sides.

    Returns:
        set[str]: Set of read names identified as unsplit 2D reads.
    """
    unsplit_2d: set[str] = set()

    with pysam.AlignmentFile(bamfile, "rb") as bam:
        for locus_key in loci:
            primary_seen: set[str] = set()
            supp_seen: set[str] = set()
            if locus_key.chrom not in bam.references:
                raise UserFacingError(
                    f"Locus chromosome {locus_key.chrom!r} is not present in --bam/--cram. "
                    "Check that --bam/--cram and --loci use the same chromosome names."
                )
            chrom_len = bam.get_reference_length(locus_key.chrom)
            if locus_key.start < 0 or locus_key.end > chrom_len:
                raise UserFacingError(
                    f"Locus {locus_key.chrom}:{locus_key.start}-{locus_key.end} is off the end of the chromosome "
                    f"(length {chrom_len}).  This can be caused by a typo, "
                    "or by --loci and --bam/--cram using different reference genome versions."
                )
            start = max(0, locus_key.start - extension)
            end = min(locus_key.end + extension, chrom_len)
            for aligned_segment in bam.fetch(locus_key.chrom, start, end):
                # matched primary or supplementary alignment for unsplit 2D reads
                if aligned_segment.query_name:
                    if aligned_segment.is_supplementary:
                        supp_seen.add(aligned_segment.query_name)
                    else:
                        primary_seen.add(aligned_segment.query_name)

            # add reads seen as both primary and supplementary
            unsplit_2d.update(primary_seen & supp_seen)

    return unsplit_2d


def check_bam_reference_compatibility(
    alignment_file: str,
    reference_fasta: str,
    chroms: Iterable[str],
) -> None:
    """Check that chromosome lengths in the alignment file header match those in the reference FASTA.

    Only checks chromosomes that appear in `chroms`.  Mismatches suggest the alignment file was
    produced with a different version of the reference genome.

    Parameters:
        alignment_file (str): Path to the BAM or CRAM file.
        reference_fasta (str): Path to the reference FASTA file.
        chroms (Iterable[str]): Chromosome names to check.

    Raises:
        ValueError: If any chromosome length differs between the alignment file and the reference.
    """
    with pysam.AlignmentFile(alignment_file, "r") as bam:
        bam_lengths = dict(zip(bam.references, bam.lengths, strict=True))
    with pysam.FastaFile(reference_fasta) as fasta:
        fasta_lengths = dict(zip(fasta.references, fasta.lengths, strict=True))
    mismatches = [
        f"  {chrom}: alignment file has length {bam_lengths[chrom]}, reference has {fasta_lengths[chrom]}"
        for chrom in set(chroms)
        if chrom in bam_lengths and chrom in fasta_lengths and bam_lengths[chrom] != fasta_lengths[chrom]
    ]
    if mismatches:
        raise UserFacingError(
            "Chromosome length mismatch between alignment file and reference genome, "
            "which probably means this BAM/CRAM is aligned to a different version of the reference.\n\n"
            "Halting execution as mahi-mahi requires that --bam/--cram, --ref and --loci all use the same "
            "coordinate system.  You may need to re-align the BAM/CRAM file to the correct reference genome.\n\n" + "\n".join(mismatches)
        )
    for chroms_present, message in [
        (bam_lengths, "reference the --bam/--cram was aligned to"),
        (fasta_lengths, "reference given as --ref"),
    ]:
        missing_chroms = [chrom for chrom in chroms if chrom not in chroms_present]
        if missing_chroms:
            raise UserFacingError(
                f"The following chromosomes have one or more loci in --loci but are missing from the {message}:\n"
                + "\n".join(f" {chrom}" for chrom in missing_chroms)
                + "\n\nHalting execution as mahi-mahi requires that --bam/--cram, --ref and --loci all use the same "
                "chromosome names.  If only a single chromosome is affected, this may be a typo in the BED file."
            )


def get_all_cyclic_shifts(motif: str) -> list[str]:
    """
    Generate all cyclic shifts of a motif.

    Parameters:
    - motif (str): A string motif.

    Returns:
    - List[str]: All cyclic shifts of the motif.
    """
    return [motif[i:] + motif[:i] for i in range(len(motif))]


def normalize_cyclic_shifts(motifs: Sequence[str] | Iterable[str], motifs_set: set[str] | None = None) -> list[str]:
    """
    Normalize motifs by collapsing cyclic shifts to their canonical forms.

    Parameters:
    - motifs (List[str]): List of motifs to normalize.
    - motifs_set (Set[str], optional): Set of motifs whose cyclic rotations should be treated as equivalent.
      If None, all motifs are normalized.

    Returns:
    - List[str]: Normalized motifs.
    """
    motif_list = list(motifs)
    if motifs_set is None:
        motifs_set = set(motif_list)
    cyclic_shifts = [get_all_cyclic_shifts(m) for m in motifs_set]
    replace_dict = {}
    for cyclic_shift in cyclic_shifts:
        # Use the lexicographically smallest cyclic shift as the key
        canonical_form = min(cyclic_shift)
        replace_dict.update(dict.fromkeys(cyclic_shift, canonical_form))
    # Normalize motifs selectively
    return [replace_dict.get(m, m) for m in motif_list]


def is_repeated_sequence(motifs: Sequence[str], motifs_set: set[str], threshold: float = 4.0) -> bool:
    """
    Flag a motif sequence as **repetitive / low-complexity** when its Shannon
    entropy falls below *threshold*.

    Steps
    -----
    1. **Normalize cyclic shifts**
       Each motif in *motifs_set* is collapsed—with all of its rotations—to
       its lexicographically smallest form.  Only motifs encountered in
       `motifs` that match these canonical forms (or any of their rotations)
       are normalised; all others stay unchanged so we don't artificially
       lower the entropy of genuinely diverse sequences.

    2. **Compute entropy** on the transformed motif list and compare it with
       *threshold*.  Return **True** if the entropy is lower (i.e. the
       sequence is dominated by repeats), otherwise **False**.

    Parameters
    ----------
    motifs : List[str]
        Ordered list of motifs (e.g. k-mers) from the sequence.
    motifs_set : Set[str]
        Motifs whose cyclic rotations should be treated as equivalent.
    threshold : float, optional
        Entropy cut-off for classifying a sequence as repetitive
        (default = 4.0 bits).

    Returns
    -------
    bool
        **True** if the sequence is low-complexity, **False** otherwise.
    """
    if not motifs:
        return True  # Empty sequence is trivially repetitive
    # Normalize motifs using the canonical forms from motifs_set
    transformed = normalize_cyclic_shifts(motifs, motifs_set)
    # Shannon entropy
    counts = Counter(transformed)
    total = len(motifs)
    entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())
    return entropy < threshold


def count_all_kmers(sequence: str, k: int, min_count: int = 0) -> dict[str, int]:
    """
    Count all k-mers in a sequence with cyclic shift normalization and optional minimum count filtering.

    This function extracts all k-mers of length `k` from the input sequence,
    normalizes cyclic shifts to their canonical forms, and returns counts
    that meet the minimum count threshold.

    Parameters:
    - sequence (str): Input DNA/RNA sequence to analyze.
    - k (int): Length of k-mers to extract (must be positive).
    - min_count (int): Minimum count threshold for k-mers to be included (default: 0).

    Returns:
    - Dict[str, int]: Dictionary mapping normalized k-mers to their counts,
      filtered by minimum count threshold.

    Example:
        >>> count_all_kmers("ATGCATGC", k=3, min_count=2)
        {'ATG': 2, 'GCT': 2}
    """
    if not sequence or k <= 0 or len(sequence) < k or min_count < 0:
        return {}

    # Extract all k-mers from the sequence
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i : i + k])

    # Normalize cyclic shifts to canonical forms
    normalized_kmers = normalize_cyclic_shifts(kmers)
    # Count occurrences of normalized k-mers
    kmer_counts = Counter(normalized_kmers)
    # Filter by minimum count threshold
    return {kmer: count for kmer, count in kmer_counts.items() if count >= min_count}


def filter_top_kmers_by_percentage(kmer_counts: dict[str, int], percentage: float = 90.0) -> dict[str, int]:
    """
    Filter k-mer counts to keep only the most frequent k-mers that collectively explain
    a specified percentage of all k-mers.

    This function sorts k-mers by their counts in descending order and keeps only those
    k-mers that together account for the specified percentage of the total count.

    Args:
        kmer_counts: Dictionary mapping k-mers to their counts.
        percentage: Percentage threshold (0-100) for cumulative k-mer coverage.

    Returns:
        dict[str, int]: Dictionary containing only the top k-mers that explain the specified
        percentage of total k-mer counts, maintaining original counts.

    Example:
        >>> counts = {"ATG": 100, "CGA": 50, "TTT": 30, "AAA": 15, "GGG": 5}
        >>> filter_top_kmers_by_percentage(counts, 90.0)
        {'ATG': 100, 'CGA': 50, 'TTT': 30, 'AAA': 15}
        >>> filter_top_kmers_by_percentage(counts, 75.0)
        {'ATG': 100, 'CGA': 50}
    """
    if not kmer_counts:
        return {}
    if percentage <= 0:
        return {}
    if percentage >= 100:
        return dict(kmer_counts)

    # Calculate total count
    total = sum(kmer_counts.values())
    threshold = total * (percentage / 100.0)
    running_total = 0
    result: dict[str, int] = {}
    # Sort k-mers by count in descending order
    for kmer, count in sorted(kmer_counts.items(), key=lambda item: item[1], reverse=True):
        result[kmer] = count
        running_total += count
        # Stop if we've reached or exceeded the target percentage
        if running_total >= threshold:
            break
    return result


def motif_rotation_representatives(motif_totals: Mapping[str, int], expected_motifs: Sequence[str]) -> dict[str, str]:
    """
    Choose the form in which to report each observed motif, collapsing cyclic rotations.

    Observed motifs (the keys of *motif_totals*) are grouped into rotation classes. Each class is reported as:
    - the expected motif, if exactly one of its rotations is expected;
    - itself (i.e. not collapsed), if several of its rotations are expected, as it is then ambiguous which to use;
    - otherwise, its most common rotation (by total count), with ties broken lexicographically.

    Parameters:
    - motif_totals (Mapping[str, int]): Total count of each observed motif, e.g. summed over an allele's reads.
    - expected_motifs (Sequence[str]): Expected motifs of the locus.

    Returns:
    - Dict[str, str]: Mapping from each observed motif to the motif it should be reported as.
    """
    classes: dict[str, list[str]] = {}
    for motif in motif_totals:
        classes.setdefault(min(get_all_cyclic_shifts(motif)), []).append(motif)
    expected_by_class: dict[str, list[str]] = {}
    for motif in dict.fromkeys(expected_motifs):
        expected_by_class.setdefault(min(get_all_cyclic_shifts(motif)), []).append(motif)
    representatives: dict[str, str] = {}
    for class_key, rotations in classes.items():
        class_expected = expected_by_class.get(class_key, [])
        if len(class_expected) > 1:
            logging.warning(f"Expected motifs {class_expected} are rotations of each other, so their rotations will not be collapsed in motif counts")
            representatives.update({motif: motif for motif in rotations})
            continue
        representative = class_expected[0] if class_expected else min(rotations, key=lambda motif: (-motif_totals[motif], motif))
        representatives.update(dict.fromkeys(rotations, representative))
    return representatives


def sanitize_plot_filename(value: str) -> str:
    """Make a value safe for use in a plot filename."""
    return re.sub("[^a-zA-Z0-9@._-]+", "_", value).replace("..", "_").strip("_")

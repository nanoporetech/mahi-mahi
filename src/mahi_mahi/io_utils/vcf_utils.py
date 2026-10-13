import logging
import string
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import pysam
from pydantic import BaseModel

from mahi_mahi.clustering import is_sex_chr

from ..pipeline_records import AlleleSummary, ReadCall, RegionCallSet
from ..utils import expand_motif_representation, get_chr_id, get_mahi_mahi_version, process_subread
from .bed_utils import Locus, is_primary_output_locus

# sorts after every chromosome get_chr_id can parse
_UNPLACED_CONTIG_ID = 1 << 30


class RefAllele(BaseModel):
    """
    Represents a reference allele and its repeat decomposition.

    Attributes:
        padded_base (str): The base sequence of the reference allele, including 1 base of padding before the repeat region.
        decomposition (List[str]): List of repeat units decomposed from the reference allele (without padding).
    """

    padded_base: str
    decomposition: list[str]


class AlleleInformation(BaseModel):
    """
    Represents the information of an allele in a VCF file.

    Attributes:
        REPCN (int): Number of repeat units spanned by the allele (non-negative integer).
        REPCNQ (str): Quartiles for the repeat count, formatted as q25-q75.
        AD (int): Number of spanning reads consistent with the allele (non-negative integer).
        AD_NSPAN (int): Number of non-spanning reads consistent with the allele (non-negative integer).
        STDDEV (float): Standard deviation of alignment lengths consistent with the allele.
        MRCN (str): Median counts of motifs formatted as motif1(count1);motif2(count2).
        MMETH (str): Fraction of the allele's called CpG sites that are methylated (5mC or 5hmC), pooled over the reads supporting the allele.
        MMETHN (str): Number of callable CpG observations pooled into MMETH.
        BPCN (int): Allele size in base pairs (non-negative integer).
    """

    REPCN: int
    AD: int
    AD_NSPAN: int
    STDDEV: float
    MRCN: str
    MMETH: str
    MMETHN: str = "."
    REPCNQ: str
    BPCN: int


class SymbolicAllele(BaseModel):
    """
    Represents a symbolic allele in a VCF file.

    Attributes:
        ID (str): Identifier of the allele.
        Description (str): Description of the allele.
        Count (int): Number of repeat units in the allele.
    """

    id: str
    description: str
    count: int


def contig_sort_key(chrom: str) -> tuple[int, str]:
    """
    Sort key putting chromosomes in numeric order, with contigs that are not recognisable chromosomes last.

    The `##contig` header lines and the data lines must agree on this order, or htsjdk-based tools
    (GATK, IGV, Picard) reject the file as not sorted.

    Args:
        chrom (str): Contig name.

    Returns:
        Tuple[int, str]: Sort key for the contig.
    """
    try:
        return (get_chr_id(chrom), chrom)
    except ValueError:
        return (_UNPLACED_CONTIG_ID, chrom)


def escape_vcf_header_value(value: str) -> str:
    """Escape control characters that cannot occur literally in a VCF header value."""
    return value.replace("\r", "%0D").replace("\n", "%0A").replace("\t", "%09")


def form_vcf_header(reference_genome_fasta: str, command: str | None = None) -> str:
    mahi_mahi_version = get_mahi_mahi_version()
    command_header = f"##mahi_mahiCommand={escape_vcf_header_value(command)}\n" if command is not None else ""
    output_vcf_header_lst = [
        f"##fileformat=VCFv4.3\n"
        f"##source=mahi_mahi-v{mahi_mahi_version}\n"
        f"##reference={reference_genome_fasta}\n"
        f"{command_header}"
        f'##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n'
        f'##INFO=<ID=REF,Number=1,Type=Integer,Description="Reference copy number">\n'
        f'##INFO=<ID=REPID,Number=1,Type=String,Description="Variant identifier as specified in the variant catalog">\n'
        f'##INFO=<ID=RL,Number=1,Type=Integer,Description="Reference length in bp">\n'
        f'##INFO=<ID=RU,Number=.,Type=String,Description="Repeat unit in the reference orientation">\n'
        f'##FILTER=<ID=LowDepth,Description="The overall locus depth or number of reads spanning the region '
        f'is below the required thresholds, or one of the alleles does not have enough read support.">\n'
        f'##FILTER=<ID=MotifBias,Description="Region affected by basecaller errors detected through strand-specific motif analysis.">\n'
        f'##FILTER=<ID=PASS,Description="All filters passed">\n'
        f'##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">\n'
        f'##FORMAT=<ID=AD_NSPAN,Number=R,Type=Integer,Description="Allelic depths from non-spanning reads only, for ref and alt alleles">\n'
        f'##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f'##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Locus coverage">\n'
        f'##FORMAT=<ID=REPCN,Number=A,Type=Integer,Description="Number of repeat units spanned by the allele">\n'
        f'##FORMAT=<ID=REPCNQ,Number=A,Type=String,Description="Repeat copy number quantiles (Q25, Q75) are reported as a range (e.g., 5-10).">\n'
        f'##FORMAT=<ID=CLSTATE,Number=1,Type=String,Description="Clustering outcome: '
        f'SUCCESS, HAPLOTAG_FALLBACK, AMBIGUOUS, or ASSUMED_HAPLOID.">\n'
        f'##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier for phased genotypes">\n'
        f'##FORMAT=<ID=MRCN,Number=A,Type=String,Description="Median counts of expanded motifs, e.g., CAG(12);CGG(55)">\n'
        f"##FORMAT=<ID=MMETH,Number=A,Type=Float,Description=\"Fraction of the allele's called CpG sites that are methylated (5mC or 5hmC), "
        f'pooled over the reads supporting the allele">\n'
        f'##FORMAT=<ID=MMETHN,Number=A,Type=Integer,Description="Number of callable CpG observations pooled into MMETH">\n'
        f'##FORMAT=<ID=BPCN,Number=A,Type=Integer,Description="Allele size in bp">'
    ]

    with pysam.FastaFile(reference_genome_fasta) as ref:
        for contig, length in sorted(zip(ref.references, ref.lengths, strict=True), key=lambda item: contig_sort_key(item[0])):
            output_vcf_header_lst.append(f"##contig=<ID={contig},length={length}>")

    return "\n".join(output_vcf_header_lst) + "\n"


def transform_rows_to_vcf_formats_and_alts(region: RegionCallSet, haploid: bool = False) -> tuple[dict[str, str], list[SymbolicAllele]]:
    """
    Convert grouped typed read records for one region into VCF-compatible FORMAT fields and ALT alleles.

    Args:
        region (RegionCallSet): Grouped results for a specific region.
        haploid (bool): If True, use only allele 1 to generate the format.

    Returns:
        Tuple[Dict[str, str], List[SymbolicAllele]]: A tuple containing the VCF format dictionary and a list of alternative alleles.
    """
    if not region.reads or not region.alleles or not any(record.analysis_exclusion_reason is None for record in region.reads):
        return {}, []

    gt = decode_gt(region.genotype)
    non0_alleles = [this_gt for this_gt in gt if this_gt != "0"]
    # we treat phased genotypes as heterozygous since we don`t compare the sequences of the allele and only the lengths
    homozygous = len(non0_alleles) <= 1 and not region.phased
    alleles = sorted(non0_alleles)

    if haploid:
        alleles = ["1"]
        homozygous = True
    elif homozygous:
        alleles = [non0_alleles[0]] if len(non0_alleles) == 1 else ["0"]

    alleles_information = [extract_allele_information(region, allele, gt[allele]) for allele in alleles]
    vcf_fmt = generate_format(alleles_information)
    alts = get_alts(alleles_information)

    # adding zero for ref allele depth as we never match to the ref allele
    vcf_fmt["AD"] = "0," + vcf_fmt["AD"]
    vcf_fmt["AD_NSPAN"] = "0," + vcf_fmt["AD_NSPAN"]
    vcf_fmt["CLSTATE"] = region.clustering_state.value
    vcf_fmt["PS"] = "."
    if not alts:
        vcf_fmt["GT"] = "." if haploid else "./."
    else:
        # Use "/" for unphased genotypes and "|" for phased genotypes.
        gt_sep = "/"
        if region.phased:
            gt_sep = "|"
            vcf_fmt["PS"] = extract_unique_phaseset(region.reads)
        if haploid:
            vcf_fmt["GT"] = "1"
        elif homozygous:
            vcf_fmt["GT"] = f"1{gt_sep}1"
        elif len(alts) == 1:
            vcf_fmt["GT"] = f"1{gt_sep}." if alleles[0] == "1" else f".{gt_sep}1"
        else:
            vcf_fmt["GT"] = f"1{gt_sep}2"

    return vcf_fmt, alts


def generate_format(alleles_format: list[AlleleInformation]) -> dict[str, str]:
    """
    Generate the format from alleles.
    Args:
        alleles_format (List[AlleleInformation]): List of AlleleInformation instances containing allele formats.

    Returns:
        Dict[str, str]: Dictionary containing the allele format.
    """
    vcf_fmt = {}
    for key in alleles_format[0].model_dump():
        values = [getattr(allele, key) for allele in alleles_format]
        vcf_fmt[key] = ",".join(map(str, values))
    return vcf_fmt


def generate_format_empty(haploid: bool = False) -> dict[str, str]:
    """
    Generate an empty format dictionary.

    Args:
        haploid (bool): If True, generate format for haploid case.

    Returns:
        Dict[str, str]: Dictionary containing empty format values.
    """
    return {
        "GT": "." if haploid else "./.",
        "AD": ".",
        "REPCN": ".",
        "REPCNQ": ".",
        "STDDEV": ".",
        "CLSTATE": ".",
        "PS": ".",
        "MRCN": ".",
        "MMETH": ".",
        "MMETHN": ".",
        "AD_NSPAN": ".",
        "BPCN": ".",
    }


def warn_on_unexpected_motifs(region: RegionCallSet, summary: AlleleSummary) -> bool:
    """
    Log a warning and return True if this allele has a large fraction of unexpected motifs

    (over 50% of the total, or the most common single motif, not in the expected motif set)

    Args:
        region (RegionCallSet): Region metadata for the allele being checked.
        summary (AlleleSummary): Allele motif/count summary for the allele being checked.
    """
    motifs_median_counts = summary.motifs_median_counts
    if motifs_median_counts is None:
        return False
    expected_motifs = expand_motif_representation(region.repeat_unit)
    expected_motifs_total_count = sum(motifs_median_counts.get(motif, 0) for motif in expected_motifs)
    most_common_motif, _ = max(motifs_median_counts.items(), key=lambda item: item[1], default=("", 0))
    # It is intentional that we check allele_q50 (median total count)
    # even when the reported total is based on a higher quantile, to be a fair comparison against motif_median_counts
    # the summary.q50>0 is to not warn for 0-length alleles (where motifs_median_counts is expected to be empty,
    # and hence most_common_motif is expected to be "")
    if ((most_common_motif not in expected_motifs) and summary.q50 > 0) or expected_motifs_total_count < 0.5 * summary.q50:
        logging.warning(
            f"Unexpected sequence at {region.repeat_id}: expected motifs {expected_motifs}, "
            f"actual counts {motifs_median_counts} out of total copy number {summary.q50}.  "
            f"This may be a real expansion with a non-standard motif, or an analysis error."
        )
        return True
    return False


def extract_allele_information(region: RegionCallSet, allele: str, allele_stat: tuple[float, int]) -> AlleleInformation:
    """
    Create an AlleleInformation object from typed read records and allele metadata.
    Args:
        region (RegionCallSet): Grouped results containing the allele summary for the region.
        allele (str): Allele identifier.
        allele_stat (Tuple[float, int]): Tuple containing the allele repeat count and support.
    Returns:
        AlleleInformation: An instance of AlleleInformation containing the allele format.
    """

    allele_repeat_count = round(allele_stat[0])
    summary = region.alleles.get(allele)
    if summary is None:
        raise KeyError(allele)

    if not isinstance(summary.motifs_median_counts, dict):
        logging.warning(
            f"No motif information available for allele {allele} of region {region.region_name}."
            " This may be due to all reads being marked as suspicious."
        )
        motifs_median_counts = "."
    else:
        _ = warn_on_unexpected_motifs(region, summary)
        # Format motifs_median_counts as key1(count1);key2(count2), or "." if there are none (e.g. a zero-length allele)
        motifs_median_counts = ";".join([f"{key}({value})" for key, value in summary.motifs_median_counts.items()]) or "."

    allele_support = allele_stat[1]
    allele_std_value = summary.stddev
    allele_std = float(f"{allele_std_value:.2f}") if allele_support > 1 else 0.0
    allele_quantiles = f"{summary.q25}-{summary.q75}"
    allele_support_nspan = summary.num_nonspanning_reads_used
    return AlleleInformation(
        REPCN=allele_repeat_count,
        AD=allele_support,
        STDDEV=allele_std,
        MRCN=motifs_median_counts,
        MMETH="." if summary.pooled_5mc_fraction is None else f"{summary.pooled_5mc_fraction:.4f}",
        MMETHN="." if summary.pooled_5mc_n is None else str(summary.pooled_5mc_n),
        REPCNQ=allele_quantiles,
        AD_NSPAN=allele_support_nspan,
        BPCN=summary.size,
    )


def extract_unique_phaseset(read_info_records: Sequence[ReadCall]) -> str:
    """
    Extract and validate a unique phaseset value from the record set.

    Args:
        read_info_records (Sequence[ReadCall]): Typed records containing the group of reads.

    Returns:
        str: The unique phaseset value (without the `ps` prefix) if all values are consistent.
             Returns "." if only "ps0" is present.

    Raises:
        ValueError: If there are multiple unique phaseset values found (excluding "ps0").
    """
    # Extract phaseset values from the `ps` field, ignoring the unphased sentinel `ps0`.
    phasesets = sorted({record.ps for record in read_info_records if record.analysis_exclusion_reason is None and record.ps != "ps0"})

    # If only "ps0" is present, return "."
    if not phasesets:
        return "."

    # Assert that there is only one unique phaseset value
    if len(phasesets) != 1:
        raise ValueError(f"Multiple unique phaseset values found: {phasesets}")

    # Remove the `ps` prefix before returning
    return phasesets[0].removeprefix("ps")


def process_ref_alleles(loci: list[Locus], reference_genome_fasta: str) -> dict[Locus, RefAllele]:
    """
    Process the reference alleles for the loci in the genome FASTA file.
    Args:
        loci (List[Locus]): List of Locus objects.
        reference_genome_fasta (str): Path to the genome FASTA file.
    Returns:
        Dict[Locus, RefAllele]: Dictionary with locus IDs as keys and their reference alleles and its decomposition as values.
    """
    ref_alleles = {}
    with pysam.FastaFile(reference_genome_fasta) as ref:
        for locus in loci:
            repeat_unit = locus.motif
            motifs_set = expand_motif_representation(repeat_unit)
            ref_padded_seq = ref.fetch(reference=locus.chrom, start=locus.start - 1, end=locus.end)
            ref_repeat_dec = process_subread(ref_padded_seq[1:], motifs_set)
            ref_alleles[locus] = RefAllele(padded_base=ref_padded_seq, decomposition=ref_repeat_dec)
    return ref_alleles


def process_coverage(loci: list[Locus], bam: str) -> dict[Locus, int]:
    """
    Process the coverage of the loci in the BAM file.
    Args:
        loci (List[Locus]): List of loci.
        bam (str): Path to the BAM file.
    Returns:
        Dict[Locus, int]: Dictionary with locus objects as keys and their coverage as values.
    """
    cov = {}
    with pysam.AlignmentFile(bam, "rb") as bam_file:
        for locus in loci:
            chrom, start_pos, end_pos = locus.chrom, locus.start, locus.end
            if start_pos == end_pos:
                # get something meaningful for 0-length regions
                start_pos = start_pos - 1
            cov_lst = [int(pileupcolumn.nsegments) for pileupcolumn in bam_file.pileup(chrom, start_pos, end_pos, truncate=True)]
            average_coverage = int(sum(cov_lst) / len(cov_lst)) if len(cov_lst) > 0 else 0
            cov[locus] = average_coverage
    return cov


def form_vcf_body(
    regions: Sequence[RegionCallSet],
    loci: list[Locus],
    ref_alleles: dict[Locus, RefAllele],
    coverages: dict[Locus, int],
    min_read_support_per_allele: int,
    min_region_coverage: int,
    sex: str,
) -> tuple[list[str], list[str]]:
    """
    Form the VCF body from typed records and loci information.

    Args:
        regions (Sequence[RegionCallSet]): Grouped region results.
        loci (List[Locus]): List of loci.
        ref_alleles (Dict[Locus, RefAllele]): Dictionary of reference alleles.
        coverages (Dict[Locus, int]): Dictionary of coverage information.
        min_read_support_per_allele (int): Minimum read support per allele.
        min_region_coverage (int): Minimum coverage for the region.
        sex (str): Sex of the sample ("XY" or "XX").

    Returns:
        Tuple[List[str], List[str]]: Tuple containing the VCF header for ALT and the VCF body lines.
    """
    output_vcf_header_alt = []
    output_vcf_body_lst = []
    regions_by_name = {region.region_name: region for region in regions}

    for locus in loci:
        name_of_group = f"{locus.chrom}_{locus.start}_{locus.end}"
        error_filter: Literal["MotifBias"] | None = None
        haploid = sex == "XY" and is_sex_chr(locus.chrom)
        region = regions_by_name.get(name_of_group)
        if region is not None and region.alleles and any(record.analysis_exclusion_reason is None for record in region.reads):
            vcf_fmt, alts = transform_rows_to_vcf_formats_and_alts(region, haploid=haploid)
            if vcf_fmt == {}:
                logging.debug(f"Skipping {name_of_group} due to insufficient read support")
                continue

            header_lines = [(alt.count, f'##ALT=<ID={alt.id},Description="{alt.description}">\n') for alt in alts]
            output_vcf_header_alt.extend(header_lines)
            alts_str = ",".join([f"<{alt.id}>" for alt in alts]) if alts else "."
            if region.region_is_suspicious:
                error_filter = "MotifBias"

        else:
            # no reads supporting the variant
            vcf_fmt = generate_format_empty(haploid=haploid)
            alts_str = "."

        vcf_fmt["DP"] = f"{coverages[locus]}"
        min_region_coverage_adjusted = min_region_coverage if not haploid else min_region_coverage // 2

        # Check depth-based filters
        depth_filter = get_filter_from_vcf_fmt(
            vcf_fmt, min_region_coverage=min_region_coverage_adjusted, min_read_support_per_allele=min_read_support_per_allele
        )
        # PASS if no filters failed, ;-separated if multiple filters failed.
        # https://github.com/samtools/hts-specs/blob/master/VCFv4.5.tex#L424
        failed_filters: list[str] = [f for f in [depth_filter, error_filter] if f is not None]
        filter: str = ";".join(failed_filters) if failed_filters else "PASS"
        ref_allele = ref_alleles[locus].padded_base
        ref_repeat_dec = ref_alleles[locus].decomposition
        info = {
            "END": f"{locus.end}",
            "REF": f"{len(ref_repeat_dec)}",
            "RL": f"{locus.end - locus.start}",
            "RU": f"{locus.motif}",
            "REPID": f"{locus.variant_id}",
        }

        info_order = ["END", "REF", "RL", "RU", "REPID"]
        info_line = ";".join([f"{key}={info[key]}" for key in info_order])

        format_order = "GT:PS:REPCN:REPCNQ:MRCN:MMETH:MMETHN:AD:AD_NSPAN:DP:CLSTATE:BPCN"
        format_line = ":".join([vcf_fmt[key] for key in format_order.split(":")])

        output_vcf_body_lst.append(
            f"{locus.chrom}\t{locus.start}\t.\t{ref_allele}\t{alts_str}\t.\t{filter}\t{info_line}\t{format_order}\t{format_line}"
        )
    # this will sort first by length of the allele (first item of each tuple) and
    # then alphabetically (second item, needed for reproducible ordering of STRn vs STRna vs STRnb)
    output_vcf_header_alt = sorted(set(output_vcf_header_alt))
    output_vcf_header_alt = [header_line for _, header_line in output_vcf_header_alt]
    return output_vcf_header_alt, output_vcf_body_lst


def get_alts(alleles_info: list[AlleleInformation]) -> list[SymbolicAllele]:
    """
    Get the alternative alleles from a list of AlleleInformation.

    Args:
        alleles_info (List[AlleleInformation]): List of AlleleInformation instances.

    Returns:
        List[SymbolicAllele]: List of alternative alleles.
    """
    allele_counts = [str(allele.REPCN) for allele in alleles_info]
    same_counts = len(set(allele_counts)) == 1 and len(alleles_info) > 1
    alts = []
    for i, allele_count in enumerate(allele_counts):
        allele_letter = string.ascii_lowercase[i]
        alt_id = f"STR{allele_count}{allele_letter}" if same_counts else f"STR{allele_count}"
        alt = SymbolicAllele(id=alt_id, description=f"Allele comprised of {allele_count} repeat units", count=int(allele_count))
        alts.append(alt)
    return alts


def decode_gt(gt_str: str) -> dict[str, tuple[float, int]]:
    """
    Decode the genotype string into a dictionary of allele names and their lengths and sizes.

    Args:
        gt_str (str): Genotype string. Example: "1:10.5(3);2:12.0(5)"

    Returns:
        Dict[str, Tuple[float, int]]: Dictionary with allele names as keys and their lengths and sizes as values.
        Example: {"1": (10.5, 3), "2": (12.0, 5)}
    """
    gt = {}
    for x in gt_str.split(";"):
        allele_name, allele_len, allele_size = (
            x.split(":")[0],
            float(x.split(":")[1].split("(")[0]) if ":" in x and "(" in x.split(":")[1] else ValueError(f"Malformed genotype string: {x}"),
            int(x.split(":")[1][:-1].split("(")[1]),
        )
        gt[allele_name] = (allele_len, allele_size)
    return gt


def compose_vcf_output(
    sample: str,
    output_vcf_header: str,
    output_vcf_header_alt: Iterable[str],
    output_vcf_body_lst: list[str],
) -> tuple[str, str]:
    """
    Compose the final VCF output from the header and body.
    Args:
        sample (str): Sample name.
        output_vcf_header (str): VCF header string.
        output_vcf_header_alt (Iterable[str]): Alternative allele header lines.
        output_vcf_body_lst (List[str]): List of VCF body lines.
    Returns:
        Tuple[str, str]: Tuple containing the final VCF header and body.
    """
    vcf_headers = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", sample]
    output_vcf_body_lst = sorted(
        output_vcf_body_lst,
        key=lambda x: (
            contig_sort_key(x.split("\t")[0]),
            int(x.split("\t")[1]),
        ),
    )
    output_vcf_header += "".join(output_vcf_header_alt)
    tab = "\t"
    output_vcf_header += f"#{tab.join(vcf_headers)}\n"
    output_vcf_body = "".join(f"{line}\n" for line in output_vcf_body_lst)
    return output_vcf_header, output_vcf_body


def write_vcf(
    regions: Sequence[RegionCallSet],
    sample: str,
    loci: list[Locus],
    reference_genome_fasta: str,
    bam: str,
    out_file: str | Path,
    min_read_support_per_allele: int,
    min_region_coverage: int,
    sex: str,
    command: str | None = None,
) -> None:
    """
    Write the VCF file from typed read records.

    Args:
        regions (Sequence[RegionCallSet]): Grouped region results.
        sample (str): Sample name.
        loci (List[Locus]): List of loci.
        reference_genome_fasta (str): Path to the genome FASTA file.
        bam (str): Path to the BAM file.
        out_file (str): Output VCF file path.
        min_read_support_per_allele (int): Minimum read support per allele.
        min_region_coverage (int): Minimum coverage for the region.
        sex (str): Sex of the sample ("XY" or "XX").
        command (str | None): The full mahi-mahi command used to generate the VCF.
    """
    output_vcf_header = form_vcf_header(reference_genome_fasta, command)

    ref_alleles = process_ref_alleles(loci, reference_genome_fasta)
    covs = process_coverage(loci, bam)
    output_vcf_header_alt, output_vcf_body_lst = form_vcf_body(
        regions,
        loci,
        ref_alleles,
        covs,
        min_read_support_per_allele,
        min_region_coverage,
        sex,
    )
    output_vcf_header, output_vcf_body = compose_vcf_output(sample, output_vcf_header, output_vcf_header_alt, output_vcf_body_lst)

    with open(out_file, "w") as out:
        out.write(output_vcf_header)
        out.write(output_vcf_body)


def get_filter_from_vcf_fmt(vcf_fmt: dict[str, str], min_region_coverage: int, min_read_support_per_allele: int) -> str | None:
    """
    Determine the filter status based on DP and AD values in the VCF format.

    Args:
        vcf_fmt (Dict[str, str]): Dictionary containing the VCF format.
        min_region_coverage (int): Minimum coverage for the region.
        min_read_support_per_allele (int): Minimum read support per allele.

    Returns:
        str | None: None if the region passes the filters, otherwise "LowDepth".
    """
    dp = float(vcf_fmt.get("DP", 0))
    if dp < min_region_coverage:
        return "LowDepth"

    ad_values = vcf_fmt.get("AD", "").split(",")
    alt_alleles = ad_values[1:]  # Skip the first value (reference allele depth)
    if any(int(ad) < min_read_support_per_allele for ad in alt_alleles if ad != "."):
        return "LowDepth"

    return None


def filter_loci_for_output(loci: list[Locus]) -> list[Locus]:
    """
    Filter loci to only those that Stranger understands, to avoid warnings.

    Args:
        loci (List[Locus]): List of loci to filter.

    Returns:
        List[Locus]: Filtered list of loci.
    """
    return [locus for locus in loci if is_primary_output_locus(locus)]

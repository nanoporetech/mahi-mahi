from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import pysam

from ..pipeline_records import ReadCall, RegionCallSet
from .bed_utils import MergedLocus

TAGS_AND_FIELDS = [
    ("XH", "allele_id"),
    ("XS", "size"),
    ("XC", "clustered"),
    ("XN", "copy_number"),
    ("XP", "is_spanning"),
    ("XD", "dec"),
    ("XE", "is_suspicious"),
]

FILTERED_TAG = "XF"

BAM_HEADER_ADDITIONAL_LINES = [
    "Tag XH: Allele ID assigned to the read",
    "Tag XS: Repeat size (in bases) assigned to the read",
    "Tag XC: Is the read clustered",
    "Tag XN: Copy number assigned to the read",
    "Tag XP: Is the read spanning the region",
    "Tag XD: Decomposition of the repeat sequence",
    "Tag XF: Supplementary unsplit-2d fragment (filtered from analysis)",
    "Tag XE: Is the read considered suspicious by strand analysis",
]

TagName: TypeAlias = str
TagValue: TypeAlias = str
ReadName: TypeAlias = str
TagCollection: TypeAlias = dict[ReadName, list[TagValue]]


def _iter_region_reads(regions: Sequence[RegionCallSet]) -> Iterator[ReadCall]:
    for region in regions:
        yield from (record for record in region.reads if record.analysis_exclusion_reason is None)


def write_tagged_bam(
    bam_file: str,
    loci: list[MergedLocus],
    out_bam: Path,
    regions: Sequence[RegionCallSet],
    temp_dir: Path,
    unsplit_2d_reads: set[str],
    flank: int = 1000,
) -> None:
    """Write a BAM with analysis tags derived from typed read records.

    The output BAM is sorted and indexed after tags are added so it can be opened
    directly in downstream viewers such as IGV.
    """
    flattened_records = list(_iter_region_reads(regions))
    region_to_max_size = calculate_max_repeat_sizes(flattened_records)
    tag_collections: dict[TagName, TagCollection] = {
        tag: collect_tags(flattened_records, field, region_to_max_size) for tag, field in TAGS_AND_FIELDS
    }

    header = produce_header(bam_file, BAM_HEADER_ADDITIONAL_LINES)

    written_reads = set()
    temp_bam = temp_dir / "unsorted.bam"

    with (
        pysam.AlignmentFile(str(bam_file), "rb") as bam,
        pysam.AlignmentFile(str(temp_bam), "wb", header=pysam.AlignmentHeader.from_dict(header)) as tagged_bam,
    ):
        for locus_key in loci:
            chr_name, start_pos, end_pos = locus_key.chrom, locus_key.start, locus_key.end
            for read_aln_seg in bam.fetch(chr_name, int(start_pos) - flank, int(end_pos) + flank):
                read_key = (read_aln_seg.query_name, read_aln_seg.reference_start, read_aln_seg.reference_end)
                is_filtered = read_aln_seg.query_name in unsplit_2d_reads and read_aln_seg.is_supplementary
                if read_aln_seg.is_secondary or read_key in written_reads:
                    continue
                if is_filtered:
                    read_aln_seg.set_tag(FILTERED_TAG, "filtered")
                else:
                    for tag, _ in TAGS_AND_FIELDS:
                        if read_aln_seg.query_name is not None:
                            tags = tag_collections[tag][read_aln_seg.query_name]
                            if tags:
                                read_aln_seg.set_tag(tag, ";".join(tags))
                tagged_bam.write(read_aln_seg)
                written_reads.add(read_key)

    # Sort the BAM file
    pysam.sort("-o", str(out_bam), str(temp_bam))
    # Index the BAM file
    pysam.index(str(out_bam))


def produce_header(bam_file: str, header_lines: list[str]) -> dict[str, Any]:
    with pysam.AlignmentFile(str(bam_file), "rb") as bam:
        header = bam.header.to_dict()
        for line in header_lines:
            header.setdefault("CO", []).append(line)
    return header


def calculate_max_repeat_sizes(records: Sequence[ReadCall]) -> dict[str, int]:
    """Calculate the maximum size for each region.

    Used for padding the size tag to support proper string sorting in IGV. For example,
    if one allele is `105` and another is `42`, we want `042` and `105` so lexicographic
    ordering still matches numeric ordering.
    """
    region_to_max_size = defaultdict(int)
    for record in records:
        if record.analysis_exclusion_reason is not None:
            continue
        region = f"{record.chrom}:{record.start}-{record.end}"
        region_to_max_size[region] = max(region_to_max_size[region], int(record.size))
    return region_to_max_size


def collect_tags(records: Sequence[ReadCall], field: str, region_to_max_size: dict[str, int]) -> TagCollection:
    """Collect tags for each read based on the requested BAM tag field."""
    tags = defaultdict(list)
    for record in records:
        if record.analysis_exclusion_reason is not None:
            continue
        if field == "size":
            # Add padding to the size to support proper sorting of strings in IGV by repeat size.
            max_size = region_to_max_size[f"{record.chrom}:{record.start}-{record.end}"]
            value = f"{record.size:0{len(str(max_size))}d}"
        else:
            value = getattr(record, field)
        tag = f"{record.chrom}:{record.start}-{record.end}:{value}"
        tags[record.read].append(tag)
    return tags

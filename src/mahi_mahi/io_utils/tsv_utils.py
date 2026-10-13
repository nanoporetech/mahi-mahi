import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..pipeline_records import RegionCallSet, iter_flattened_rows

TSV_COLUMN_FIELDS = [
    ("#chrom", "chrom"),
    ("start", "start"),
    ("end", "end"),
    ("repeat_id", "repeat_id"),
    ("variant_id", "variant_id"),
    ("repeat_unit", "repeat_unit"),
    ("genotype", "genotype"),
    ("read", "read"),
    ("copy_number", "copy_number"),
    ("size", "size"),
    ("read_start", "read_start"),
    ("strand", "strand"),
    ("original_hap", "original_hap"),
    ("allele_id", "allele_id"),
    ("analysis_exclusion_reason", "analysis_exclusion_reason"),
    ("allele_copy_number", "allele_copy_number"),
    ("allele_num_reads_used", "allele_num_reads_used"),
    ("allele_num_nonspanning_reads_used", "allele_num_nonspanning_reads_used"),
    ("clustered", "clustered"),
    ("clstate", "clstate"),
    ("phased", "phased"),
    ("is_spanning", "is_spanning"),
    ("included_in_allele_size_estimation", "included_in_allele_size_estimation"),
    ("allele_size_estimation_quantile", "allele_size_estimation_quantile"),
    ("dec", "dec"),
    ("is_suspicious", "is_suspicious"),
    ("5mC_sites", "mc_sites"),
    ("5mC_fraction", "mc_fraction"),
    ("5mC_nocall_fraction", "mc_nocall_fraction"),
]

TSV_HEADER = [column for column, _ in TSV_COLUMN_FIELDS]


def write_tsv(regions: Sequence[RegionCallSet], filename: str | Path) -> None:
    """Write per-read results as a TSV."""
    with Path(filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for region in regions:
            for record in iter_flattened_rows(region):
                writer.writerow(_project_tsv_row(record))


def _project_tsv_row(record: Mapping[str, object | None]) -> dict[str, object | None]:
    row: dict[str, object | None] = {column: record[field_name] for column, field_name in TSV_COLUMN_FIELDS}
    # "." means "this read has no methylation data", which is distinct from the empty
    # string meaning "this read has methylation data but this TR contains no CpG sites"
    if record["mc_sites"] is None:
        row["5mC_sites"] = "."
    for column, field_name in (("5mC_fraction", "mc_fraction"), ("5mC_nocall_fraction", "mc_nocall_fraction")):
        value = record[field_name]
        if isinstance(value, float):
            row[column] = f"{value:.4f}"
    return row

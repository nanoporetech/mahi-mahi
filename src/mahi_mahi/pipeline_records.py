from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from mahi_mahi.methylation import methylation_counts, methylation_fractions

Strand = Literal["+", "-"]
ReadExclusionReason = Literal[
    "hard_clip_left",
    "hard_clip_right",
    "non_repeated_sequence",
    "non_selected_phaseset",
]

# Runtime allowlist: a future exclusion reason need not automatically be retained in output.
RETAINED_READ_EXCLUSION_REASONS: frozenset[ReadExclusionReason] = frozenset(
    {
        "hard_clip_left",
        "hard_clip_right",
        "non_repeated_sequence",
        "non_selected_phaseset",
    }
)


class ClusteringState(str, Enum):
    """How the reads of one locus were assigned to alleles.

    This records the *route* taken, not whether the call is correct: in particular,
    ``HAPLOTAG_FALLBACK`` is expected when there are two alleles with the same total length.
    """

    SUCCESS = "SUCCESS"
    """Read-length clustering succeeded, producing one or more alleles, and its result was used."""

    HAPLOTAG_FALLBACK = "HAPLOTAG_FALLBACK"
    """Haplotags were used, either to split a single-peak clustering or after a failed one."""

    AMBIGUOUS = "AMBIGUOUS"
    """Clustering failed or was rejected and no haplotags were available, or it was not run
    because there were not enough reads."""

    ASSUMED_HAPLOID = "ASSUMED_HAPLOID"
    """Clustering was not run because the locus is on a sex chromosome of an XY sample."""


@dataclass(frozen=True)
class AlleleAssignmentContext:
    """Identity and location of the locus whose reads are being assigned."""

    chrom: str
    region_name: str
    clustering_id: str

    @property
    def display_name(self) -> str:
        return f"{self.clustering_id}_{self.region_name}"


_ReadRecordT = TypeVar("_ReadRecordT", bound="_ReadRecord")


class _ReadRecord(BaseModel):
    """Shared validated storage for internal and output read records."""

    read: str
    strand: Strand
    ps: str
    original_hap: str
    clustered: bool = False
    is_spanning: bool = False
    included_in_allele_size_estimation: bool = False
    allele_size_estimation_quantile: str = "allele_q50"
    is_suspicious: bool = False
    allele_id: str | None = None
    analysis_exclusion_reason: ReadExclusionReason | None = None

    model_config = ConfigDict(extra="forbid")

    def clone(self: _ReadRecordT, **changes: object) -> _ReadRecordT:
        """Return a deep-copied record with optional field overrides."""
        if changes:
            data = self.model_dump(mode="python", exclude_computed_fields=True)
            data.update(changes)
            return type(self).model_validate(data, strict=True)
        return self.model_copy(deep=True)


class MergedLocusRead(_ReadRecord):
    motif_dec: tuple[str, ...]
    read_start: int
    mc_sites: str | None = None
    """Per-CpG methylation state over the whole merged locus, or None if this read has no methylation data."""

    @computed_field
    @property
    def size(self) -> int:
        """Return the full decomposition length used for clustering."""
        return sum(len(motif) for motif in self.motif_dec)


class ReadCall(_ReadRecord):
    chrom: str
    start: int
    end: int
    repeat_id: str
    variant_id: str
    repeat_unit: str
    copy_number: int
    size: int
    read_start: int
    dec: str
    lst_dec: str
    motif_counts: dict[str, int]
    region_name: str
    mc_sites: str | None = None
    """Per-CpG methylation state over this TR only, or None if this read has no methylation data."""

    @property
    def mc_counts(self) -> tuple[int, int]:
        """Return (methylated sites, called sites) for this read's CpG sites in this TR."""
        return methylation_counts(self.mc_sites)

    @computed_field
    @property
    def mc_fraction(self) -> float:
        """Return the fraction of this read's called CpG sites in this TR that are methylated."""
        return methylation_fractions(self.mc_sites)[0]

    @computed_field
    @property
    def mc_nocall_fraction(self) -> float:
        """Return the fraction of this read's CpG sites in this TR that the basecaller did not call."""
        return methylation_fractions(self.mc_sites)[1]


class AlleleSummary(BaseModel):
    allele_id: str
    copy_number: int
    size: int
    stddev: float
    q10: int
    q25: int
    q50: int
    q75: int
    q90: int
    num_reads_used: int
    num_nonspanning_reads_used: int
    motifs_median_counts: dict[str, int] | None
    pooled_5mc_fraction: float | None = None
    pooled_5mc_n: int | None = None
    quantile_used: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class RegionCallSet(BaseModel):
    chrom: str
    start: int
    end: int
    repeat_id: str
    variant_id: str
    repeat_unit: str
    region_name: str
    phased: bool
    genotype: str
    region_is_suspicious: bool
    clustering_state: ClusteringState = ClusteringState.AMBIGUOUS
    reads: list[ReadCall] = Field(default_factory=list)
    alleles: dict[str, AlleleSummary] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def empty(cls) -> RegionCallSet:
        return cls(
            chrom="",
            start=0,
            end=0,
            repeat_id="",
            variant_id="",
            repeat_unit="",
            region_name="",
            phased=False,
            genotype="",
            region_is_suspicious=False,
            clustering_state=ClusteringState.AMBIGUOUS,
        )

    @classmethod
    def from_reads(
        cls,
        reads: Sequence[ReadCall],
        *,
        alleles: Mapping[str, AlleleSummary] | None = None,
        phased: bool = False,
        genotype: str = "",
        region_is_suspicious: bool = False,
        clustering_state: ClusteringState = ClusteringState.AMBIGUOUS,
    ) -> RegionCallSet:
        if not reads:
            return cls.empty()
        first = reads[0]
        return cls(
            chrom=first.chrom,
            start=first.start,
            end=first.end,
            repeat_id=first.repeat_id,
            variant_id=first.variant_id,
            repeat_unit=first.repeat_unit,
            region_name=first.region_name,
            phased=phased,
            genotype=genotype,
            region_is_suspicious=region_is_suspicious,
            clustering_state=clustering_state,
            reads=list(reads),
            alleles=dict(alleles or {}),
        )


ALLELE_SUMMARY_ROW_FIELDS = {
    "allele_copy_number": "copy_number",
    "allele_num_reads_used": "num_reads_used",
    "allele_num_nonspanning_reads_used": "num_nonspanning_reads_used",
    "allele_std": "stddev",
    "allele_size": "size",
    "allele_q10": "q10",
    "allele_q25": "q25",
    "allele_q50": "q50",
    "allele_q75": "q75",
    "allele_q90": "q90",
    "motifs_median_counts": "motifs_median_counts",
}


def format_genotype(reads: Sequence[ReadCall], alleles: Mapping[str, AlleleSummary]) -> str:
    """Build the legacy genotype string using the order of selected reads.

    Example:
        Included reads for allele summaries `1 -> copy_number=18, num_reads_used=3`
        and `2 -> copy_number=24, num_reads_used=2` return
        `"1:18(3);2:24(2)"`.
    """
    seen: list[str] = []
    for record in reads:
        if not record.included_in_allele_size_estimation:
            continue
        allele_id = record.allele_id or ""
        summary = alleles.get(allele_id)
        if summary is None:
            continue
        allele_info = f"{allele_id}:{summary.copy_number}({summary.num_reads_used})"
        if allele_info not in seen:
            seen.append(allele_info)
    return ";".join(seen)


def iter_flattened_rows(region: RegionCallSet) -> Iterator[dict[str, object | None]]:
    """Project grouped region/allele summaries back into the legacy row contract."""
    for record in region.reads:
        allele_summary = region.alleles.get(record.allele_id or "")
        include_summary = allele_summary is not None and (record.included_in_allele_size_estimation or record.allele_id not in (None, "", "0"))
        if include_summary:
            assert allele_summary is not None
            allele_fields = {row_name: getattr(allele_summary, model_name) for row_name, model_name in ALLELE_SUMMARY_ROW_FIELDS.items()}
        else:
            allele_fields = dict.fromkeys(ALLELE_SUMMARY_ROW_FIELDS, None)
        yield (
            record.model_dump(mode="python")
            | {
                "genotype": region.genotype,
                "phased": region.phased,
                "region_is_suspicious": region.region_is_suspicious,
                "clstate": region.clustering_state.value,
            }
            | allele_fields
        )

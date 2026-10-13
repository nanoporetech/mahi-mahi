#!/usr/bin/env python
import argparse
import collections
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import pysam
from trviz.decomposer import Decomposer

from mahi_mahi.utils import expand_motif_representation

_REGIONS_TABLE_HEADER = "# chrom\tstart\tend\tmotif\trepeat_id\tvariant_id"


def load_regions_table(regions_file: Path) -> tuple[pd.DataFrame, list[tuple[str, str | None]]]:
    """
    Load regions table from mahi-mahi BED file into a DataFrame.

    Args:
        regions_file (Path): Path to the BED file containing regions data.
    Returns:
        tuple: (DataFrame indexed by variant_id, list of (comment_line, variant_id) pairs)
            where variant_id is the variant the comment precedes, or None for trailing comments.
            The standard column-header line is not included in the comments list.
    """
    comments: list[tuple[str, str | None]] = []
    data_rows: list[dict] = []
    pending_comments: list[str] = []

    with open(regions_file) as fh:
        for line in fh:
            line_stripped = line.rstrip("\n")
            stripped = line_stripped.strip()
            if not stripped or stripped.startswith("#"):
                if stripped != _REGIONS_TABLE_HEADER:
                    pending_comments.append(line_stripped)
                continue
            parts = line_stripped.split("\t")
            if len(parts) < 6:
                raise ValueError(f"Data line with fewer than 6 columns in regions file {regions_file}: {line_stripped}")
            start = round(float(parts[1]))
            end = round(float(parts[2]))
            variant_id = parts[5]
            for comment in pending_comments:
                comments.append((comment, variant_id))
            pending_comments = []
            data_rows.append({"chrom": parts[0], "start": start, "end": end, "motif": parts[3], "repeat_id": parts[4], "variant_id": variant_id})

    for comment in pending_comments:
        comments.append((comment, None))

    regions_table = pd.DataFrame(data_rows, columns=pd.Index(["chrom", "start", "end", "motif", "repeat_id", "variant_id"]))
    regions_table = regions_table.set_index("variant_id")
    regions_table["reference_length"] = regions_table["end"] - regions_table["start"]
    return regions_table, comments


def save_regions_table(regions_table: pd.DataFrame, output_file: Path, comments: list[tuple[str, str | None]] | None = None) -> None:
    """Save regions table DataFrame to a BED file.

    Args:
        regions_table (pd.DataFrame): DataFrame indexed by variant_id.
        output_file (Path): Path to the output BED file.
        comments (list[tuple[str, str | None]] | None): Comments as returned by load_regions_table.
            If provided, they are reinserted before the corresponding variant_id row;
            those with variant_id=None are appended at the end.
    """
    regions_table_out = regions_table.reset_index(drop=False).rename(columns={"chrom": "# chrom"})
    regions_table_out = regions_table_out.loc[:, ["# chrom", "start", "end", "motif", "repeat_id", "variant_id"]]
    if not comments:
        # mahi-mahi doesn't actually use the header row, but it's useful as a comment to human readers
        regions_table_out.to_csv(output_file, sep="\t", index=False)
        return
    comments_before: dict[str | None, list[str]] = collections.defaultdict(list)
    for comment_text, variant_id in comments:
        comments_before[variant_id].append(comment_text)
    with open(output_file, "w") as fh:
        fh.write(_REGIONS_TABLE_HEADER + "\n")
        for _, row in regions_table_out.iterrows():
            variant_id = row["variant_id"]
            for comment in comments_before.get(variant_id, []):
                fh.write(comment + "\n")
            fh.write("\t".join(str(v) for v in row) + "\n")
        for comment in comments_before.get(None, []):
            fh.write(comment + "\n")


def convert_stranger_json_to_regions_table(  ## noqa: C901
    stranger_catalog_file: Path, *, fix_composite_TRs: bool = True, convert_names: bool, sort_order: Literal["position", "name", None] = "position"
) -> pd.DataFrame:
    """Convert Stranger/ExpansionHunter variant catalog JSON file to a regions table DataFrame.
    Args:
        stranger_catalog_file (Path): Path to the Stranger variant catalog JSON file.
        fix_composite_TRs (bool): Whether to fix composite TRs whose original form is incompatible with mahi-mahi.
        convert_names (bool): Whether to convert chromosome and variant names to our usual format.
        sort_order (Literal["position", "name", None]): Whether to sort the resulting table by position, by variant name, or not at all (None).
    """
    regions_table = pd.DataFrame(columns=pd.Index(["chrom", "start", "end", "motif", "repeat_id"]))
    regions_table.index.name = "variant_id"
    with open(stranger_catalog_file) as f:
        stranger_variants = json.load(f)
    strip_variant_names = convert_names and sum("_" in variant["LocusId"] for variant in stranger_variants) > 0.5 * len(stranger_variants)
    for variant in stranger_variants:
        motifs = re.findall(r"\(([A-Z]+)\)[*+]", variant["LocusStructure"])  # + is to accept older Stranger versions
        if variant["LocusStructure"] != "".join("(" + motif + ")*" for motif in motifs):
            if (
                "HTT" in variant["LocusId"]
                and variant["LocusStructure"] == "(CAG)*CAACAG(CCG)*"
                and len(variant["VariantId"]) == 2
                and len(variant["ReferenceRegion"]) == 2
            ):
                if fix_composite_TRs:
                    motifs = ["CAG", "CAACAG", "CCG"]
                    variant["VariantId"] = [variant["VariantId"][0], variant["LocusId"] + "_nonrepeat", variant["VariantId"][1]]
                    nonrepeat_region = (
                        variant["ReferenceRegion"][0].split(":")[0]
                        + ":"
                        + variant["ReferenceRegion"][0].split("-")[1]
                        + "-"
                        + variant["ReferenceRegion"][1].split(":")[1].split("-")[0]
                    )
                    variant["ReferenceRegion"] = [variant["ReferenceRegion"][0], nonrepeat_region, variant["ReferenceRegion"][1]]
            elif ("ATXN7" in variant["LocusId"] and variant["LocusStructure"] == "(GCA)*(GCC)+") or (
                "BEAN" in variant["LocusId"] and variant["LocusStructure"] == "(TGGAA)*" + "TAAAA" * 10
            ):
                pass
            else:
                logging.warning(f"Unexpected LocusStructure format: {variant['LocusId']} {variant['LocusStructure']}")
        if "VariantId" not in variant or type(variant["VariantId"]) is str:
            variant["VariantId"] = [variant["LocusId"]]
            variant["ReferenceRegion"] = [variant["ReferenceRegion"]]
        # when using a composite TR to represent a locus with pathogenic motif != reference motif,
        # there should be 3 parts: reference motif, pathogenic motif, reference motif.
        zero_length_ref_genes = ["DAB1", "EIF4A3", "MARCHF6", "RAI1", "RAPGEF2", "SAMD12", "STARD7", "TNRC6A", "YEATS2"]
        if fix_composite_TRs and variant["LocusId"].split("_")[-1] in zero_length_ref_genes and len(variant["ReferenceRegion"]) > 1:
            zero_length_part_idx = variant["VariantId"].index(variant["LocusId"])
            # this is -1 when the parts are listed in reverse order (sometimes done for negative-strand genes e.g. Stranger SAMD12)
            part_order = -1 if int(variant["ReferenceRegion"][0].split("-")[1]) > int(variant["ReferenceRegion"][-1].split("-")[1]) else 1
            # whether the pathogenic part is at the high-coordinate end of the TR (as opposed to the low-coordinate one)
            zero_part_at_high_end = 0 <= zero_length_part_idx - part_order < len(variant["ReferenceRegion"])
            if zero_part_at_high_end:
                new_boundary = variant["ReferenceRegion"][zero_length_part_idx].split("-")[1]
                variant["ReferenceRegion"][zero_length_part_idx - part_order] = (
                    variant["ReferenceRegion"][zero_length_part_idx - part_order].split("-")[0] + "-" + new_boundary
                )
                if 0 <= zero_length_part_idx + part_order < len(variant["ReferenceRegion"]):
                    variant["ReferenceRegion"][zero_length_part_idx + part_order] = (
                        variant["ReferenceRegion"][zero_length_part_idx + part_order].split(":")[0]
                        + ":"
                        + new_boundary
                        + "-"
                        + variant["ReferenceRegion"][zero_length_part_idx + part_order].split("-")[1]
                    )
            else:
                new_boundary = variant["ReferenceRegion"][zero_length_part_idx].split(":")[1].split("-")[0]
                variant["ReferenceRegion"][zero_length_part_idx + part_order] = (
                    variant["ReferenceRegion"][zero_length_part_idx + part_order].split(":")[0]
                    + ":"
                    + new_boundary
                    + "-"
                    + variant["ReferenceRegion"][zero_length_part_idx + part_order].split("-")[1]
                )
            variant["ReferenceRegion"][zero_length_part_idx] = (
                variant["ReferenceRegion"][zero_length_part_idx].split(":")[0] + ":" + new_boundary + "-" + new_boundary
            )
            if len(variant["ReferenceRegion"]) == 2:
                # A 2-part composite TR implicitly assumes that the pathogenic insertion happens at that end of the TR.
                # Avoid that assumption by adding a second copy of the reference-motif part on the other side of the
                # pathogenic part, so that the expansion can be flanked by the reference motif.  The new part is taken
                # from the adjacent end of the existing reference part (rather than being zero-length, which would give
                # two adjacent zero-length parts); the exact position of internal boundaries of a composite TR does not
                # matter, only the order of the parts.
                reference_part_idx = 1 - zero_length_part_idx
                # keep the new part adjacent to the pathogenic part, on the opposite side from the existing reference part
                insert_idx = zero_length_part_idx + 1 if reference_part_idx < zero_length_part_idx else zero_length_part_idx
                chrom_name, reference_positions = variant["ReferenceRegion"][reference_part_idx].split(":")
                reference_start, reference_end = [int(pos) for pos in reference_positions.split("-")]
                new_part_length = min(5, reference_end - reference_start)
                boundary = int(new_boundary)
                if zero_part_at_high_end:
                    new_boundary = str(boundary - new_part_length)
                    new_part_region = f"{chrom_name}:{new_boundary}-{boundary}"
                    variant["ReferenceRegion"][reference_part_idx] = f"{chrom_name}:{reference_start}-{new_boundary}"
                else:
                    new_boundary = str(boundary + new_part_length)
                    new_part_region = f"{chrom_name}:{boundary}-{new_boundary}"
                    variant["ReferenceRegion"][reference_part_idx] = f"{chrom_name}:{new_boundary}-{reference_end}"
                variant["ReferenceRegion"][zero_length_part_idx] = f"{chrom_name}:{new_boundary}-{new_boundary}"
                variant["VariantId"].insert(insert_idx, variant["VariantId"][reference_part_idx] + "2")
                variant["ReferenceRegion"].insert(insert_idx, new_part_region)
                motifs.insert(insert_idx, motifs[reference_part_idx])
                logging.info(
                    f"Variant {variant['LocusId']} used a 2-part composite TR to represent a pathogenic motif that is "
                    f"different from the reference motif; converted it to a 3-part one by adding {new_part_length}-base part "
                    f"{variant['VariantId'][insert_idx]}."
                )

        for variant_id, reference_region, motif in zip(variant["VariantId"], variant["ReferenceRegion"], motifs, strict=True):
            repeat_id = variant["LocusId"]  # this is inside the loop to not already be stripped by a previous variant_id
            if strip_variant_names and "_" in repeat_id and "ARX" not in repeat_id:
                # convert STRchive disease_gene variant names to just gene names
                # excluding ARX because that would make the names non-unique
                variant_id = variant_id.split("_", 1)[1]
                repeat_id = repeat_id.split("_", 1)[1]
            chrom, positions = reference_region.split(":")
            if convert_names and "chr" not in chrom.lower():
                # convert Stranger number-only chromosome names to chrN format
                chrom = "chr" + chrom
            start, end = [int(pos) for pos in positions.split("-")]
            if "XYLT1" in variant["LocusId"] and end - start < 80:  # end-start=15 in GRCh37/38, 93 in T2T
                logging.warning(
                    "XYLT1 is missing ~150 bases of flanking sequence in the GRCh37 and GRCh38 references, which will cause incorrect results; "
                    "we do not recommend including it if using such a reference"
                )
            if len(motif) == 1:  # homopolymer
                if (
                    fix_composite_TRs and variant_id == repeat_id + "_" + motif
                ):  # non-primary part of a composite TR - convert to multi-base homopolymer motif
                    logging.warning(
                        f"Variant {variant_id} has single-base motif {motif}, which is not allowed in mahi-mahi.  "
                        f"Replacing with {motif * 4}, and renaming because this will change counts."
                    )
                    variant_id = variant_id + motif * 3
                    motif = motif * 4
                else:  # primary TR - let it fail, converting would change the counts
                    logging.warning(
                        f"Variant {variant_id} has single-base motif {motif}, which is not allowed in mahi-mahi.  "
                        f"Multi-base homopolymers (e.g. {motif * 3}) are allowed, but remember to rescale any count thresholds.  "
                        "Also be aware that homopolymer lengths in general are subject to increased sequencing error."
                    )
            extra_motifs = variant.get("PathogenicMotifs", []) + variant.get("BenignMotifs", []) + variant.get("Motifs", [])
            extra_motifs = [extra_motif for extra_motif in extra_motifs if extra_motif not in expand_motif_representation(",".join(motifs))]
            if len(variant["VariantId"]) > 1 and len(expand_motif_representation(motif)) + len(extra_motifs) > 1:
                logging.warning(f"Variant {variant_id} is both a composite TR and a mixed-motif TR, which is not allowed")
            if extra_motifs:
                motif += "," + ",".join(extra_motifs)
            if variant_id in regions_table.index:
                raise KeyError(f"Duplicate VariantId {variant_id} in Stranger catalog")
            reference_count = (end - start) / len(motif.split(",")[0])  # this assumes the reference motif is the same length as the first motif
            if repeat_id == variant_id and variant.get("PathologicMin", float("inf")) <= reference_count + 3:
                logging.warning(
                    f"Variant {variant_id} has reference count {reference_count} close to or above pathogenic threshold "
                    f"{variant.get('PathologicMin', float('inf'))}, which may cause it to be misclassified as pathogenic "
                    f"if the region boundaries are not defined carefully. Run `mahi-mahi-loci-tools check` and read any "
                    f"warnings from this locus. (If this is a locus where only some of {motif} are pathogenic, "
                    "note that only overall (not per-motif) counts are passed to Stranger.)"
                )
            regions_table.loc[variant_id] = {
                "chrom": chrom,
                "start": start,
                "end": end,
                "motif": motif,
                "repeat_id": repeat_id,
            }
    # sort variants if requested
    return sort_regions_table(regions_table, sort_order)


def sort_regions_table(regions_table: pd.DataFrame, sort_order: Literal["position", "name", None] = "position") -> pd.DataFrame:
    """Sort a regions table by position or by name.

    Args:
        regions_table (pd.DataFrame): Regions table indexed by variant_id.
        sort_order (Literal["position", "name", None]): Whether to sort by position, by variant name, or not at all (None).
    Returns:
        pd.DataFrame: The sorted regions table.
    """
    if sort_order == "position":
        regions_table = regions_table.sort_values(by=["start", "end"], key=lambda x: x.astype(int)).sort_values(
            by=["chrom"], key=lambda x: x.str.replace("chr", "").str.replace("X", "23").str.replace("Y", "24").astype(float), kind="stable"
        )
    elif sort_order == "name":
        regions_table = regions_table.sort_values(by=["repeat_id"], kind="stable")
    elif sort_order is not None:
        raise ValueError(f"Invalid sort_order {sort_order}, must be 'position', 'name', or None")
    return regions_table


def compare_loci_sets(regions_table1: pd.DataFrame, regions_table2: pd.DataFrame) -> None:  # noqa: C901
    """Compare two regions tables and log any mismatches (i.e. possible errors) in coordinates or motifs

    Loci are matched by index (i.e. variant_id), so the two lists should use the same name format but need not be in the same order.

    Args:
        regions_table1, regions_table2 (pd.DataFrame): Regions tables to compare.
    """
    shared_variant_ids = set(regions_table1.index) & set(regions_table2.index)
    for variant_id in shared_variant_ids:
        row1 = regions_table1.loc[variant_id]
        row2 = regions_table2.loc[variant_id]
        motifs1 = set(expand_motif_representation(row1["motif"]))
        motifs2 = set(expand_motif_representation(row2["motif"]))
        motifs2_rolled = [{motif[i:] + motif[:i] for motif in motifs2} for i in range(min(len(motif) for motif in motifs2))]
        mismatch_reason = ""
        if not (
            row1["chrom"] == row2["chrom"]
            and (row1["start"] - row1["end"]) == (row2["start"] - row2["end"])
            and abs(row1["start"] - row2["start"]) < 3
        ):
            mismatch_reason = "coordinate mismatch"
        elif row1["start"] != row2["start"]:
            mismatch_reason = "1-2 base shift (this is OK to align region start with motif)"
        if motifs1 != motifs2:
            if mismatch_reason:
                mismatch_reason += ", and "
            if (motifs1 > motifs2) or (motifs2 > motifs1):
                mismatch_reason += "motifs added or removed (this is OK to report counts for more motifs)"
            elif motifs1 in motifs2_rolled:
                mismatch_reason += "motif cyclic shift (this is OK to align region start with motif)"
            else:
                mismatch_reason += "motif mismatch"
        if mismatch_reason:
            logging.warning(
                f"Mismatch for VariantId {variant_id}: {mismatch_reason}, "
                f"ours {row1['chrom']}:{row1['start']}-{row1['end']} motif {row1['motif']}, "
                f"other {row2['chrom']}:{row2['start']}-{row2['end']} motif {row2['motif']}, "
                f"reference (our TR region in uppercase) {row1.get('reference_sequence_with_flanks', 'unknown')}"
            )
    set1_only = sorted(set(regions_table1.index) - set(regions_table2.index))
    set2_only = sorted(set(regions_table2.index) - set(regions_table1.index))
    if set1_only:
        logging.warning(f"Only in first set: {set1_only}")
    if set2_only:
        logging.warning(f"Only in second set: {set2_only}")
        if set1_only:
            logging.warning("(this only compares loci with the same name, so cannot effectively compare loci sets that use different name formats)")


def check_loci_match_reference(regions_table: pd.DataFrame, reference_file: Path, ignore_small_differences: bool = False) -> pd.DataFrame:  # noqa: C901
    """Check that these loci appear valid and match the reference sequence

    (i.e. most, though not necessarily all, of the reference region is an expected motif)

    Args:
        regions_table (pd.DataFrame): DataFrame of loci (will be modified).
        reference_file (Path): Path to the reference genome FASTA file.
        ignore_small_differences (bool): Whether to ignore issues that change the count by only 1-2 motifs
        (e.g. regions that start or end on an incomplete motif)."""
    reference_fasta = pysam.FastaFile(str(reference_file))
    for idx, row in regions_table.iterrows():
        chrom: str = row["chrom"]  # type: ignore
        start: int = row["start"]  # type: ignore
        end: int = row["end"]  # type: ignore
        motifs_str: str = row["motif"]  # type: ignore
        motifs = expand_motif_representation(motifs_str)
        if start > end:
            raise ValueError(f"Region {idx} start {start} is greater than end {end} - typo?")
        if chrom not in reference_fasta.references:
            raise KeyError(f"Region {idx} chromosome {chrom} not found in reference FASTA - typo? mismatched chromosome name format?")
        if end > reference_fasta.get_reference_length(chrom):
            raise IndexError(
                f"Region {idx} end {end} exceeds length of chromosome {chrom} - typo? mismatched reference version (for regions near the end)?"
            )
        # 'probably' because these are approximate boundaries, the exact ones depend on reference version
        if chrom[-1] in ["X", "Y"] and (start < 2.8e6 or end > reference_fasta.get_reference_length(chrom) - 4e5):
            logging.warning(f"Region {idx} {chrom}:{start}-{end} is probably in the chrX/Y PAR region, which mahi-mahi is not designed to handle")
        sequence = reference_fasta.fetch(chrom, start, end).upper()
        sequence_with_flanks = (
            reference_fasta.fetch(chrom, start - 15, start).lower() + sequence.upper() + reference_fasta.fetch(chrom, end, end + 15).lower()
        )
        regions_table.at[idx, "reference_sequence_with_flanks"] = sequence_with_flanks
        if max(len(motif) for motif in motifs) > 24:  # this is a gradual "longer is worse", not a sharp limit
            # the issue is partly sequencing error and partly that long-motif TRs are often also imperfect repeats
            logging.warning(f"Region {idx} motif {motifs_str} has very long motifs, which mahi-mahi is not designed to handle")
        if start == end:
            # avoid crashing on length-0 reference alleles
            regions_table.at[idx, "motif_plus_interruptions"] = motifs_str
            regions_table.at[idx, "interruption_count"] = 0
            continue
        motif_length: int | None = None
        reference_motifs = Decomposer().decompose(sequence, motifs)
        # always use the full decomposition for interruption analysis
        interruptions = collections.Counter(motif for motif in reference_motifs if motif not in motifs)
        regions_table.at[idx, "motif_plus_interruptions"] = ",".join([motifs_str, *interruptions.keys()])
        interruption_count = sum(interruptions.values())
        regions_table.at[idx, "interruption_count"] = interruption_count
        # when all motifs have the same length, split the sequence into equal-length parts for most common motif checking
        # to ensure we notice cyclic shifts
        if not any(len(motif) != len(motifs[0]) for motif in motifs):
            motif_length = len(motifs[0])
            reference_motifs = [sequence[i : i + motif_length] for i in range(0, len(sequence), motif_length)]
            repeat_count = len(sequence) / motif_length
            if (not repeat_count.is_integer()) and (not ignore_small_differences):
                logging.warning(
                    f"Region {idx} {chrom}:{start}-{end} motif {motifs_str} length {motif_length} "
                    f"does not divide length {len(sequence)} of reference sequence {sequence}; "
                    "this implies an incomplete motif at the start or end of the region, which may or may not get counted. "
                    "This message can be ignored if you don't care about small (1-2 motif) count differences."
                )
        reference_most_common_motif = max(set(reference_motifs), key=reference_motifs.count)
        motifs_rolled = [motif[i:] + motif[:i] for motif in motifs for i in range(len(motif))]
        if reference_most_common_motif not in motifs_rolled:
            logging.warning(
                f"Region {idx} {chrom}:{start}-{end} given motif {motifs_str} does not include the "
                f"most common motif {reference_most_common_motif} of reference sequence {sequence}.\n\n"
                "If this was intended to exclude a non-pathogenic reference motif from the total count, it will not work: "
                "mahi-mahi's total count is everything between the given coordinates not just everything that "
                "matches one of the given motifs.  Consider using a 3-part composite TR instead.\n\n"
                "If you expected a reference allele that did mostly match, check that the coordinates are for the version "
                "of the reference genome you are using, and that the motif is given in the reference orientation."
            )
        elif reference_most_common_motif not in motifs:
            if not ignore_small_differences:
                logging.warning(
                    f"Region {idx} {chrom}:{start}-{end} given motif {motifs_str} is a cyclic shift "
                    f"(i.e. region does not start on a motif boundary, which is likely to add 1 to the apparent count) "
                    f"of most common motif {reference_most_common_motif} of reference sequence {sequence}. "
                    "To fix this, either rotate the motif to match the region boundaries, or shift the region boundaries to match the motif. "
                    "This message can be ignored if you don't care about small (1-2 motif) count differences."
                )
        elif interruption_count > 3:
            sequence_interruptions_marked = "".join(motif.upper() if motif not in motifs else motif.lower() for motif in reference_motifs)
            logging.warning(
                f"Region {idx} {chrom}:{start}-{end} motif {motifs_str} has {interruption_count} > 3 "
                f"repeats in reference sequence {sequence_interruptions_marked} that are not any of its expected motifs. "
                "These motifs will be included in the total count, but there will probably not be per-motif counts for them. "
                "If you are OK with this, this message can be ignored. "
                "If you do want per-motif counts, add them to the expected motif list, but "
                "be aware that motif counts that are small fractions of the total count are subject to sequencing error."
            )

    reference_fasta.close()
    return regions_table


def _boundary_trim_length(boundary_motif: str, motifs: list[str], primary_motif_length: int, *, at_start: bool) -> int | None:
    """Number of bases to trim from one end of a region to remove a partial motif there.

    Args:
        boundary_motif (str): The first (if at_start) or last decomposed motif of the reference sequence.
        motifs (list[str]): The expected motifs of this region.
        primary_motif_length (int): Length of the first given motif, i.e. how short a motif must be to be a partial one.
        at_start (bool): Whether this is the start (as opposed to the end) of the region, i.e. which side the partial motif is on.
    Returns:
        int | None: Number of bases to trim (0 if nothing needs trimming), or None if this end does not
        look like an expected motif possibly preceded/followed by a partial one.
    """
    if boundary_motif in motifs:
        return 0
    if len(boundary_motif) < primary_motif_length:
        return len(boundary_motif)
    # e.g. a region with motif CCG ending in CCGC, which is one full motif plus a partial one
    # prefer the longest matching motif, as that leaves the smallest partial motif to trim
    for motif in sorted(motifs, key=len, reverse=True):
        matches = boundary_motif.endswith(motif) if at_start else boundary_motif.startswith(motif)
        if matches and 0 < len(boundary_motif) - len(motif) < primary_motif_length:
            return len(boundary_motif) - len(motif)
    return None


def trim_locus_boundaries(regions_table: pd.DataFrame, reference_file: Path) -> pd.DataFrame:  # noqa: C901
    """Move region boundaries inward to remove partial motifs at the start and/or end of the reference allele.

    The reference sequence of each region is decomposed into motifs; if the first and/or last of these is
    a partial motif, i.e. shorter than the first given motif, the boundary is moved past it. This includes
    partial motifs that the decomposition merged into an adjacent full motif, e.g. a CCG region ending in CCGC.
    Ends whose motif is not one of the given motifs and cannot be split this way are left alone (with a
    warning), because those are more likely to be a genuine locus definition disagreement.

    Boundaries that are shared with another section of the same composite TR are never moved, as that would
    leave a gap between the sections.

    Args:
        regions_table (pd.DataFrame): DataFrame of loci (will be modified).
        reference_file (Path): Path to the reference genome FASTA file.
    Returns:
        pd.DataFrame: The regions table with trimmed start/end coordinates.
    """
    reference_fasta = pysam.FastaFile(str(reference_file))
    for idx, row in regions_table.iterrows():
        chrom: str = row["chrom"]  # type: ignore
        start: int = row["start"]  # type: ignore
        end: int = row["end"]  # type: ignore
        motifs_str: str = row["motif"]  # type: ignore
        if start >= end:  # zero-length reference allele, nothing to trim
            continue
        same_repeat = regions_table[(regions_table["repeat_id"] == row["repeat_id"]) & (regions_table.index != idx)]
        start_is_shared = bool((same_repeat["end"] == start).any())
        end_is_shared = bool((same_repeat["start"] == end).any())
        motifs = expand_motif_representation(motifs_str)
        primary_motif_length = len(motifs_str.split(",")[0])
        sequence = reference_fasta.fetch(chrom, start, end).upper()
        reference_motifs = Decomposer().decompose(sequence, motifs)
        if len(reference_motifs) < 2:
            # a single decomposed motif gives no repeat to keep after trimming
            continue
        trim_at_start = 0
        trim_at_end = 0
        first_motif = reference_motifs[0]
        start_trim_length = _boundary_trim_length(first_motif, motifs, primary_motif_length, at_start=True)
        if start_trim_length is None:
            logging.warning(
                f"Region {idx} {chrom}:{start}-{end} motif {motifs_str} starts with {first_motif}, which is neither one of its motifs "
                "nor one of them plus a partial one, so it was not trimmed. Check whether the region or the motif list is wrong."
            )
        elif start_trim_length > 0:
            if start_is_shared:
                logging.warning(
                    f"Region {idx} {chrom}:{start}-{end} motif {motifs_str} starts with partial motif in {first_motif}, but its start is shared "
                    "with another section of the same composite TR, so it was not trimmed (that would leave a gap between the sections). "
                    "This can usually be ignored, as mahi-mahi mostly does not use internal boundaries of composite TRs."
                )
            else:
                trim_at_start = start_trim_length
        last_motif = reference_motifs[-1]
        end_trim_length = _boundary_trim_length(last_motif, motifs, primary_motif_length, at_start=False)
        if end_trim_length is None:
            logging.warning(
                f"Region {idx} {chrom}:{start}-{end} motif {motifs_str} ends with {last_motif}, which is neither one of its motifs "
                "nor one of them plus a partial one, so it was not trimmed. Check whether the region or the motif list is wrong."
            )
        elif end_trim_length > 0:
            if end_is_shared:
                logging.warning(
                    f"Region {idx} {chrom}:{start}-{end} motif {motifs_str} ends with partial motif in {last_motif}, but its end is shared "
                    "with another section of the same composite TR, so it was not trimmed (that would leave a gap between the sections). "
                    "This can usually be ignored, as mahi-mahi mostly does not use internal boundaries of composite TRs."
                )
            else:
                trim_at_end = end_trim_length
        if trim_at_start + trim_at_end == 0:
            continue
        new_start, new_end = start + trim_at_start, end - trim_at_end
        message = (
            f"Region {idx} {chrom}:{start}-{end} motif {motifs_str} trimmed by {trim_at_start}+{trim_at_end} bases to {chrom}:{new_start}-{new_end}"
        )
        if trim_at_start + trim_at_end >= 3:
            logging.warning(message + ">= 3")
        else:
            logging.info(message)
        regions_table.at[idx, "start"] = new_start
        regions_table.at[idx, "end"] = new_end
        if "reference_length" in regions_table.columns:
            regions_table.at[idx, "reference_length"] = new_end - new_start

    reference_fasta.close()
    return regions_table


def load_and_check_loci(
    regions_file: Path,
    reference_file: Path,
    stranger_regions_file: Path | None = None,
    test_reads_file: Path | None = None,
    *,
    ignore_small_differences: bool = False,
) -> pd.DataFrame:
    """Load loci from a mahi-mahi BED file and check them in all available ways.

    Args:
        regions_file (Path): Path to the BED file containing regions data.
        reference_file (Path): Path to the reference genome FASTA file.
        stranger_regions_file (Path | None): Path to a Stranger variant catalog JSON file to compare against, or None to skip this check.
        test_reads_file (Path | None): Path to a test reads file to check the loci against, or None to skip this check.
    Returns:
        pd.DataFrame: DataFrame of loci with any added information from the checks.
    If test_reads_file is given, output files are in the same directory as regions_file."""
    regions_table, _ = load_regions_table(regions_file)
    regions_table = check_loci_match_reference(regions_table, reference_file, ignore_small_differences=ignore_small_differences)
    if stranger_regions_file is not None:
        stranger_regions_table = convert_stranger_json_to_regions_table(stranger_regions_file, convert_names=True)
        compare_loci_sets(regions_table, stranger_regions_table)
    if test_reads_file is not None:
        output_dir = Path(regions_file).parent
        sample_name = Path(regions_file).stem + "_locitest"
        if any((output_dir / (sample_name + ext)).exists() for ext in [".vcf", ".tsv", ".tagged.bam", ".log"]):
            raise FileExistsError(
                "Skipping test reads run because it would overwrite the existing "
                f"{output_dir / sample_name}.vcf, .tsv, .tagged.bam, .log. "
                "If you do wish to run it, please move or delete these files first."
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mahi_mahi.main",
                "--bam",
                test_reads_file,
                "--ref",
                reference_file,
                "--loci",
                regions_file,
                "--sample",
                sample_name,
                "--sex",
                "XY",
                "-o",
                str(output_dir),
                # The output directory is the loci file's own directory, so it is never empty.
                "--overwrite",
            ],
            check=True,
        )
        logging.warning(
            "This runs mahi-mahi on the test reads, but does not attempt to check that the output makes sense. "
            f"It can be found at {output_dir}/{sample_name}.vcf"
        )
    return regions_table


def _load_table_from_bed_or_json(path: Path, *, convert_names: bool = True) -> pd.DataFrame:
    """Load a regions table from a BED or JSON file, auto-detecting the format."""
    if path.suffix == ".json":
        return convert_stranger_json_to_regions_table(path, convert_names=convert_names)
    regions_table, _ = load_regions_table(path)
    return regions_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Tools for creating and verifying mahi-mahi BED region files.")
    parser.add_argument(
        "--ignore-small-differences",
        action=argparse.BooleanOptionalAction,
        help="Ignore issues that change the count by only 1-2 motifs (default true for convert, default false for check).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # convert
    convert_parser = subparsers.add_parser("convert", help="Convert a Stranger/ExpansionHunter JSON catalog to a mahi-mahi BED file.")
    convert_parser.add_argument("json", type=Path, help="Input Stranger/ExpansionHunter JSON catalog file")
    convert_parser.add_argument("output_bed", type=Path, help="Output BED file path")
    convert_parser.add_argument("reference", type=Path, nargs="?", default=None, help="Optional reference genome FASTA file (for checking)")
    convert_parser.add_argument("--no-fix-composite-trs", action="store_true", help="Do not fix composite TRs")
    convert_parser.add_argument("--no-convert-names", action="store_true", help="Do not convert chromosome/variant names")

    # check
    check_parser = subparsers.add_parser("check", help="Load loci from a BED file and check that they appear to be valid.")
    check_parser.add_argument("bed", type=Path, help="Input mahi-mahi BED file")
    check_parser.add_argument("reference", type=Path, help="Reference genome FASTA file")
    check_parser.add_argument("bam", type=Path, nargs="?", default=None, help="Optional BAM file for test run")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare two loci sets (BED or JSON).")
    compare_parser.add_argument("bed1_or_json", type=Path, help="First loci file (BED or Stranger JSON)")
    compare_parser.add_argument("bed2_or_json", type=Path, help="Second loci file (BED or Stranger JSON)")

    add_interruptions_parser = subparsers.add_parser(
        "add-interruptions",
        help="Add interruptions present in the reference to motif lists. (Looks up loci by coordinates, "
        "i.e. pass only a reference that matches the .bed, NOT an individual's genome.)",
    )
    add_interruptions_parser.add_argument("bed", type=Path, help="Input BED file")
    add_interruptions_parser.add_argument("output_bed", type=Path, help="Output BED file path")
    add_interruptions_parser.add_argument("reference", type=Path, help="Reference genome FASTA file")

    trim_boundaries_parser = subparsers.add_parser(
        "trim-boundaries",
        help="Move region boundaries inward to remove partial motifs at the ends of the reference allele. "
        "(Looks up loci by coordinates, i.e. pass only a reference that matches the .bed, NOT an individual's genome.)",
    )
    trim_boundaries_parser.add_argument("bed", type=Path, help="Input BED file")
    trim_boundaries_parser.add_argument("output_bed", type=Path, help="Output BED file path")
    trim_boundaries_parser.add_argument("reference", type=Path, help="Reference genome FASTA file")

    sort_parser = subparsers.add_parser("sort", help="Sort an existing BED file by position or by name, keeping its comments.")
    sort_parser.add_argument("bed", type=Path, help="Input BED file")
    sort_parser.add_argument("output_bed", type=Path, help="Output BED file path")
    sort_parser.add_argument("--sort-order", choices=["position", "name"], default="position", help="How to sort the loci (default: position)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "convert":
        if args.ignore_small_differences is None:
            args.ignore_small_differences = True  # default true for convert
        regions_table = convert_stranger_json_to_regions_table(
            args.json,
            fix_composite_TRs=not args.no_fix_composite_trs,
            convert_names=not args.no_convert_names,
        )
        save_regions_table(regions_table, args.output_bed)
        logging.info("Wrote %d regions to %s", len(regions_table), args.output_bed)
        if args.reference is not None:
            check_loci_match_reference(regions_table, args.reference, ignore_small_differences=args.ignore_small_differences)
        else:
            logging.warning(
                "Because no reference genome was provided, these loci were not checked for errors. "
                "Consider running mahi-mahi-loci-tools check on them."
            )

    elif args.command == "check":
        if args.ignore_small_differences is None:
            args.ignore_small_differences = False  # default false for check
        load_and_check_loci(
            regions_file=args.bed,
            reference_file=args.reference,
            test_reads_file=args.bam,
            ignore_small_differences=args.ignore_small_differences,
        )

    elif args.command == "compare":
        regions_table1 = _load_table_from_bed_or_json(args.bed1_or_json)
        regions_table2 = _load_table_from_bed_or_json(args.bed2_or_json)
        compare_loci_sets(regions_table1, regions_table2)

    elif args.command == "add-interruptions":
        regions_table, comments = load_regions_table(args.bed)
        regions_table = check_loci_match_reference(regions_table, args.reference, ignore_small_differences=args.ignore_small_differences)
        # composites can't have multiple motifs per section, so we don't add their interruptions
        is_non_composite = regions_table.groupby("repeat_id").transform("size") == 1
        regions_table.loc[is_non_composite, "motif"] = regions_table.loc[is_non_composite, "motif_plus_interruptions"]
        save_regions_table(regions_table, args.output_bed, comments)

    elif args.command == "trim-boundaries":
        regions_table, comments = load_regions_table(args.bed)
        regions_table = trim_locus_boundaries(regions_table, args.reference)
        save_regions_table(regions_table, args.output_bed, comments)

    elif args.command == "sort":
        regions_table, comments = load_regions_table(args.bed)
        regions_table = sort_regions_table(regions_table, args.sort_order)
        save_regions_table(regions_table, args.output_bed, comments)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

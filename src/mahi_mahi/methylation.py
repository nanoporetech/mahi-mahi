"""Extract CpG methylation calls from SAM base-modification tags.

Methylation is encoded in the ``MM``/``ML`` tags described in the "Base modifications"
section of the SAM tag specification.  This module turns those tags into a compact
per-CpG state string, one character per CpG dinucleotide of a read's tandem-repeat
sequence, in 5'->3' order of the reference-oriented sequence:

``M``
    modified cytosine: combined 5mC + 5hmC probability
``U``
    unmodified cytosine is the most likely state
``?``
    no call - the basecaller provided no 5mC and/or no 5hmC information for this site
``!``
    uncertain - no state reached the configured confidence threshold

A read with no usable methylation data at all is represented by ``None`` rather than a
string of ``?``, so that "this basecaller did not call modifications" stays
distinguishable from "this basecaller declined to call these particular sites".

Only 5mC (``C+m``, equivalently ChEBI 27551) and 5hmC (``C+h``, equivalently ChEBI 76792)
at CpG dinucleotides are considered; other modification codes and non-CpG cytosines are
ignored.  Both modifications must be present for a site to be called, so reads from a
basecaller that reports only one of them have no usable methylation data.
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pysam

METHYLATED = "M"
UNMETHYLATED = "U"
NO_CALL = "?"
UNCERTAIN = "!"

DEFAULT_STATE_CONFIDENCE_THRESHOLD = 0.80

# ML stores probability as a byte, where value N means a probability in [N/256, (N+1)/256).
ML_SCALE = 256

# 5-Methylcytosine is code "m", or equivalently ChEBI 27551.
_5MC_CODES = frozenset({"m", "27551"})

# 5-Hydroxymethylcytosine is code "h", or equivalently ChEBI 76792.
_5HMC_CODES = frozenset({"h", "76792"})

# Leading part of one MM entry: base, strand, modification code(s) and the optional
# "." / "?" flag saying how unlisted bases of that type should be interpreted.
_MM_ENTRY_HEAD = re.compile(r"([ACGTUN])([-+])([A-Za-z]+|[0-9]+)([.?]?)")

#: How to treat a cytosine that the basecaller did not list.
#: ``"assume_unmodified"`` is the spec default (MM flag "." or absent);
#: ``"no_call"`` corresponds to the "?" flag.
SkipPolicy = Literal["assume_unmodified", "no_call"]


def cpg_offsets(sequence: str) -> list[int]:
    """Return the offset of the C of every CpG dinucleotide in *sequence*."""
    return [i for i in range(len(sequence) - 1) if sequence[i : i + 2] == "CG"]


def _skip_policy(mm_tag: str, wanted_codes: frozenset[str]) -> SkipPolicy | None:
    """Return how to treat unlisted cytosines, or None if *mm_tag* has no entry for *wanted_codes*."""
    for entry in mm_tag.split(";"):
        match = _MM_ENTRY_HEAD.match(entry.strip())
        if match is None:
            continue
        base, _strand, codes, flag = match.groups()
        if base != "C":
            continue
        # A numeric code is a single ChEBI id; letters are a list of single-character codes.
        listed = {codes} if codes.isdigit() else set(codes)
        if listed & wanted_codes:
            return "no_call" if flag == "?" else "assume_unmodified"
    return None


def _tags_are_stale(read: pysam.AlignedSegment) -> bool:
    """Return whether MN disagrees with the current SEQ length, invalidating MM/ML."""
    if not read.has_tag("MN"):
        return False
    sequence = read.query_sequence
    return sequence is not None and int(read.get_tag("MN")) != len(sequence)


def _modification_calls(modified_bases: Mapping[tuple[str, int, str], list[tuple[int, int]]], wanted_codes: frozenset[str]) -> dict[int, int]:
    """Return {query_sequence offset: ML probability byte} for every listed site of *wanted_codes*.

    A probability of ``-1`` means the site is listed in MM but has no ML probability; such
    sites are kept so they become no-calls rather than falling back to the skip policy.

    Offsets are into ``read.query_sequence`` (the reference-oriented sequence), which is
    the same coordinate space as the ``aln_start`` used elsewhere in the pipeline.  For a
    reverse-strand read the offsets point at the G of each CpG, because the cytosine that
    was actually measured is the one on the opposite strand.
    """
    calls: dict[int, int] = {}
    for (base, _strand, code), values in modified_bases.items():
        if base != "C" or str(code) not in wanted_codes:
            continue
        for position, probability in values:
            calls[position] = probability
    return calls


def _site_probability(calls: dict[int, int], position: int, policy: SkipPolicy) -> int | None:
    """Return the ML probability byte at *position*, or None if there is no call there."""
    probability = calls.get(position)
    if probability is None:
        return 0 if policy == "assume_unmodified" else None
    return None if probability < 0 else probability  # a negative value means listed in MM but absent from ML


def _state(methylated: int, hydroxymethylated: int, confidence_threshold: float) -> str:
    """Return a confident combined CpG state, or UNCERTAIN."""
    modified = methylated + hydroxymethylated
    unmodified = ML_SCALE - modified
    state, probability = (METHYLATED, modified) if modified >= unmodified else (UNMETHYLATED, unmodified)
    return state if probability / ML_SCALE >= confidence_threshold else UNCERTAIN


def read_cpg_states(
    read: pysam.AlignedSegment,
    subread_seq: str,
    aln_start: int,
    confidence_threshold: float = DEFAULT_STATE_CONFIDENCE_THRESHOLD,
) -> str | None:
    """Return the per-CpG state string for *subread_seq*, or None if the read has no methylation data.

    Args:
        read: The aligned segment the subread was taken from.
        subread_seq: The extracted tandem-repeat sequence, reference-oriented.
        aln_start: Offset of *subread_seq* within ``read.query_sequence``.
    """
    if not read.has_tag("MM") or _tags_are_stale(read):
        return None
    mm_tag = str(read.get_tag("MM"))
    mc_policy = _skip_policy(mm_tag, _5MC_CODES)
    hmc_policy = _skip_policy(mm_tag, _5HMC_CODES)
    # Without both, the combined modification probability cannot be determined for any site.
    if mc_policy is None or hmc_policy is None:
        return None
    modified_bases = read.modified_bases or {}
    mc_calls = _modification_calls(modified_bases, _5MC_CODES)
    hmc_calls = _modification_calls(modified_bases, _5HMC_CODES)

    states: list[str] = []
    for offset in cpg_offsets(subread_seq):
        # A forward read reports the top-strand C; a reverse read reports the bottom-strand
        # C, which appears at the G of the CpG once htslib has reference-oriented SEQ.
        position = aln_start + offset
        if position not in mc_calls and position not in hmc_calls:
            position += 1
        methylated = _site_probability(mc_calls, position, mc_policy)
        hydroxymethylated = _site_probability(hmc_calls, position, hmc_policy)
        if methylated is None or hydroxymethylated is None:
            states.append(NO_CALL)
        else:
            states.append(_state(methylated, hydroxymethylated, confidence_threshold))
    return "".join(states)


def split_cpg_states(states: str | None, sequence: str, section_lengths: Sequence[int]) -> list[str | None]:
    """Split a per-CpG state string into one string per section of *sequence*.

    Each CpG is assigned to the section containing its C, so a CpG straddling a section
    boundary is counted exactly once.  *section_lengths* must sum to ``len(sequence)``.
    """
    if states is None:
        return [None] * len(section_lengths)
    offsets = cpg_offsets(sequence)
    if len(offsets) != len(states):
        raise ValueError(f"Expected {len(offsets)} CpG states for a {len(sequence)}bp sequence, got {len(states)}")
    result: list[str | None] = []
    cursor = 0
    section_end = 0
    for length in section_lengths:
        section_end += length
        count = 0
        while cursor + count < len(offsets) and offsets[cursor + count] < section_end:
            count += 1
        result.append(states[cursor : cursor + count])
        cursor += count
    return result


def methylation_counts(states: str | None) -> tuple[int, int]:
    """Return (number of methylated sites, number of called sites) for a per-CpG state string.

    Both 5mC and 5hmC count as methylated.
    """
    if not states:
        return 0, 0
    called = len(states) - states.count(NO_CALL) - states.count(UNCERTAIN)
    return states.count(METHYLATED), called


def methylation_fractions(states: str | None) -> tuple[float, float]:
    """Return (methylated fraction, unavailable fraction) for a per-CpG state string.

    The methylated fraction counts only sites that have a call, so sites the basecaller
    explicitly declined to call or that are below the confidence threshold do not drag it
    down.  Both values are NaN when there is nothing to average over.
    """
    if not states:
        return math.nan, math.nan
    methylated, called = methylation_counts(states)
    methylated_fraction = methylated / called if called else math.nan
    return methylated_fraction, (len(states) - called) / len(states)


def bam_has_methylation(bam_file: str | Path, *, max_records: int = 10000) -> bool:
    """Check sampled alignments for 5mC/5hmC tags and warn about implicit canonical calls."""
    import pysam

    has_methylation = False
    has_implicit_calls = False
    with pysam.AlignmentFile(str(bam_file), "rb") as bam:
        for count, read in enumerate(bam.fetch(until_eof=True)):
            if count >= max_records:
                break
            if read.is_secondary or not read.has_tag("MM"):
                continue
            mm_tag = str(read.get_tag("MM"))
            policies = (_skip_policy(mm_tag, _5MC_CODES), _skip_policy(mm_tag, _5HMC_CODES))
            has_methylation |= all(policy is not None for policy in policies)
            has_implicit_calls |= "assume_unmodified" in policies
    if has_implicit_calls:
        logging.warning(
            "Sampled MM tags use implicit canonical semantics ('.' or an omitted flag). "
            "Unlisted CpGs are treated as unmodified without explicit confidence probabilities; "
            "the methylation confidence threshold cannot be applied to those observations."
        )
    return has_methylation

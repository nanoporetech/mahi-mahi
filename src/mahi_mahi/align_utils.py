import logging
import re
from collections.abc import Callable, Generator, Sequence

import parasail
from pysam import AlignedSegment

EXPANDED_FLANK_WINDOW_INDEL_ALLOWANCE = 200


def map_to_subread(
    subread: str, sequence: str, identity_threshold: float = 0.8, gap_open: int = 8, gap_extend: int = 2, match: int = 1, mismatch: int = -3
) -> tuple[int, int]:
    """
    Map a small sequence to a subread using parasail and return the start and end position indices.

    Args:
        subread (str): The subread string.
        sequence (str): The small sequence to map.
        identity_threshold (float): Minimum identity threshold for a valid mapping (default: 0.9).
        gap_open (int): Gap opening penalty (default: 10).
        gap_extend (int): Gap extension penalty (default: 2).
        match (int): Match score (default: 10).
        mismatch (int): Mismatch penalty (default: -10).

    Returns:
        tuple[int, int]:
            - (start, end) position indices of the mapping in the subread (end is exclusive).
            - Returns (-1, -1) if the sequence is not mapped.
    """
    if not sequence:
        return (0, 0)
    if not subread:
        return (-1, -1)
    # Perform global alignment using parasail
    strict_matrix = parasail.matrix_create("ACGT", match, mismatch)
    alignment = parasail.sg_dx_trace_striped_32(sequence, subread, gap_open, gap_extend, strict_matrix)
    if not alignment or alignment.score < 0:
        logging.debug("No valid alignment found.")
        return (-1, -1)
    alignment_start, cigar = parasail_to_sam(alignment, sequence)
    identity = query_identity_ignore_edge_deletions(cigar)
    logging.debug(f"Realign flank: alignment_start={alignment_start}, cigar={cigar}, identity={identity:.4f}")
    if identity < identity_threshold:
        return (-1, -1)

    # Extract the start and end positions from the alignment
    start = alignment_start
    end = alignment.end_ref + 1  # end_ref is inclusive, so add 1 for exclusive
    return (start, end)


def query_identity_ignore_edge_deletions(cigar: str) -> float:
    """
    Calculate query-based identity from CIGAR string, excluding edge deletions.

    Identity = matches / (matches + mismatches + insertions)

    Edge deletions (leading and trailing D's) are excluded, while soft clips
    contribute to the denominator so clipped bases reduce identity.

    Parameters:
        cigar (str): A CIGAR string like "10D50=2X3I10D"

    Returns:
        float: identity (0.0-1.0)
    """
    # Parse the CIGAR string; include soft clips so clipped query bases contribute to identity
    operations = re.findall(r"(\d+)([=XIDS])", cigar)

    # Remove leading deletions
    while operations and operations[0][1] == "D":
        operations.pop(0)

    # Remove trailing deletions
    while operations and operations[-1][1] == "D":
        operations.pop()

    # Count query-based components
    match_count = 0
    mismatch_count = 0
    insertion_count = 0

    for count_str, op in operations:
        count = int(count_str)
        if op == "=":
            match_count += count
        elif op == "X":
            mismatch_count += count
        elif op == "I":
            insertion_count += count
        elif op == "S":
            mismatch_count += count
        # Ignore internal deletions in the query length denominator

    query_length = match_count + mismatch_count + insertion_count
    if query_length == 0:
        return 0.0

    identity = match_count / query_length
    return identity


def parasail_to_sam(result: parasail.Result, seq: str) -> tuple[int, str]:
    """Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.

    """
    # Ensure cigstr is correctly handled based on its type
    cigstr = result.cigar.decode.decode() if isinstance(result.cigar.decode, bytes) else result.cigar.decode

    first = next(cigar_ops_from_start(cigstr))
    prefix = "".join(first)
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query
    clip = "" if cliplen == 0 else f"{cliplen}S"
    if first[1] == "I":
        pre = f"{int(first[0]) + cliplen}S"
    elif first[1] == "D":
        pre = clip
        rstart = int(first[0])
    else:
        pre = f"{clip}{prefix}"
    # if last cigar op is I, result.end_query will not consider the insertion
    # as aligned hence recalculate end_query here.
    end_query = result.end_query + 1  # end_query is inclusive
    last = next(cigar_ops_from_end(cigstr))
    if last[1] == "I":
        end_query -= int(last[0])
        cigstr = cigstr[: -len("".join(last))]
    end_clip = len(seq) - end_query
    suf = f"{end_clip}S" if end_clip > 0 else ""
    mid = cigstr[len(prefix) :]
    new_cigstr = "".join((pre, mid, suf))
    return rstart, new_cigstr


def cigar_ops_from_end(cigar: str) -> Generator[tuple[str, str], None, None]:
    """Yield cigar string operations from end of cigar (in reverse order).

    :param cigar: cigar string.

    :yields: str, op. length, str op. type
    """
    re_split_cigar_rev = re.compile(r"(?P<op>\D+)(?P<len>\d+)")
    for m in re.finditer(re_split_cigar_rev, cigar[::-1]):
        yield m.group("len")[::-1], m.group("op")[::-1]


def cigar_ops_from_start(cigar: str) -> Generator[tuple[str, str], None, None]:
    """Yield cigar string operations from start of cigar (in order).

    :param cigar: cigar string.

    :yields: str, op. length, str op. type
    """
    re_split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")
    for m in re.finditer(re_split_cigar, cigar):
        yield m.group("len"), m.group("op")


def realign_flank_to_query(read_aln_seg: AlignedSegment, query_start: int, query_end: int, flank: str) -> tuple[bool, int, int]:
    """
    Realign a flank to the query sequence.

    Args:
        read_aln_seg (AlignedSegment): The aligned read segment.
        query_start (int): Start position on the query.
        query_end (int): End position on the query.
        flank (str): The sequence to realign.

    Returns:
        Tuple[bool, int, int]:
            - success (bool): Whether the realignment was successful.
            - new_start (int): New start coordinate of the flank.
            - new_end (int): New end coordinate of the flank.
    """
    if read_aln_seg.query_sequence is None:
        return False, -1, -1

    query_len = len(read_aln_seg.query_sequence)  # type: ignore
    query_start = max(0, query_start)
    query_end = min(query_end, query_len)

    realignment_seq = read_aln_seg.query_sequence[query_start:query_end]  # type: ignore
    map_result = map_to_subread(realignment_seq, flank)
    if map_result == (-1, -1):
        return False, -1, -1

    new_start = map_result[0] + query_start
    new_end = map_result[1] + query_start
    return True, new_start, new_end


def _shrink_flank_from_left(flank: str, size: int) -> str:
    """Reduce flank length to specified size while keeping the suffix."""
    if size >= len(flank):
        return flank
    return flank[-size:]


def _shrink_flank_from_right(flank: str, size: int) -> str:
    """Reduce flank length to specified size while keeping the prefix."""
    if size >= len(flank):
        return flank
    return flank[:size]


def _realign_with_shrinking_flank(
    read_aln_seg: AlignedSegment,
    window: tuple[int, int],
    flank_seq: str,
    min_flank_length: int,
    shrink_fn: Callable[[str, int], str],
    extension: int,
    side_read_length: int,
    expanded_window: tuple[int, int] | None = None,
    expanded_first: bool = False,
) -> tuple[bool, int, int]:
    """
    Attempt to realign a flank sequence, shrinking it iteratively and trying one longer flank.

    When ``expanded_first`` is true, the longer flank is tried before the
    existing extension-length and shrinking attempts. This is used only for
    the wide-window fallback, where the longer flank can help disambiguate a
    realignment without changing the normal local search order.
    """
    attempts: list[tuple[str, tuple[int, int]]] = []
    expanded_size = min(len(flank_seq), side_read_length)
    expanded_attempt: tuple[str, tuple[int, int]] | None = None
    if expanded_size > extension:
        expanded_flank = shrink_fn(flank_seq, expanded_size)
        expanded_attempt = (expanded_flank, expanded_window or window)
    if expanded_first and expanded_attempt is not None:
        attempts.append(expanded_attempt)

    current_flank = shrink_fn(flank_seq, min(extension, len(flank_seq)))
    while len(current_flank) >= min_flank_length:
        attempts.append((current_flank, window))
        if len(current_flank) == min_flank_length:
            break
        # Shrink the flank by half each iteration
        new_size = len(current_flank) // 2
        new_size = max(new_size, min_flank_length)
        current_flank = shrink_fn(current_flank, new_size)

    if not expanded_first and expanded_attempt is not None:
        attempts.append(expanded_attempt)

    for flank, (window_start, window_end) in attempts:
        logging.debug(f"Aligning flank of length {len(flank)} to read {read_aln_seg.query_name}")
        mapped, start_realigned, end_realigned = realign_flank_to_query(read_aln_seg, window_start, window_end, flank)
        if mapped:
            return True, start_realigned, end_realigned

    return False, -1, -1


def compute_boundary(
    read_aln_seg: AlignedSegment,
    alignment_pairs: list[tuple[int, int]],
    boundary: int,
    extension: int,
    flank_seq_this_side: str,
    min_flank_length: int,
    read_len: int,
    of_left_side: bool,
) -> tuple[int, bool, bool, int | None]:
    """
    Determine one (left or right) boundary of the aligned subread, attempting re-alignment when needed.
    """
    if of_left_side:
        boundary_cons_pos, boundary_ref_pos = alignment_pairs[0]
        aln_boundary = 0
        search_interval1 = (boundary_cons_pos - 2 * extension, boundary_cons_pos + 1)
        search_interval2 = (0, boundary_cons_pos + extension)
        _shrink_flank = _shrink_flank_from_left
        original_boundary = boundary_cons_pos
        side_read_length = boundary_cons_pos
    else:
        boundary_cons_pos, boundary_ref_pos = alignment_pairs[-1]
        aln_boundary = read_len
        search_interval1 = (boundary_cons_pos, boundary_cons_pos + 2 * extension)
        search_interval2 = (boundary_cons_pos - extension, read_len)
        _shrink_flank = _shrink_flank_from_right
        original_boundary = boundary_cons_pos + 1
        side_read_length = read_len - original_boundary
    expanded_flank_size = max(0, min(len(flank_seq_this_side), side_read_length))
    if of_left_side:
        expanded_search_interval1 = (
            max(0, boundary_cons_pos - expanded_flank_size - EXPANDED_FLANK_WINDOW_INDEL_ALLOWANCE),
            boundary_cons_pos + 1,
        )
    else:
        expanded_search_interval1 = (
            boundary_cons_pos,
            min(read_len, boundary_cons_pos + expanded_flank_size + EXPANDED_FLANK_WINDOW_INDEL_ALLOWANCE + 1),
        )
    is_spanning_this_side = False
    this_side_realigned = False
    mapped = False
    realignment_shift: int | None = None

    if boundary_ref_pos == boundary:
        is_spanning_this_side = True
        mapped, start_realigned_pos, end_realigned_pos = _realign_with_shrinking_flank(
            read_aln_seg,
            search_interval1,
            flank_seq_this_side,
            min_flank_length,
            _shrink_flank,
            extension,
            side_read_length,
            expanded_window=expanded_search_interval1,
        )
        this_side_realigned_pos = end_realigned_pos if of_left_side else start_realigned_pos
        if mapped and this_side_realigned_pos is not None:
            aln_boundary = this_side_realigned_pos
            realignment_shift = aln_boundary - original_boundary
            if aln_boundary != original_boundary:
                logging.debug(
                    f"Read {read_aln_seg.query_name} spanning"
                    f" {'left' if of_left_side else 'right'} end realigned"
                    f" from {original_boundary} to {aln_boundary}"
                )
            this_side_realigned = True

    if not mapped:
        mapped, start_realigned_pos, end_realigned_pos = _realign_with_shrinking_flank(
            read_aln_seg,
            search_interval2,
            flank_seq_this_side,
            min_flank_length,
            _shrink_flank,
            extension,
            side_read_length,
            expanded_first=True,
        )
        this_side_realigned_pos = end_realigned_pos if of_left_side else start_realigned_pos
        is_spanning_this_side = mapped
        if mapped and this_side_realigned_pos is not None:
            aln_boundary = this_side_realigned_pos
            realignment_shift = aln_boundary - original_boundary
            this_side_realigned = True

    logging.debug(
        f"Computed {'left' if of_left_side else 'right'} boundary:"
        f" aln_boundary={aln_boundary},"
        f" realignment_shift={realignment_shift},"
        f" is_spanning_this_side={is_spanning_this_side},"
        f" this_side_realigned={this_side_realigned}, mapped ={mapped}"
    )
    return aln_boundary, is_spanning_this_side, this_side_realigned, realignment_shift


def levenshtein(a: str, b: str) -> float:
    n, m = len(a), len(b)
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prow = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                prow[j] + 1,  # delete
                row[j - 1] + 1,  # insert
                prow[j - 1] + (ai != b[j - 1]),  # subst/match
            )
    return float(dp[n][m])


def _cyclic_shifts(motif: str) -> list[str]:
    return [motif[i:] + motif[:i] for i in range(len(motif))]


def rotation_aware_levenshtein(observed_motif: str, expected_motif: str) -> float:
    return min(levenshtein(shift, expected_motif) for shift in _cyclic_shifts(observed_motif))


def _motif_distance(observed_motif: str, expected_motif: str, *, rotation_aware_distance: bool) -> float:
    if rotation_aware_distance:
        return rotation_aware_levenshtein(observed_motif, expected_motif)
    return levenshtein(observed_motif, expected_motif)


def _section_distance(observed_motif: str, expected_motifs: Sequence[str], *, rotation_aware_distance: bool) -> float:
    return min(_motif_distance(observed_motif, expected_motif, rotation_aware_distance=rotation_aware_distance) for expected_motif in expected_motifs)


def _known_other_motif_matches(observed_motif: str, expected_motif: str, *, rotation_aware: bool) -> bool:
    """Return whether a read motif matches a motif expected in a different section."""
    if observed_motif == expected_motif:
        return True
    if not rotation_aware or len(observed_motif) != len(expected_motif):
        return False
    return expected_motif in _cyclic_shifts(observed_motif)


def _known_other_section_matches(observed_motif: str, expected_motifs: Sequence[str], *, rotation_aware: bool) -> bool:
    """Return whether a read motif matches any motif expected in a different section."""
    return any(_known_other_motif_matches(observed_motif, expected_motif, rotation_aware=rotation_aware) for expected_motif in expected_motifs)


def _scored_distances_for_motif(
    read_motif: str,
    pattern: Sequence[Sequence[str]],
    *,
    rotation_aware_distance: bool,
    main_idx: int | None,
    non_main_penalty_factor: float,
    known_other_motif_penalty_factor: float,
    known_other_motif_rotation_aware: bool,
) -> list[float]:
    """Return per-section distances for one observed motif after applying scoring penalties."""
    base_distances = [_section_distance(read_motif, section_motifs, rotation_aware_distance=rotation_aware_distance) for section_motifs in pattern]
    scored_distances: list[float] = []
    for j, distance in enumerate(base_distances):
        if distance > 0:
            if main_idx is not None and j != main_idx:
                distance *= non_main_penalty_factor
            if any(
                _known_other_section_matches(read_motif, pattern[k], rotation_aware=known_other_motif_rotation_aware)
                for k in range(len(pattern))
                if k != j
            ):
                distance *= known_other_motif_penalty_factor
        scored_distances.append(distance)
    return scored_distances


def _build_edit_distance_matrix(
    pattern: Sequence[Sequence[str]],
    motifs: list[str],
    *,
    rotation_aware_distance: bool,
    main_idx: int | None,
    non_main_penalty_factor: float,
    known_other_motif_penalty_factor: float,
    known_other_motif_rotation_aware: bool,
) -> list[list[float]]:
    # Per-motif cost matrix (edit distance).
    return [
        _scored_distances_for_motif(
            read_motif,
            pattern,
            rotation_aware_distance=rotation_aware_distance,
            main_idx=main_idx,
            non_main_penalty_factor=non_main_penalty_factor,
            known_other_motif_penalty_factor=known_other_motif_penalty_factor,
            known_other_motif_rotation_aware=known_other_motif_rotation_aware,
        )
        for read_motif in motifs
    ]


def _build_prefix_sums(edit_distances: list[list[float]], section_count: int, read_motif_count: int) -> list[list[float]]:
    """Build per-section prefix sums so contiguous segment costs can be queried in O(1)."""
    # Prefix sums per section.
    pref = [[0.0] * (read_motif_count + 1) for _ in range(section_count)]
    for j in range(section_count):
        acc = 0.0
        for i in range(1, read_motif_count + 1):
            acc += edit_distances[i - 1][j]
            pref[j][i] = acc
    return pref


def _run_align_all_star_dp(pref: list[list[float]], read_motif_count: int, section_count: int) -> tuple[list[list[float]], list[list[int]]]:
    """Run the contiguous-segment DP used by ``align_all_star``.

    Parameters:
        pref: Prefix sums of per-motif costs, indexed by section then read motif count.
        read_motif_count: Number of observed motifs in the read decomposition.
        section_count: Number of expected sections / output segments.

    Returns:
        tuple[list[list[float]], list[list[int]]]:
            - dp[i][j] is the minimum cost to assign the first ``i`` read motifs to the first ``j`` sections.
            - cut[i][j] stores the start index of the last segment chosen for that optimum, so assignments can be reconstructed.

    The inner loop maintains a running minimum of ``dp[t][j - 1] - pref[j - 1][t]`` so each column can be filled in
    linear time instead of trying every possible previous cut position for every ``i``.
    """
    inf = float("inf")
    # DP over segments (each section takes one contiguous segment, possibly empty).
    dp = [[inf] * (section_count + 1) for _ in range(read_motif_count + 1)]
    cut = [[-1] * (section_count + 1) for _ in range(read_motif_count + 1)]

    dp[0][0] = 0.0
    for j in range(1, section_count + 1):
        dp[0][j] = 0.0
        best_val = inf
        best_t = -1
        for i in range(read_motif_count + 1):
            # Option 1: skip section j (empty).
            best_cost = dp[i][j - 1]
            motif_idx = i

            # Option 2: take non-empty segment S[t..i-1].
            candidate_cost = pref[j - 1][i] + best_val  # would have + open_cost[j - 1] if we were using gap opening costs
            if i > 0 and candidate_cost < best_cost:
                best_cost = candidate_cost
                motif_idx = best_t

            dp[i][j] = best_cost
            cut[i][j] = motif_idx

            # Update running minimum.
            val = dp[i][j - 1] - pref[j - 1][i]
            if val < best_val:
                best_val, best_t = val, i
    return dp, cut


def _reconstruct_segment_indices(cut: list[list[int]], read_motif_count: int, section_count: int) -> list[list[int]]:
    """Backtrack the DP cut table into per-section lists of read-motif indices."""
    # Reconstruct index segments.
    assigns_ext: list[list[int]] = [[] for _ in range(section_count)]
    i = read_motif_count
    for j in range(section_count, 0, -1):
        t = cut[i][j]
        assigns_ext[j - 1] = list(range(t, i))
        i = t
    assert i == 0
    return assigns_ext


def align_all_star(
    pattern: Sequence[Sequence[str]],
    motifs: list[str],
    *,
    rotation_aware_distance: bool = True,
    main_idx: int | None = None,
    non_main_penalty_factor: float = 1.5,
    known_other_motif_penalty_factor: float = 2.0,
    known_other_motif_rotation_aware: bool = False,
) -> tuple[float, list[list[str]]]:
    """
    Split a read segment (given as a motif decomposition) from multiple adjacent TRs into a segment corresponding to each TR.

    This will always assign every base of the segment to one of the TRs, even if it's clearly not a good match
    for any of them.  To check for this, use the total_cost output.

    pattern: Expected TR sections, as a sequence of motif groups. Sections must be in the order given.
    motifs: Motif decomposition of the read segment.
    rotation_aware_distance: Whether motif-to-segment comparisons treat cyclic rotations as equivalent.
    known_other_motif_rotation_aware: Controls whether the "other known motif" penalty treats cyclic rotations as matches.

    Returns:
      total_cost,
      segments: List[List[str]]  (one list of motifs per section),
    """
    if non_main_penalty_factor < 1:
        raise ValueError(f"non_main_penalty_factor must be >= 1, got {non_main_penalty_factor}")
    if known_other_motif_penalty_factor < 1:
        raise ValueError(f"known_other_motif_penalty_factor must be >= 1, got {known_other_motif_penalty_factor}")
    assert all(motif for motif in motifs), f"align_all_star requires non-empty observed motifs, got {motifs}"
    read_motif_count = len(motifs)
    section_count = len(pattern)

    if any(len(section_motifs) == 0 for section_motifs in pattern):
        raise ValueError(f"align_all_star requires at least one motif per section, got {pattern}")
    if main_idx is not None and not 0 <= main_idx < section_count:
        raise ValueError(f"main_idx must refer to a valid section index, got {main_idx}")

    edit_distances = _build_edit_distance_matrix(
        pattern,
        motifs,
        rotation_aware_distance=rotation_aware_distance,
        main_idx=main_idx,
        non_main_penalty_factor=non_main_penalty_factor,
        known_other_motif_penalty_factor=known_other_motif_penalty_factor,
        known_other_motif_rotation_aware=known_other_motif_rotation_aware,
    )
    pref = _build_prefix_sums(edit_distances, section_count, read_motif_count)
    dp, cut = _run_align_all_star_dp(pref, read_motif_count, section_count)
    assigns_ext = _reconstruct_segment_indices(cut, read_motif_count, section_count)
    # Format as lists of motifs, one inner list per section.
    segments = [[motifs[i] for i in seg] for seg in assigns_ext]

    return dp[read_motif_count][section_count], segments


def segment_exact_match_fraction(expected_motifs: Sequence[str], segment: list[str]) -> float:
    """Return the fraction of segment motifs that exactly match one of the expected motifs."""
    if not segment:
        return 1.0
    expected_motif_set = set(expected_motifs)
    return sum(motif in expected_motif_set for motif in segment) / len(segment)

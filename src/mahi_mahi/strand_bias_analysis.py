#!/usr/bin/env python
"""
Strand Bias Analysis Module

This module provides statistical functions for analyzing strand bias in k-mer counts,
particularly useful for detecting sequencing errors or systematic biases in strand-specific data.

The module implements:
1. G-test for independence between k-mer counts and strand orientation
2. Identification of error-signature k-mers using empirical Bayes shrinkage and multiple testing correction
"""

import logging
import math
import random
from typing import Any


def _collect_and_filter_kmers(plus_counts: dict[str, int], minus_counts: dict[str, int], keep_mass: float) -> list[tuple[str, int, int, int]]:
    """Collect k-mers and filter by mass threshold."""
    cols = []
    for k in set(plus_counts) | set(minus_counts):
        ap = plus_counts.get(k, 0)  # plus strand count
        am = minus_counts.get(k, 0)  # minus strand count
        tot = ap + am
        if tot > 0:
            cols.append((k, ap, am, tot))

    # Sort by total count (descending) and keep top k-mers until mass threshold
    cols.sort(key=lambda x: x[3], reverse=True)
    grand_tot = sum(t for _, _, _, t in cols)

    keep = []
    acc = 0
    for k, ap, am, t in cols:
        if acc / max(1, grand_tot) < keep_mass:
            keep.append((k, ap, am, t))
            acc += t
        else:
            break

    # Aggregate remaining k-mers into "__OTHER__" category
    rest_ap = sum(ap for _, ap, _, _ in cols[len(keep) :])
    rest_am = sum(am for _, _, am, _ in cols[len(keep) :])
    if rest_ap + rest_am > 0:
        keep.append(("__OTHER__", rest_ap, rest_am, rest_ap + rest_am))

    return keep


def _compute_g_statistic(keep: list[tuple[str, int, int, int]]) -> tuple[float, int]:
    """Compute G-statistic and degrees of freedom."""
    r1 = sum(ap for _, ap, _, _ in keep)  # plus strand total
    r2 = sum(am for _, _, am, _ in keep)  # minus strand total
    n = r1 + r2  # grand total

    G = 0.0
    for _, ap, am, t in keep:
        # Expected counts under independence hypothesis
        e_plus = t * (r1 / n) if n else 0
        e_minus = t * (r2 / n) if n else 0

        # Add contributions to G-statistic (with log-likelihood terms)
        if ap > 0 and e_plus > 0:
            G += 2 * ap * math.log(ap / e_plus)
        if am > 0 and e_minus > 0:
            G += 2 * am * math.log(am / e_minus)

    # Degrees of freedom: (rows-1) * (cols-1) = (2-1) * (M-1) = M-1
    df = max(1, len(keep) - 1)
    return G, df


def _monte_carlo_pvalue(keep: list[tuple[str, int, int, int]], G: float, n_perm: int, seed: int) -> float:
    """Compute Monte Carlo p-value."""
    rnd = random.Random(seed)
    r1 = sum(ap for _, ap, _, _ in keep)  # plus strand total
    r2 = sum(am for _, _, am, _ in keep)  # minus strand total
    n = r1 + r2

    if n == 0:
        return 1.0

    p_plus = r1 / n  # overall plus strand fraction
    more_extreme = 0

    for _ in range(n_perm):
        G_sim = 0.0
        for _, _ap, _am, t in keep:
            # Simulate under null: each k-mer count follows binomial distribution
            ap_sim = sum(1 for _ in range(t) if rnd.random() < p_plus)
            am_sim = t - ap_sim

            # Expected counts (same as observed data)
            e_p = t * p_plus
            e_m = t * (1 - p_plus)

            # Compute G-statistic for simulated data
            if ap_sim > 0 and e_p > 0:
                G_sim += 2 * ap_sim * math.log(ap_sim / e_p)
            if am_sim > 0 and e_m > 0:
                G_sim += 2 * am_sim * math.log(am_sim / e_m)

        if G_sim >= G:
            more_extreme += 1

    # Add-one smoothing to avoid p=0
    return (more_extreme + 1) / (n_perm + 1)


def _compute_kmer_contributions(keep: list[tuple[str, int, int, int]]) -> dict[str, float]:
    """Compute per-k-mer contributions to G-statistic."""
    r1 = sum(ap for _, ap, _, _ in keep)  # plus strand total
    r2 = sum(am for _, _, am, _ in keep)  # minus strand total
    n = r1 + r2

    contrib = {}
    for k, ap, am, t in keep:
        e_plus = t * (r1 / n) if n else 0
        e_minus = t * (r2 / n) if n else 0
        c = 0.0
        if ap > 0 and e_plus > 0:
            c += 2 * ap * math.log(ap / e_plus)
        if am > 0 and e_minus > 0:
            c += 2 * am * math.log(am / e_minus)
        contrib[k] = c

    return contrib


def g_test_region(
    plus_counts: dict[str, int], minus_counts: dict[str, int], keep_mass: float = 0.99, n_perm: int = 0, seed: int = 1
) -> dict[str, Any]:
    """
    Perform a G-test (likelihood ratio test) for independence between k-mer counts and strand orientation.

    This function tests whether k-mer distributions differ significantly between plus and minus strands,
    which can indicate strand-specific sequencing errors or biological biases. The test uses a
    contingency table approach where each k-mer forms a column and strand orientation forms rows.

    To handle sparse data, the function:
    1. Keeps only the most abundant k-mers (by total count) until reaching keep_mass fraction
    2. Aggregates remaining k-mers into an "__OTHER__" category
    3. Computes the G-statistic for the resulting contingency table
    4. Optionally performs Monte Carlo permutation testing for p-value estimation

    Args:
        plus_counts: Dictionary mapping k-mers to their counts on the plus strand
        minus_counts: Dictionary mapping k-mers to their counts on the minus strand
        keep_mass: Fraction of total mass to retain (default: 0.99). K-mers are sorted by
                  total count and top k-mers are kept until cumulative mass >= keep_mass
        n_perm: Number of permutations for Monte Carlo p-value (default: 0, no MC test)
        seed: Random seed for reproducible Monte Carlo testing (default: 1)

    Returns:
        Dictionary containing:
        - 'G': G-statistic value (higher values indicate stronger association)
        - 'df': Degrees of freedom (number of k-mers - 1)
        - 'p_mc': Monte Carlo p-value (None if n_perm=0)
        - 'contrib': Per-k-mer contributions to G-statistic (for identifying key drivers)
        - 'table_cols': List of k-mers included in the final contingency table

    Mathematical Details:
        G = 2 * Σ(O_ij * ln(O_ij / E_ij))
        where O_ij is observed count, E_ij is expected count under independence

    Example:
        >>> plus = {'AAA': 100, 'TTT': 50, 'GGG': 10}
        >>> minus = {'AAA': 20, 'TTT': 80, 'GGG': 5}
        >>> result = g_test_region(plus, minus, n_perm=1000)
        >>> print(f"G-statistic: {result['G']:.3f}, p-value: {result['p_mc']:.3f}")
    """
    # Step 1: Collect and filter k-mers
    keep = _collect_and_filter_kmers(plus_counts, minus_counts, keep_mass)

    # Step 2: Compute G-statistic
    G, df = _compute_g_statistic(keep)

    # Step 3: Optional Monte Carlo p-value estimation
    p_mc = None
    if n_perm > 0:
        p_mc = _monte_carlo_pvalue(keep, G, n_perm, seed)

    # Step 4: Compute per-k-mer contributions
    contrib = _compute_kmer_contributions(keep)

    return {"G": G, "df": df, "p_mc": p_mc, "contrib": contrib, "table_cols": [k for k, _, _, _ in keep]}


def _passes_gates_with_reason(
    a: int,
    c: int,
    Np: int,
    Nm: int,
    N: int,
    bg: dict[str, float],
    k: str,
    min_delta: float,
    min_or: float,
    min_enriched_frac: float,
    max_other_frac: float,
    min_global_frac: float,
    shrink_alpha: float,
) -> tuple[bool, str]:
    """Check if a k-mer passes all filtering gates and return reason if it fails."""
    # Gate 1: Global rarity filter (use raw frequencies)
    global_freq = (a + c) / N
    if global_freq < min_global_frac:
        return False, f"global_rarity (freq={global_freq:.6f} < {min_global_frac})"

    # Gate 2: Empirical Bayes shrinkage for strand fractions
    p0 = bg[k]  # Global background frequency
    p_plus_shr = (a + shrink_alpha * p0) / (Np + shrink_alpha)
    p_minus_shr = (c + shrink_alpha * p0) / (Nm + shrink_alpha)
    delta = p_plus_shr - p_minus_shr
    if abs(delta) < min_delta:  # Not enough strand bias after shrinkage
        return False, f"delta (delta={delta:.6f} < {min_delta})"

    # Gate 3: Prevalence filters on shrunk fractions
    enriched_frac = p_plus_shr if delta > 0 else p_minus_shr
    other_frac = p_minus_shr if delta > 0 else p_plus_shr

    if enriched_frac < min_enriched_frac:  # Enriched strand too rare
        return False, f"prevalence_enriched (frac={enriched_frac:.6f} < {min_enriched_frac})"
    if other_frac > max_other_frac:  # Other strand too common
        return False, f"prevalence_other (frac={other_frac:.6f} > {max_other_frac})"

    # Gate 4: Odds ratio filter (with Haldane-Anscombe correction)
    b = Np - a  # other k-mers on plus strand
    d = Nm - c  # other k-mers on minus strand
    OR = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    if not (min_or <= OR or 1 / min_or >= OR):
        return False, f"odds_ratio (OR={OR:.3f}, not >= {min_or} or <= {1 / min_or:.3f})"

    return True, "passed"


def _logchoose(n: int, k: int) -> float:
    """Compute log of binomial coefficient C(n,k) using lgamma for numerical stability."""
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_logpmf(a: int, r1: int, c1: int, n: int) -> float:
    """Compute log probability mass function of hypergeometric distribution."""
    return _logchoose(c1, a) + _logchoose(n - c1, r1 - a) - _logchoose(n, r1)


def _fisher_one_sided_midp(a: int, b: int, c: int, d: int, direction: str = "ge") -> float:
    """
    Compute one-sided Fisher's exact test p-value with mid-p correction.

    Mid-p correction: p = (tail probability) - 0.5 * P(X = observed)
    This reduces conservatism of Fisher's exact test.
    """
    r1, c1, n = a + b, a + c, a + b + c + d
    lo = max(0, r1 - (n - c1))
    hi = min(r1, c1)

    # Compute all hypergeometric probabilities
    logpmfs = {x: _hypergeom_logpmf(x, r1, c1, n) for x in range(lo, hi + 1)}
    pmfs = {x: math.exp(lp) for x, lp in logpmfs.items()}

    p_obs = pmfs.get(a, 0.0)  # Probability of observed value

    x_limits = (a, hi + 1) if direction == "ge" else (lo, a + 1)
    tail = sum(pmfs[x] for x in range(*x_limits))

    # Mid-p correction: subtract half the point probability
    return max(0.0, min(1.0, tail - 0.5 * p_obs))


def _fisher_two_sided_midp(a: int, b: int, c: int, d: int) -> float:
    """
    Compute two-sided Fisher's exact test p-value with mid-p correction.

    Two-sided p-value = 2 * min(left tail, right tail)
    """
    pr = _fisher_one_sided_midp(a, b, c, d, "ge")
    pl = _fisher_one_sided_midp(a, b, c, d, "le")
    return min(1.0, 2.0 * min(pl, pr))


def _bh_fdr(pvals: list[float]) -> list[float]:
    """
    Apply Benjamini-Hochberg false discovery rate correction.

    For m tests with p-values p(1) <= p(2) <= ... <= p(m):
    q(i) = min(q(i+1), p(i) * m / i)
    where q(m) = p(m)
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    running = 1.0

    # Work backwards from highest p-value
    for rank, _i in enumerate(reversed(order), start=1):
        j = m - rank
        val = pvals[order[j]] * m / (j + 1)
        running = min(running, val)  # Monotonicity constraint
        q[order[j]] = running

    return q


def error_kmer_sets_strict(
    plus_counts: dict[str, int],
    minus_counts: dict[str, int],
    *,
    # Effect size gates
    min_delta: float = 0.0,
    min_or: float = 5.0,
    # Prevalence gates (on shrunk fractions)
    min_enriched_frac: float = 0.0,
    max_other_frac: float = 0.1,
    # Global rarity gate (on raw fractions)
    min_global_frac: float = 0.002,
    # Empirical Bayes shrinkage parameter
    shrink_alpha: float = 100.0,
    # Multiple testing correction
    fdr: float = 0.005,
) -> tuple[set[str], set[str]]:
    """
    Identify strand-asymmetric error-signature k-mers using rigorous statistical filtering.

    This function implements a comprehensive pipeline for detecting k-mers that show
    systematic strand bias, which often indicates sequencing errors or technical artifacts.
    The method combines:

    1. Empirical Bayes shrinkage to stabilize estimates for low-count k-mers
    2. Multiple filtering gates based on effect size, prevalence, and global frequency
    3. Fisher's exact test with mid-p correction for significance testing
    4. Benjamini-Hochberg FDR correction for multiple testing

    Args:
        plus_counts: K-mer counts on plus strand
        minus_counts: K-mer counts on minus strand
        min_delta: Minimum absolute difference in strand fractions after shrinkage (default: 0 = no delta filter)
        min_or: Minimum odds ratio for association (or 1/min_or for negative association) (default: 5.0)
        min_enriched_frac: Minimum fraction on enriched strand after shrinkage (default: 0 = no filter)
        max_other_frac: Maximum fraction on depleted strand after shrinkage (default: 0.1 = 10%)
        min_global_frac: Minimum global frequency to avoid rare k-mers (default: 0.002 = 0.2%)
        shrink_alpha: Empirical Bayes shrinkage parameter, higher = more shrinkage (default: 100.0)
        fdr: False discovery rate threshold for multiple testing (default: 0.005 = 0.5%)

    Returns:
        Tuple of (E_plus, E_minus) where:
        - E_plus: Set of k-mers significantly enriched on plus strand
        - E_minus: Set of k-mers significantly enriched on minus strand

    Statistical Method:
        For each k-mer, constructs 2x2 contingency table:
                    k-mer    other
        plus:         a        b
        minus:        c        d

        Uses Fisher's exact test with mid-p correction to test independence,
        then applies Benjamini-Hochberg correction across all k-mers.

    Empirical Bayes Shrinkage:
        Shrunk fraction = (observed + alpha * prior) / (total + alpha)
        where prior is the global frequency of the k-mer across both strands.
        This stabilizes estimates for low-count k-mers.

    Example:
        >>> plus = {'AAA': 100, 'TTT': 10, 'GGG': 200}
        >>> minus = {'AAA': 20, 'TTT': 150, 'GGG': 180}
        >>> e_plus, e_minus = error_kmer_sets_strict(plus, minus)
        >>> print(f"Plus-enriched: {e_plus}")
        >>> print(f"Minus-enriched: {e_minus}")
    """

    # Main analysis starts here
    Np = sum(plus_counts.values())  # Total plus strand counts
    Nm = sum(minus_counts.values())  # Total minus strand counts

    logging.debug(f"error_kmer_sets_strict: Plus strand total={Np}, Minus strand total={Nm}")

    # Handle edge cases: no data on one or both strands
    if Np == 0 or Nm == 0:
        logging.debug("error_kmer_sets_strict: No data on one or both strands, returning empty sets")
        return set(), set()

    N = Np + Nm  # Grand total
    logging.debug(f"error_kmer_sets_strict: Grand total N={N}")

    # Compute background frequency p0(k) from both strands combined
    union = set(plus_counts) | set(minus_counts)
    bg = {k: (plus_counts.get(k, 0) + minus_counts.get(k, 0)) / N for k in union}

    logging.debug(f"error_kmer_sets_strict: Found {len(union)} unique k-mers to analyze")
    logging.debug(
        f"error_kmer_sets_strict: Filter parameters - min_delta={min_delta}, min_or={min_or}, "
        f"min_enriched_frac={min_enriched_frac}, max_other_frac={max_other_frac}, "
        f"min_global_frac={min_global_frac}, shrink_alpha={shrink_alpha}, fdr={fdr}"
    )

    # Apply filtering gates and collect candidates
    cand = []
    pvals = []
    filtered_counts = {"global_rarity": 0, "prevalence": 0, "odds_ratio": 0}

    for k in union:
        # Set up 2x2 contingency table
        a = plus_counts.get(k, 0)  # k-mer on plus strand
        c = minus_counts.get(k, 0)  # k-mer on minus strand

        logging.debug(f"error_kmer_sets_strict: Analyzing k-mer '{k}' - plus={a}, minus={c}, total={a + c}")

        # Apply all filtering gates
        passes, reason = _passes_gates_with_reason(
            a, c, Np, Nm, N, bg, k, min_delta, min_or, min_enriched_frac, max_other_frac, min_global_frac, shrink_alpha
        )
        if not passes:
            logging.debug(f"error_kmer_sets_strict: K-mer '{k}' filtered out - {reason}")
            filtered_counts[reason] = filtered_counts.get(reason, 0) + 1
            continue

        # Calculate delta for strand assignment
        p0 = bg[k]
        p_plus_shr = (a + shrink_alpha * p0) / (Np + shrink_alpha)
        p_minus_shr = (c + shrink_alpha * p0) / (Nm + shrink_alpha)
        delta = p_plus_shr - p_minus_shr

        # Statistical significance test
        b = Np - a  # other k-mers on plus strand
        d = Nm - c  # other k-mers on minus strand
        p = _fisher_two_sided_midp(a, b, c, d)
        cand.append((k, delta, p))
        pvals.append(p)

        logging.debug(f"error_kmer_sets_strict: K-mer '{k}' passed filters - delta={delta:.4f}, p_value={p:.6f}")

    logging.debug(f"error_kmer_sets_strict: Filtering summary - {dict(filtered_counts)}")
    logging.debug(f"error_kmer_sets_strict: {len(cand)} k-mers passed all filters and will be tested")

    if not cand:
        logging.debug("error_kmer_sets_strict: No k-mers passed filters, returning empty sets")
        return set(), set()

    # Apply Benjamini-Hochberg FDR correction
    qvals = _bh_fdr(pvals)
    logging.debug(f"error_kmer_sets_strict: Applied BH-FDR correction, q-values range: {min(qvals):.6f} - {max(qvals):.6f}")

    # Collect significant k-mers by strand enrichment
    E_plus, E_minus = set(), set()
    significant_count = 0
    for (k, delta, p), q in zip(cand, qvals, strict=True):
        if q <= fdr:
            significant_count += 1
            if delta > 0:
                E_plus.add(k)  # Enriched on plus strand
                logging.debug(f"error_kmer_sets_strict: K-mer '{k}' significant plus-enriched (delta={delta:.4f}, p={p:.6f}, q={q:.6f})")
            else:
                E_minus.add(k)  # Enriched on minus strand
                logging.debug(f"error_kmer_sets_strict: K-mer '{k}' significant minus-enriched (delta={delta:.4f}, p={p:.6f}, q={q:.6f})")

    logging.debug(f"error_kmer_sets_strict: Found {significant_count} significant k-mers: {len(E_plus)} plus-enriched, {len(E_minus)} minus-enriched")

    return E_plus, E_minus

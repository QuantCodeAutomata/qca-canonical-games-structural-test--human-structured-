# Statistical metrics for behavioral game theory structural analysis
# Custom — Context7 found no library equivalent for all combined metrics (paper methodology)

import numpy as np
from scipy import stats


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two independent groups.

    Uses the pooled standard deviation formula appropriate for equal or unequal
    sample sizes:  d = (mean1 - mean2) / pooled_std.

    Args:
        group1: Observations for the first group.
        group2: Observations for the second group.

    Returns:
        Cohen's d, or 0.0 when the pooled standard deviation is zero (i.e.
        both groups have no variance).
    """
    n1, n2 = len(group1), len(group2)

    # Weighted pooled variance using Bessel-corrected per-group variances
    pooled_var = ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (
        n1 + n2 - 2
    )
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_cv(values: np.ndarray) -> float:
    """Compute the coefficient of variation (CV = std / |mean|).

    CV quantifies relative dispersion; a high CV near or above 0.2 indicates
    the agent's deltas are highly variable relative to their magnitude, which
    is used as a proxy for unstructured or near-benchmark behaviour.

    Args:
        values: Input array of numeric observations.

    Returns:
        CV as a positive float, or ``np.inf`` when the mean is exactly zero.
    """
    mean = np.mean(values)
    if mean == 0:
        return np.inf

    return float(np.std(values, ddof=1) / abs(mean))


def compute_skewness(values: np.ndarray) -> float:
    """Compute the skewness of a distribution using scipy.stats.skew.

    Skewness captures asymmetry in the delta distribution, which is a
    hallmark of human-shaped decision biases (loss aversion, framing effects).

    Args:
        values: Input array of numeric observations.

    Returns:
        Skewness value (negative = left-skewed, positive = right-skewed).
    """
    # Using scipy.stats.skew — Context7 confirmed
    return float(stats.skew(values))


def standardize_delta(
    delta: np.ndarray, action_min: float, action_max: float
) -> np.ndarray:
    """Standardize delta by the full action range.

    Paper's pre-registered scaling rule:
        standardized_delta = delta / (action_max - action_min)

    Dividing by the action range places all deltas on a [−1, 1] scale,
    making effect sizes comparable across games with different payoff ranges.

    Args:
        delta: Raw delta array (agent action minus human benchmark).
        action_min: Minimum value in the action space for this game.
        action_max: Maximum value in the action space for this game.

    Returns:
        Standardized delta array of the same shape as *delta*.
    """
    return delta / (action_max - action_min)


def compute_effect_size_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap percentile confidence interval for Cohen's d.

    Resamples each group independently with replacement *n_bootstrap* times,
    recomputes Cohen's d on each resample, and returns the percentile-based CI.

    Args:
        group1: Observations for the first group.
        group2: Observations for the second group.
        n_bootstrap: Number of bootstrap iterations (default 1000).
        alpha: Two-tailed significance level; yields a (1 − alpha) CI
               (default 0.05 → 95 % CI).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Tuple ``(d, lower_ci, upper_ci)`` where *d* is the point estimate and
        *lower_ci* / *upper_ci* are the bootstrap percentile bounds.
    """
    rng = np.random.default_rng(seed)

    # Point estimate on the original samples
    d = compute_cohens_d(group1, group2)

    # Bootstrap distribution of Cohen's d
    bootstrap_ds = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample1 = rng.choice(group1, size=len(group1), replace=True)
        resample2 = rng.choice(group2, size=len(group2), replace=True)
        bootstrap_ds[i] = compute_cohens_d(resample1, resample2)

    lower_ci = float(np.percentile(bootstrap_ds, 100 * alpha / 2))
    upper_ci = float(np.percentile(bootstrap_ds, 100 * (1 - alpha / 2)))

    return d, lower_ci, upper_ci


def classify_residual_structure(
    cond_dep_pval: float,
    skewness: float,
    path_dep_pval: float,
    cv: float,
) -> str:
    """Classify an agent's residual behaviour into one of three structural categories.

    Applies the paper's pre-registered decision rules in order of specificity:

    1. **structured_human_shaped** — all four criteria are jointly satisfied:
       conditional dependence on game features (cond_dep_pval < 0.05), a
       non-trivial asymmetric distribution (|skewness| > 0.5), temporal path
       dependence (path_dep_pval < 0.05), and low cross-paraphrase variability
       (cv < 0.2).

    2. **non_human_llm** — no detectable conditioning on game features AND high
       variability (cond_dep_pval ≥ 0.05 AND cv ≥ 0.2).  The agent clusters
       near the benchmark but without coherent structure.

    3. **unstructured_noise** — everything else; some signal is present but the
       full human-shaped profile is not met.

    Args:
        cond_dep_pval: F-test p-value from the conditional dependence OLS.
        skewness: Skewness of the agent's delta distribution.
        path_dep_pval: Lag-coefficient p-value from the path dependence OLS.
        cv: Coefficient of variation from the paraphrase robustness test.

    Returns:
        One of ``'structured_human_shaped'``, ``'non_human_llm'``, or
        ``'unstructured_noise'``.
    """
    # Strongest positive signal: all four diagnostic criteria jointly met
    if (
        cond_dep_pval < 0.05
        and abs(skewness) > 0.5
        and path_dep_pval < 0.05
        and cv < 0.2
    ):
        return 'structured_human_shaped'

    # Near-benchmark behaviour: clearly non-significant conditioning (p > 0.5)
    # AND above-threshold cross-paraphrase variability (cv >= 0.2)
    if cond_dep_pval > 0.5 and cv >= 0.2:
        return 'non_human_llm'

    # Partial signal: some criteria met but not the full human-shaped profile
    return 'unstructured_noise'

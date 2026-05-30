"""Four structural diagnostics from the behavioral game-theory paper methodology.

Each test isolates a distinct dimension of structured, human-shaped behaviour:
  T1 — conditional dependence on game features
  T2 — distributional asymmetry of deltas
  T3 — path dependence across game rounds
  T4 — robustness across paraphrase variants
"""

import re

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm

from src.metrics import compute_cv, compute_skewness, standardize_delta
from src.metrics import classify_residual_structure


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_colname(name: str) -> str:
    """Convert an arbitrary string to a valid Python identifier for OLS formulas.

    Replaces any character that is not alphanumeric or an underscore with '_',
    and prepends 'x_' when the result starts with a digit.

    Args:
        name: Raw column name.

    Returns:
        Sanitized name safe for use in statsmodels formula strings.
    """
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', str(name))
    if sanitized and sanitized[0].isdigit():
        sanitized = 'x_' + sanitized
    return sanitized


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with all column names sanitized via :func:`_sanitize_colname`.

    Appends ``_dup{n}`` suffixes to resolve any collisions that arise after
    sanitization.

    Args:
        df: DataFrame whose columns may contain special characters.

    Returns:
        Copy of *df* with safe, unique column names.
    """
    seen: dict[str, int] = {}
    new_names = []
    for col in df.columns:
        clean = _sanitize_colname(col)
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_dup{seen[clean]}"
        else:
            seen[clean] = 0
        new_names.append(clean)
    renamed = df.copy()
    renamed.columns = new_names
    return renamed


# ---------------------------------------------------------------------------
# Test 1 — Conditional Dependence
# ---------------------------------------------------------------------------

def test_conditional_dependence(
    df: pd.DataFrame,
    delta_col: str = 'delta',
    feature_cols: list[str] | None = None,
    cluster_col: str | None = None,
) -> dict:
    """Regress delta on game-feature predictors and assess structural conditioning.

    Encodes categorical features with :func:`pd.get_dummies` (``drop_first=True``),
    sanitizes column names for the statsmodels formula API, then fits an OLS model.
    When *cluster_col* is supplied, robust HC3 standard errors are used.

    Args:
        df: DataFrame containing at least *delta_col* and the feature columns.
        delta_col: Name of the dependent variable (agent delta from benchmark).
        feature_cols: Predictor column names.  Defaults to
            ``['game_family', 'opponent_info', 'stake', 'framing']``.
            Columns absent from *df* are silently ignored.
        cluster_col: Optional grouping column; activates HC3 covariance.

    Returns:
        Dictionary with keys:

        - ``'model_summary'`` (*str*): Full statsmodels OLS summary text.
        - ``'r_squared'`` (*float*): In-sample R².
        - ``'f_pvalue'`` (*float*): Omnibus F-test p-value.
        - ``'coef_table'`` (*pd.DataFrame*): Coefficients, p-values, 95 % CIs.
        - ``'significant_features'`` (*list[str]*): Original feature names with
          at least one significant coefficient (p < 0.05).
        - ``'structured'`` (*bool*): ``True`` when ``f_pvalue < 0.05``.
    """
    if feature_cols is None:
        feature_cols = ['game_family', 'opponent_info', 'stake', 'framing']

    # Keep only features that are actually present in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Build a minimal working copy with the relevant columns
    working_cols = [delta_col] + feature_cols
    working_df = df[working_cols].dropna().copy()

    # Identify categorical columns for dummy encoding
    categorical_cols = (
        working_df[feature_cols]
        .select_dtypes(include=['object', 'category', 'bool'])
        .columns.tolist()
    )

    # Encode categorical features with pd.get_dummies
    if categorical_cols:
        working_df = pd.get_dummies(
            working_df, columns=categorical_cols, drop_first=True
        )

    # Sanitize all column names so the formula API can parse them
    working_df = _rename_columns(working_df)

    # Recover the sanitized name of the dependent variable
    delta_col_safe = _sanitize_colname(delta_col)

    predictor_cols = [c for c in working_df.columns if c != delta_col_safe]

    if not predictor_cols:
        # No usable predictors — return a degenerate result
        return {
            'model_summary': 'No predictors available.',
            'r_squared': 0.0,
            'f_pvalue': 1.0,
            'coef_table': pd.DataFrame(),
            'significant_features': [],
            'structured': False,
        }

    formula = f"{delta_col_safe} ~ {' + '.join(predictor_cols)}"

    # Using statsmodels OLS — Context7 confirmed
    model = ols(formula, data=working_df)

    if cluster_col is not None and cluster_col in df.columns:
        # Robust HC3 covariance for clustered / heteroskedastic data
        result = model.fit(cov_type='HC3')
    else:
        result = model.fit()

    # Build a tidy coefficient table
    ci = result.conf_int()
    ci.columns = ['ci_lower', 'ci_upper']
    coef_table = pd.DataFrame({
        'coef': result.params,
        'pvalue': result.pvalues,
    }).join(ci)

    # Map dummy column names back to original feature names for readability
    significant_features: list[str] = []
    for feat in feature_cols:
        feat_safe = _sanitize_colname(feat)
        for param_name, pval in result.pvalues.items():
            if param_name == 'Intercept':
                continue
            # Numeric feature: exact match; categorical dummy: prefix match
            is_this_feature = (
                param_name == feat_safe
                or param_name.startswith(feat_safe + '_')
            )
            if is_this_feature and pval < 0.05 and feat not in significant_features:
                significant_features.append(feat)
                break

    return {
        'model_summary': result.summary().as_text(),
        'r_squared': float(result.rsquared),
        'f_pvalue': float(result.f_pvalue),
        'coef_table': coef_table,
        'significant_features': significant_features,
        'structured': bool(result.f_pvalue < 0.05),
    }


# ---------------------------------------------------------------------------
# Test 2 — Distributional Asymmetry
# ---------------------------------------------------------------------------

def test_distributional_asymmetry(
    df: pd.DataFrame,
    delta_col: str = 'delta',
    group_col: str | None = None,
    threshold: float = 0.5,
) -> dict:
    """Assess skewness in the delta distribution as evidence of systematic bias.

    Human decision-making characteristically produces skewed deviations from
    game-theoretic benchmarks (e.g. loss-aversion, framing effects).  An
    absolute skewness exceeding *threshold* is treated as a positive signal.

    Args:
        df: DataFrame containing *delta_col* and, optionally, *group_col*.
        delta_col: Column of agent deviations from the benchmark.
        group_col: Optional stratification column for per-group skewness.
        threshold: Minimum |skewness| to flag asymmetry (default 0.5).

    Returns:
        Dictionary with keys:

        - ``'skewness'`` (*float*): Overall skewness of *delta_col*.
        - ``'skewness_test_stat'`` (*float*): D'Agostino–Pearson test statistic.
        - ``'skewness_pvalue'`` (*float*): Corresponding p-value.
        - ``'exceeds_threshold'`` (*bool*): ``|skewness| > threshold``.
        - ``'group_skewness'`` (*dict*): Per-group skewness if *group_col* given.
        - ``'asymmetric'`` (*bool*): Alias for ``exceeds_threshold``.
    """
    values = df[delta_col].dropna().to_numpy(dtype=float)

    # Using scipy.stats.skew — Context7 confirmed
    skewness = float(stats.skew(values))

    # scipy.stats.skewtest requires n >= 8; fall back gracefully
    if len(values) >= 8:
        skew_stat, skew_pval = stats.skewtest(values)
    else:
        skew_stat, skew_pval = float('nan'), float('nan')

    exceeds = abs(skewness) > threshold

    result: dict = {
        'skewness': skewness,
        'skewness_test_stat': float(skew_stat),
        'skewness_pvalue': float(skew_pval),
        'exceeds_threshold': exceeds,
        'asymmetric': exceeds,
    }

    # Per-group skewness when a grouping column is provided
    if group_col is not None and group_col in df.columns:
        group_skewness: dict[str, float] = {}
        for group, group_df in df.groupby(group_col):
            group_vals = group_df[delta_col].dropna().to_numpy(dtype=float)
            if len(group_vals) >= 3:
                # Using scipy.stats.skew — Context7 confirmed
                group_skewness[str(group)] = float(stats.skew(group_vals))
        result['group_skewness'] = group_skewness

    return result


# ---------------------------------------------------------------------------
# Test 3 — Path Dependence
# ---------------------------------------------------------------------------

def test_path_dependence(
    df: pd.DataFrame,
    y_col: str = 'y',
    lag_col: str = 'y_lag1',
    state_cols: list[str] | None = None,
    session_col: str = 'session_id',
    round_col: str = 'round_idx',
) -> dict:
    """Test whether an agent's actions are auto-correlated across game rounds.

    Creates a within-session lag of *y_col*, then regresses ``y_t`` on
    ``y_{t-1}`` plus any *state_cols* controls using OLS.  Durbin-Watson
    autocorrelation in the residuals is also computed.  If *state_cols* are
    provided and at least one session has ≥ 10 observations, Granger causality
    tests are run to check whether those state variables predict future actions
    beyond the autoregressive component.

    Args:
        df: Repeated-game DataFrame with one row per (session, round) pair.
        y_col: Agent action column to be modelled.
        lag_col: Name to assign to the within-session lagged *y_col* column.
            Recomputed internally regardless of whether it already exists.
        state_cols: Additional right-hand-side columns (e.g. opponent's last
            action).  Absent columns are silently dropped.
        session_col: Column identifying independent game sessions.
        round_col: Column indexing the temporal order within each session.

    Returns:
        Dictionary with keys:

        - ``'lag_coefficient'`` (*float*): OLS coefficient on ``y_{t-1}``.
        - ``'lag_pvalue'`` (*float*): Two-sided p-value for that coefficient.
        - ``'lag_significant'`` (*bool*): ``lag_pvalue < 0.05``.
        - ``'durbin_watson'`` (*float*): DW statistic on OLS residuals.
        - ``'granger_results'`` (*dict* or *None*): Granger test p-values keyed
          by state column and then by lag order; ``None`` when not applicable.
        - ``'path_dependent'`` (*bool*): ``True`` when ``lag_pvalue < 0.05``.
    """
    if state_cols is None:
        state_cols = []

    # Sort and create within-session lag — paper specification
    working_df = df.copy().sort_values([session_col, round_col])
    working_df[lag_col] = (
        working_df.groupby(session_col)[y_col].shift(1)
    )

    # Drop rows where the lag is undefined (first round of each session)
    working_df = working_df.dropna(subset=[lag_col, y_col])

    # Keep only state_cols that actually exist in the dataframe
    state_cols = [c for c in state_cols if c in working_df.columns]

    # Sanitize column names for the formula API
    y_safe = _sanitize_colname(y_col)
    lag_safe = _sanitize_colname(lag_col)

    rhs_terms = [lag_safe] + [_sanitize_colname(c) for c in state_cols]
    formula_df = _rename_columns(working_df[[y_col, lag_col] + state_cols])
    # Align y_col and lag_col to their sanitized names in formula_df
    formula_df.rename(
        columns={
            _sanitize_colname(y_col): y_safe,
            _sanitize_colname(lag_col): lag_safe,
        },
        inplace=True,
    )

    formula = f"{y_safe} ~ {' + '.join(rhs_terms)}"

    # Using statsmodels OLS — Context7 confirmed
    lag_model = ols(formula, data=formula_df).fit()

    lag_coefficient = float(lag_model.params.get(lag_safe, np.nan))
    lag_pvalue = float(lag_model.pvalues.get(lag_safe, np.nan))

    # Durbin-Watson statistic on OLS residuals to check for serial correlation
    dw_stat = float(durbin_watson(lag_model.resid))

    # Granger causality — run only when state_cols are present and sessions
    # are long enough to support the test (≥ 10 observations per session)
    granger_results: dict | None = None
    if state_cols:
        session_lengths = working_df.groupby(session_col)[y_col].count()
        long_sessions = session_lengths[session_lengths >= 10].index

        if len(long_sessions) > 0:
            # Choose the longest qualifying session for a cleaner time series
            longest = session_lengths[long_sessions].idxmax()
            session_data = (
                working_df[working_df[session_col] == longest]
                .sort_values(round_col)
            )

            granger_results = {}
            for scol in state_cols:
                gc_data = session_data[[y_col, scol]].dropna().to_numpy(dtype=float)
                n_obs = len(gc_data)
                if n_obs < 10:
                    continue

                # maxlag capped so the model has enough degrees of freedom
                maxlag = min(2, max(1, n_obs // 5 - 1))
                try:
                    gc_test = grangercausalitytests(gc_data, maxlag=maxlag, verbose=False)
                    granger_results[scol] = {
                        lag: {
                            'ssr_ftest_pvalue': float(gc_test[lag][0]['ssr_ftest'][1]),
                            'ssr_ftest_stat': float(gc_test[lag][0]['ssr_ftest'][0]),
                        }
                        for lag in gc_test
                    }
                except Exception as exc:  # pragma: no cover
                    granger_results[scol] = {'error': str(exc)}

            # If no state column produced valid results, keep None
            if not granger_results:
                granger_results = None

    return {
        'lag_coefficient': lag_coefficient,
        'lag_pvalue': lag_pvalue,
        'lag_significant': bool(lag_pvalue < 0.05),
        'durbin_watson': dw_stat,
        'granger_results': granger_results,
        'path_dependent': bool(lag_pvalue < 0.05),
    }


# ---------------------------------------------------------------------------
# Test 4 — Paraphrase Robustness
# ---------------------------------------------------------------------------

def test_paraphrase_robustness(
    df: pd.DataFrame,
    delta_col: str = 'standardized_delta',
    paraphrase_col: str = 'paraphrase_id',
    cv_threshold: float = 0.2,
) -> dict:
    """Assess consistency of agent behaviour across semantically equivalent prompts.

    Computes the mean standardized delta per paraphrase variant, then measures
    the coefficient of variation (CV) of those means.  A low CV (< *cv_threshold*)
    signals that the agent's decisions are stable regardless of surface-level
    rephrasing — a property consistent with genuine strategic understanding rather
    than prompt sensitivity.

    Args:
        df: DataFrame with *delta_col* and *paraphrase_col*.
        delta_col: Standardized delta column (see :func:`~src.metrics.standardize_delta`).
        paraphrase_col: Column identifying paraphrase variants.
        cv_threshold: CV below which behaviour is deemed robust (default 0.2).

    Returns:
        Dictionary with keys:

        - ``'paraphrase_means'`` (*dict*): Mean *delta_col* keyed by paraphrase ID.
        - ``'cv'`` (*float*): CV of the per-paraphrase means.
        - ``'robust'`` (*bool*): ``cv < cv_threshold``.
        - ``'n_paraphrases'`` (*int*): Number of distinct paraphrase variants.
    """
    # Mean standardized delta per paraphrase variant
    paraphrase_means_series = df.groupby(paraphrase_col)[delta_col].mean()
    paraphrase_means = paraphrase_means_series.to_dict()

    means_array = paraphrase_means_series.to_numpy(dtype=float)

    cv = compute_cv(means_array)

    return {
        'paraphrase_means': {str(k): float(v) for k, v in paraphrase_means.items()},
        'cv': float(cv),
        'robust': bool(cv < cv_threshold),
        'n_paraphrases': int(len(paraphrase_means)),
    }


# ---------------------------------------------------------------------------
# Wrapper — run all four tests
# ---------------------------------------------------------------------------

def run_all_structural_tests(
    df: pd.DataFrame,
    agent_label: str = 'human',
) -> dict:
    """Run all four structural diagnostics and classify the agent's behaviour.

    Executes :func:`test_conditional_dependence`, :func:`test_distributional_asymmetry`,
    :func:`test_path_dependence`, and :func:`test_paraphrase_robustness` on *df*,
    then passes the key scalar outputs to
    :func:`~src.metrics.classify_residual_structure` to produce a final label.

    If an individual test raises an exception (e.g. a required column is absent),
    a sentinel result is recorded and conservative defaults are used for
    classification so that the remaining tests still complete.

    Args:
        df: Filtered DataFrame for a single agent / condition combination.
        agent_label: Human-readable identifier included in the returned dict.

    Returns:
        Nested dictionary with keys:

        - ``'agent_label'`` (*str*)
        - ``'t1_conditional_dep'`` (*dict*): Output of T1, or error dict.
        - ``'t2_distributional_asym'`` (*dict*): Output of T2, or error dict.
        - ``'t3_path_dependence'`` (*dict*): Output of T3, or error dict.
        - ``'t4_paraphrase_robustness'`` (*dict*): Output of T4, or error dict.
        - ``'classification'`` (*str*): One of ``'structured_human_shaped'``,
          ``'non_human_llm'``, or ``'unstructured_noise'``.
    """
    results: dict = {'agent_label': agent_label}

    # --- T1: Conditional Dependence ---
    try:
        t1 = test_conditional_dependence(df)
    except Exception as exc:  # pragma: no cover
        t1 = {'error': str(exc), 'f_pvalue': 1.0, 'structured': False}
    results['t1_conditional_dep'] = t1

    # --- T2: Distributional Asymmetry ---
    try:
        t2 = test_distributional_asymmetry(df)
    except Exception as exc:  # pragma: no cover
        t2 = {'error': str(exc), 'skewness': 0.0, 'asymmetric': False}
    results['t2_distributional_asym'] = t2

    # --- T3: Path Dependence ---
    try:
        t3 = test_path_dependence(df)
    except Exception as exc:  # pragma: no cover
        t3 = {'error': str(exc), 'lag_pvalue': 1.0, 'path_dependent': False}
    results['t3_path_dependence'] = t3

    # --- T4: Paraphrase Robustness ---
    try:
        t4 = test_paraphrase_robustness(df)
    except Exception as exc:  # pragma: no cover
        t4 = {'error': str(exc), 'cv': np.inf, 'robust': False}
    results['t4_paraphrase_robustness'] = t4

    # --- Classification ---
    # Extract scalar inputs, using conservative defaults on failure
    cond_dep_pval = float(t1.get('f_pvalue', 1.0))
    skewness = float(t2.get('skewness', 0.0))
    path_dep_pval = float(t3.get('lag_pvalue', 1.0))
    cv = float(t4.get('cv', np.inf))

    results['classification'] = classify_residual_structure(
        cond_dep_pval=cond_dep_pval,
        skewness=skewness,
        path_dep_pval=path_dep_pval,
        cv=cv,
    )

    return results

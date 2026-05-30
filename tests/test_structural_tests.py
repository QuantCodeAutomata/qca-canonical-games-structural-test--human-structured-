import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.structural_tests import (
    test_conditional_dependence as _test_conditional_dependence,
    test_distributional_asymmetry as _test_distributional_asymmetry,
    test_path_dependence as _test_path_dependence,
    test_paraphrase_robustness as _test_paraphrase_robustness,
    run_all_structural_tests,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_path_dep_df(rng, n_sessions=10, n_rounds=10, ar_coef=0.7):
    """Build a session × round DataFrame with AR(1) structure."""
    records = []
    for s in range(n_sessions):
        y_prev = rng.normal(0, 1)
        for r in range(n_rounds):
            y = ar_coef * y_prev + rng.normal(0, 1)
            records.append({'session_id': s, 'round_idx': r, 'y': y})
            y_prev = y
    return pd.DataFrame(records)


def _make_full_df(rng, n=100):
    """Minimal df satisfying every structural test's column requirements."""
    game_family = ['A'] * (n // 2) + ['B'] * (n // 2)
    delta = rng.normal(0, 1, n)
    session_ids = np.repeat(np.arange(n // 10), 10)
    round_idxs = np.tile(np.arange(10), n // 10)
    paraphrase_ids = np.tile(np.arange(5), n // 5)
    return pd.DataFrame({
        'delta': delta,
        'game_family': game_family,
        'session_id': session_ids,
        'round_idx': round_idxs,
        'y': delta,
        'paraphrase_id': paraphrase_ids,
        'standardized_delta': delta / (np.std(delta) + 1e-9),
    })


# ---------------------------------------------------------------------------
# test_conditional_dependence
# ---------------------------------------------------------------------------

def test_conditional_dependence_detects_effect():
    rng = np.random.default_rng(42)
    n = 200
    game_family = ['A'] * 100 + ['B'] * 100
    noise = rng.normal(0, 1, n)
    delta = np.array([5.0 * (gf == 'A') + e for gf, e in zip(game_family, noise)])
    df = pd.DataFrame({'delta': delta, 'game_family': game_family})

    result = _test_conditional_dependence(df)
    assert 'game_family_A' in result['significant_features'] or result['structured'] is True, (
        "Expected effect of game_family to be detected"
    )


def test_conditional_dependence_no_effect():
    rng = np.random.default_rng(99)
    n = 200
    delta = rng.normal(0, 1, n)
    game_family = rng.choice(['A', 'B'], size=n)
    df = pd.DataFrame({'delta': delta, 'game_family': game_family})

    result = _test_conditional_dependence(df)
    assert result['structured'] is False, (
        "Pure noise data should not be classified as structured"
    )


# ---------------------------------------------------------------------------
# test_distributional_asymmetry
# ---------------------------------------------------------------------------

def test_distributional_asymmetry_right_skew():
    rng = np.random.default_rng(1)
    delta = rng.exponential(scale=5, size=500)
    df = pd.DataFrame({'delta': delta})

    result = _test_distributional_asymmetry(df)
    assert result['skewness'] > 0, (
        f"Exponential data should have positive skewness, got {result['skewness']}"
    )
    assert result['exceeds_threshold'] is True, (
        "Strong right skew should exceed asymmetry threshold"
    )


def test_distributional_asymmetry_symmetric():
    rng = np.random.default_rng(2)
    delta = rng.normal(0, 1, 500)
    df = pd.DataFrame({'delta': delta})

    result = _test_distributional_asymmetry(df)
    assert result['exceeds_threshold'] is False, (
        "Symmetric normal data should not exceed asymmetry threshold"
    )


# ---------------------------------------------------------------------------
# test_path_dependence
# ---------------------------------------------------------------------------

def test_path_dependence_with_lag():
    rng = np.random.default_rng(3)
    df = _make_path_dep_df(rng, n_sessions=10, n_rounds=10, ar_coef=0.7)

    result = _test_path_dependence(df)
    assert result['lag_significant'] is True, (
        "AR(1) time series should show significant lag dependence"
    )


def test_path_dependence_no_lag():
    rng = np.random.default_rng(4)
    n_sessions, n_rounds = 10, 10
    records = [
        {'session_id': s, 'round_idx': r, 'y': rng.normal(0, 1)}
        for s in range(n_sessions)
        for r in range(n_rounds)
    ]
    df = pd.DataFrame(records)

    result = _test_path_dependence(df)
    assert result['lag_significant'] is False, (
        "Independent noise should not show significant lag dependence"
    )


# ---------------------------------------------------------------------------
# test_paraphrase_robustness
# ---------------------------------------------------------------------------

def test_paraphrase_robustness_stable():
    rng = np.random.default_rng(5)
    n_per_paraphrase = 30
    records = []
    for pid in range(5):
        vals = 0.5 + rng.normal(0, 0.005, n_per_paraphrase)
        records.extend({'paraphrase_id': pid, 'standardized_delta': v} for v in vals)
    df = pd.DataFrame(records)

    result = _test_paraphrase_robustness(df)
    assert result['cv'] < 0.2, f"Stable paraphrases should have CV < 0.2, got {result['cv']}"
    assert result['robust'] is True, "Stable paraphrases should be classified as robust"


def test_paraphrase_robustness_unstable():
    rng = np.random.default_rng(6)
    n_per_paraphrase = 30
    records = []
    for pid in range(5):
        mean_delta = (pid + 1) * 2.0          # means: 2, 4, 6, 8, 10 — large spread
        vals = mean_delta + rng.normal(0, 0.05, n_per_paraphrase)
        records.extend({'paraphrase_id': pid, 'standardized_delta': v} for v in vals)
    df = pd.DataFrame(records)

    result = _test_paraphrase_robustness(df)
    assert result['cv'] > 0.2, f"Unstable paraphrases should have CV > 0.2, got {result['cv']}"
    assert result['robust'] is False, "Unstable paraphrases should not be classified as robust"


# ---------------------------------------------------------------------------
# run_all_structural_tests
# ---------------------------------------------------------------------------

def test_run_all_structural_tests_returns_all_keys():
    rng = np.random.default_rng(0)
    df = _make_full_df(rng, n=100)

    result = run_all_structural_tests(df)

    required_keys = {
        't1_conditional_dep',
        't2_distributional_asym',
        't3_path_dependence',
        't4_paraphrase_robustness',
        'classification',
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys in run_all_structural_tests output: {missing}"

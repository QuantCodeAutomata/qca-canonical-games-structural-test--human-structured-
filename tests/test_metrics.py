import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.metrics import (
    compute_cohens_d,
    compute_cv,
    compute_skewness,
    standardize_delta,
    classify_residual_structure,
)


def test_cohens_d_known():
    # Zero-variance case: pooled std = 0, should return 0.0 (not raise)
    group1_const = np.array([10] * 50)
    group2_const = np.array([20] * 50)
    d_zero_var = compute_cohens_d(group1_const, group2_const)
    assert d_zero_var == 0.0, (
        f"Zero-variance Cohen's d should be 0.0, got {d_zero_var}"
    )

    # N(0,1) vs N(1,1): true d = 1.0; with n=500 should be within 0.2
    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1, 500)
    g2 = rng.normal(1, 1, 500)
    d = compute_cohens_d(g1, g2)
    assert abs(abs(d) - 1.0) < 0.2, f"Expected |Cohen's d| ≈ 1.0, got {d}"


def test_cohens_d_equal_groups():
    rng = np.random.default_rng(0)
    group = rng.normal(5, 2, 100)
    d = compute_cohens_d(group, group)
    assert d == 0.0, f"Equal groups should yield d=0.0, got {d}"


def test_cv_computation():
    # Constant values: std=0, cv should be 0
    cv_zero = compute_cv(np.array([1.0, 1.0, 1.0, 1.0]))
    assert cv_zero == 0.0, f"CV of constant array should be 0.0, got {cv_zero}"

    # Known values [1,2,3,4]: cv = std/mean
    values = np.array([1.0, 2.0, 3.0, 4.0])
    expected_cv = np.std(values, ddof=1) / np.mean(values)
    cv = compute_cv(values)
    assert abs(cv - expected_cv) < 1e-9, (
        f"CV([1,2,3,4]) should be {expected_cv:.6f}, got {cv:.6f}"
    )


def test_cv_zero_mean():
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    cv = compute_cv(values)
    assert cv == np.inf or np.isnan(cv), (
        f"CV with zero mean should be inf or nan, got {cv}"
    )


def test_skewness_symmetric():
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1, 1000)
    skew = compute_skewness(data)
    assert abs(skew) < 0.3, (
        f"Skewness of normal distribution should be near 0, got {skew}"
    )


def test_skewness_right_tail():
    rng = np.random.default_rng(7)
    data = rng.exponential(scale=5, size=500)
    skew = compute_skewness(data)
    assert skew > 0, f"Exponential distribution should have positive skewness, got {skew}"


def test_standardize_delta():
    result = standardize_delta(10, 0, 100)
    assert abs(result - 0.1) < 1e-9, f"standardize_delta(10, 0, 100) should be 0.1, got {result}"


def test_classify_structured():
    label = classify_residual_structure(
        cond_dep_pval=0.01,
        skewness=0.8,
        path_dep_pval=0.02,
        cv=0.1,
    )
    assert label == 'structured_human_shaped', (
        f"Expected 'structured_human_shaped', got '{label}'"
    )


def test_classify_noise():
    label = classify_residual_structure(
        cond_dep_pval=0.5,
        skewness=0.1,
        path_dep_pval=0.6,
        cv=0.5,
    )
    assert label == 'unstructured_noise', (
        f"Expected 'unstructured_noise', got '{label}'"
    )


def test_classify_llm():
    label = classify_residual_structure(
        cond_dep_pval=0.8,
        skewness=0.1,
        path_dep_pval=0.3,
        cv=0.5,
    )
    assert label == 'non_human_llm', (
        f"Expected 'non_human_llm', got '{label}'"
    )

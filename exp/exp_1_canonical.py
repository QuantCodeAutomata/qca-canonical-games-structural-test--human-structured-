"""Experiment 1: Canonical Games Structural Test.

Loads or generates canonical-game data, runs all four structural tests
(conditional_dep, distributional_asym, path_dep, paraphrase_robustness)
for each agent type, computes summary statistics and Cohen's d comparisons,
and persists results to disk.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

from data.generate_data import generate_exp1_data
from src.metrics import compute_cohens_d
from src.structural_tests import run_all_structural_tests

AGENT_TYPES: list[str] = ['human', 'gpt-base', 'gpt-reasoning']


def _json_default(obj: object) -> object:
    """Fallback serialiser for json.dump to handle NumPy / pandas types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return str(obj)


def run_exp1(
    data_path: str = 'data/exp1_data.csv',
    results_dir: str = 'results',
) -> dict:
    """Run Experiment 1: Canonical Games Structural Test.

    Steps
    -----
    1. Load ``data_path``; generate via :func:`generate_exp1_data` if absent.
    2. Split into per-agent-type subsets.
    3. Run all four structural tests for each agent type (the path_dep test
       operates only on rows where ``round_idx > 0``; this is handled
       internally by :func:`run_all_structural_tests`).
    4. Compute summary statistics (mean delta, mean |delta|, std delta) grouped
       by ``agent_type`` × ``game_id``.
    5. Compute Cohen's d for human vs gpt-base on |delta|.
    6. Compare human vs LLM mean |delta| and residual-structure classifications.
    7. Persist results to ``results_dir/exp1_results.json`` and
       ``results_dir/exp1_summary.csv``.
    8. Print a formatted summary table.

    Parameters
    ----------
    data_path:
        Path to the canonical-game CSV.  Generated automatically when absent.
    results_dir:
        Directory that receives the JSON and CSV output files.

    Returns
    -------
    dict
        Full results dict containing test outputs, summary statistics, and
        comparison metrics.
    """
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load or generate data
    # ------------------------------------------------------------------ #
    if not os.path.exists(data_path):
        print(f"[exp1] '{data_path}' not found — generating...")
        generate_exp1_data(data_path)

    print(f"[exp1] Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(
        f"[exp1] {len(df)} rows loaded | "
        f"agent types present: {df['agent_type'].unique().tolist()}"
    )

    df['abs_delta'] = df['delta'].abs()

    # ------------------------------------------------------------------ #
    # 2. Structural tests — one call per agent type
    # ------------------------------------------------------------------ #
    test_results: dict = {}
    for agent_type in AGENT_TYPES:
        agent_df = df[df['agent_type'] == agent_type].copy()
        if agent_df.empty:
            print(f"[exp1] No rows for '{agent_type}' — skipping.")
            continue
        has_repeated = bool((agent_df['round_idx'] > 0).any())
        print(
            f"[exp1] Structural tests for '{agent_type}' "
            f"({len(agent_df)} rows, repeated-game data: {has_repeated})..."
        )
        test_results[agent_type] = run_all_structural_tests(agent_df)

    # ------------------------------------------------------------------ #
    # 3. Summary statistics: mean delta / mean |delta| / std delta
    # ------------------------------------------------------------------ #
    summary = (
        df.groupby(['agent_type', 'game_id'])
        .agg(
            mean_delta=('delta', 'mean'),
            mean_abs_delta=('abs_delta', 'mean'),
            std_delta=('delta', 'std'),
            n=('delta', 'count'),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # 4. Cohen's d — human vs gpt-base on |delta|
    # ------------------------------------------------------------------ #
    human_abs = df.loc[df['agent_type'] == 'human', 'abs_delta'].dropna().values
    gptbase_abs = df.loc[df['agent_type'] == 'gpt-base', 'abs_delta'].dropna().values
    cohens_d_h_vs_base = float(compute_cohens_d(human_abs, gptbase_abs))

    # ------------------------------------------------------------------ #
    # 5. Human vs LLM delta comparison
    # ------------------------------------------------------------------ #
    llm_abs = (
        df.loc[df['agent_type'].isin(['gpt-base', 'gpt-reasoning']), 'abs_delta']
        .dropna()
        .values
    )
    human_mean_abs = float(human_abs.mean()) if len(human_abs) > 0 else float('nan')
    llm_mean_abs = float(llm_abs.mean()) if len(llm_abs) > 0 else float('nan')
    human_delta_larger = bool(human_mean_abs > llm_mean_abs)

    # ------------------------------------------------------------------ #
    # 6. Residual-structure classification per agent type (from test results)
    # ------------------------------------------------------------------ #
    residual_structure: dict = {}
    for agent_type in AGENT_TYPES:
        if agent_type in test_results:
            residual_structure[agent_type] = test_results[agent_type].get('classification', 'unknown')

    human_structured = residual_structure.get('human')
    llm_structures = {
        at: residual_structure[at]
        for at in ['gpt-base', 'gpt-reasoning']
        if at in residual_structure
    }

    # ------------------------------------------------------------------ #
    # 7. Assemble results dict
    # ------------------------------------------------------------------ #
    results: dict = {
        'experiment': 'exp1_canonical',
        'n_total': int(len(df)),
        'agent_types': AGENT_TYPES,
        'test_results': test_results,
        'cohens_d_human_vs_gptbase': cohens_d_h_vs_base,
        'human_mean_abs_delta': human_mean_abs,
        'llm_mean_abs_delta': llm_mean_abs,
        'human_delta_larger': human_delta_larger,
        'residual_structure': residual_structure,
        'human_residual_structured': human_structured,
        'llm_residual_structures': llm_structures,
        'summary': summary.to_dict(orient='records'),
    }

    # ------------------------------------------------------------------ #
    # 8. Persist outputs
    # ------------------------------------------------------------------ #
    results_path = os.path.join(results_dir, 'exp1_results.json')
    with open(results_path, 'w') as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    print(f"[exp1] Results saved to '{results_path}'")

    summary_path = os.path.join(results_dir, 'exp1_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"[exp1] Summary saved to '{summary_path}'")

    # ------------------------------------------------------------------ #
    # 9. Printed summary table
    # ------------------------------------------------------------------ #
    agent_level = (
        df.groupby('agent_type')
        .agg(
            mean_delta=('delta', 'mean'),
            mean_abs_delta=('abs_delta', 'mean'),
            std_delta=('delta', 'std'),
        )
        .reset_index()
    )
    agent_level['residual_structure'] = agent_level['agent_type'].map(residual_structure)

    print("\n=== Experiment 1: Canonical Games — Agent-Level Summary ===")
    print(agent_level.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nCohen's d (human vs gpt-base |delta|) : {cohens_d_h_vs_base:.4f}")
    print(f"Human mean |delta|                     : {human_mean_abs:.4f}")
    print(f"LLM mean |delta|                       : {llm_mean_abs:.4f}")
    print(f"Human delta larger than LLM            : {human_delta_larger}")
    print("\nPer-agent-type × per-game summary (first 20 rows):")
    print(summary.head(20).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results = run_exp1()
    print('Exp 1 complete. Results saved to results/')

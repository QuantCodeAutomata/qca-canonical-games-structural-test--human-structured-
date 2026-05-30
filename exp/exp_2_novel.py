"""Experiment 2: Novel Games Structural Test.

Loads or generates novel-game data, runs all four structural tests per agent
type, and performs the key canonical-vs-novel comparison of LLM delta
magnitude.  The central prediction is that LLM |delta| on novel games exceeds
|delta| on canonical games, whereas human residuals remain structured on
both game families.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

from data.generate_data import generate_exp2_data
from src.metrics import compute_cohens_d
from src.structural_tests import run_all_structural_tests

AGENT_TYPES: list[str] = ['human', 'gpt-base', 'gpt-reasoning']
LLM_TYPES: list[str] = ['gpt-base', 'gpt-reasoning']


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


def run_exp2(
    data_path: str = 'data/exp2_data.csv',
    results_dir: str = 'results',
) -> dict:
    """Run Experiment 2: Novel Games Structural Test.

    Steps
    -----
    1. Load ``data_path``; generate via :func:`generate_exp2_data` if absent.
    2. Run all four structural tests for each agent type.
    3. Compare LLM |delta| magnitude: novel games vs canonical games (exp1).
       Novel prediction: LLM |delta| on novel > LLM |delta| on canonical.
    4. Classify human residual structure on novel games.
    5. Build a 2 × 3 comparison table (game_type × agent_type) of mean |delta|.
    6. Persist results to ``results_dir/exp2_results.json`` and
       ``results_dir/exp2_summary.csv``.
    7. Print a formatted summary table.

    Parameters
    ----------
    data_path:
        Path to the novel-game CSV.  Generated automatically when absent.
    results_dir:
        Directory that receives the JSON and CSV output files.

    Returns
    -------
    dict
        Full results dict containing test outputs, comparison metrics, and
        summary statistics.
    """
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load or generate novel-game data
    # ------------------------------------------------------------------ #
    if not os.path.exists(data_path):
        print(f"[exp2] '{data_path}' not found — generating...")
        generate_exp2_data(data_path)

    print(f"[exp2] Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(
        f"[exp2] {len(df)} rows loaded | "
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
            print(f"[exp2] No rows for '{agent_type}' — skipping.")
            continue
        has_repeated = bool((agent_df['round_idx'] > 0).any())
        print(
            f"[exp2] Structural tests for '{agent_type}' "
            f"({len(agent_df)} rows, repeated-game data: {has_repeated})..."
        )
        test_results[agent_type] = run_all_structural_tests(agent_df)

    # ------------------------------------------------------------------ #
    # 3. Summary statistics: mean delta / mean |delta| / std delta
    # ------------------------------------------------------------------ #
    summary_novel = (
        df.groupby(['agent_type', 'game_id'])
        .agg(
            mean_delta=('delta', 'mean'),
            mean_abs_delta=('abs_delta', 'mean'),
            std_delta=('delta', 'std'),
            n=('delta', 'count'),
        )
        .reset_index()
    )
    summary_novel['game_type'] = 'novel'

    # ------------------------------------------------------------------ #
    # 4. Canonical-vs-novel comparison
    #    Load exp1 data if available; otherwise canonical_means stays empty.
    # ------------------------------------------------------------------ #
    exp1_path = os.path.join(os.path.dirname(data_path), 'exp1_data.csv')
    canonical_df: pd.DataFrame | None = None
    canonical_means: dict = {}

    if os.path.exists(exp1_path):
        print(f"[exp2] Loading canonical data from '{exp1_path}' for comparison...")
        canonical_df = pd.read_csv(exp1_path)
        canonical_df['abs_delta'] = canonical_df['delta'].abs()
        for agent_type in AGENT_TYPES:
            vals = (
                canonical_df.loc[canonical_df['agent_type'] == agent_type, 'abs_delta']
                .dropna()
                .values
            )
            canonical_means[agent_type] = float(vals.mean()) if len(vals) > 0 else float('nan')
    else:
        print(f"[exp2] '{exp1_path}' not found — canonical comparison unavailable.")

    novel_means: dict = {}
    for agent_type in AGENT_TYPES:
        vals = df.loc[df['agent_type'] == agent_type, 'abs_delta'].dropna().values
        novel_means[agent_type] = float(vals.mean()) if len(vals) > 0 else float('nan')

    # Novel prediction: LLM novel |delta| > canonical |delta|
    novel_larger_than_canonical: dict = {}
    for at in LLM_TYPES:
        n_mean = novel_means.get(at, float('nan'))
        c_mean = canonical_means.get(at, float('nan'))
        if np.isnan(n_mean) or np.isnan(c_mean):
            novel_larger_than_canonical[at] = None
        else:
            novel_larger_than_canonical[at] = bool(n_mean > c_mean)

    # Cohen's d: novel vs canonical for each LLM type
    cohens_d_novel_vs_canonical: dict = {}
    if canonical_df is not None:
        for at in LLM_TYPES:
            canon_abs = (
                canonical_df.loc[canonical_df['agent_type'] == at, 'abs_delta']
                .dropna()
                .values
            )
            novel_abs = df.loc[df['agent_type'] == at, 'abs_delta'].dropna().values
            if len(canon_abs) > 0 and len(novel_abs) > 0:
                cohens_d_novel_vs_canonical[at] = float(
                    compute_cohens_d(novel_abs, canon_abs)
                )

    # ------------------------------------------------------------------ #
    # 5. Residual-structure classification (novel games, from test results)
    # ------------------------------------------------------------------ #
    residual_structure: dict = {}
    for agent_type in AGENT_TYPES:
        if agent_type in test_results:
            residual_structure[agent_type] = test_results[agent_type].get('classification', 'unknown')

    # ------------------------------------------------------------------ #
    # 6. Comparison table: canonical vs novel × agent type
    # ------------------------------------------------------------------ #
    comparison_rows = []
    for agent_type in AGENT_TYPES:
        comparison_rows.append({
            'agent_type': agent_type,
            'game_type': 'canonical',
            'mean_abs_delta': canonical_means.get(agent_type, float('nan')),
        })
        comparison_rows.append({
            'agent_type': agent_type,
            'game_type': 'novel',
            'mean_abs_delta': novel_means.get(agent_type, float('nan')),
        })
    comparison_table = pd.DataFrame(comparison_rows)

    # ------------------------------------------------------------------ #
    # 7. Assemble results dict
    # ------------------------------------------------------------------ #
    results: dict = {
        'experiment': 'exp2_novel',
        'n_total': int(len(df)),
        'agent_types': AGENT_TYPES,
        'test_results': test_results,
        'novel_means': novel_means,
        'canonical_means': canonical_means,
        'novel_larger_than_canonical': novel_larger_than_canonical,
        'cohens_d_novel_vs_canonical': cohens_d_novel_vs_canonical,
        'residual_structure': residual_structure,
        'comparison_table': comparison_table.to_dict(orient='records'),
        'summary': summary_novel.to_dict(orient='records'),
    }

    # ------------------------------------------------------------------ #
    # 8. Persist outputs
    # ------------------------------------------------------------------ #
    results_path = os.path.join(results_dir, 'exp2_results.json')
    with open(results_path, 'w') as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    print(f"[exp2] Results saved to '{results_path}'")

    summary_path = os.path.join(results_dir, 'exp2_summary.csv')
    summary_novel.to_csv(summary_path, index=False)
    print(f"[exp2] Summary saved to '{summary_path}'")

    # ------------------------------------------------------------------ #
    # 9. Printed summary table
    # ------------------------------------------------------------------ #
    print("\n=== Experiment 2: Novel Games — Canonical vs Novel Comparison ===")
    pivot = comparison_table.pivot(
        index='agent_type', columns='game_type', values='mean_abs_delta'
    )
    if 'canonical' in pivot.columns and 'novel' in pivot.columns:
        pivot['novel_larger'] = pivot['novel'] > pivot['canonical']
    print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\nLLM novel > canonical (|delta| magnitude):")
    for at, result in novel_larger_than_canonical.items():
        print(f"  {at}: {result}")

    if cohens_d_novel_vs_canonical:
        print("\nCohen's d (novel vs canonical |delta|):")
        for at, d in cohens_d_novel_vs_canonical.items():
            print(f"  {at}: {d:.4f}")

    print("\nResidual structure on novel games:")
    for at, struct in residual_structure.items():
        print(f"  {at}: {struct}")

    print("\nPer-agent-type × per-game summary (first 20 rows):")
    print(summary_novel.head(20).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results = run_exp2()
    print('Exp 2 complete. Results saved to results/')

"""Experiment 3: Contest-Based Peer-Signal Individuation Moderator Test.

Tests whether named (individualised) vs aggregate peer-signal conditions
produce a meaningful difference in |delta| for human participants.  The
primary falsification criterion is Cohen's d >= 0.5 for the human named vs
aggregate contrast.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data.generate_data import generate_exp3_data
from src.metrics import compute_cohens_d
from src.structural_tests import run_all_structural_tests

AGENT_TYPES: list[str] = ['human', 'gpt-base', 'gpt-reasoning']
LLM_TYPES: list[str] = ['gpt-base', 'gpt-reasoning']
COHENS_D_THRESHOLD: float = 0.5
ALPHA: float = 0.05


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


def run_exp3(
    data_path: str = 'data/exp3_data.csv',
    results_dir: str = 'results',
) -> dict:
    """Run Experiment 3: Contest-Based Peer-Signal Individuation Moderator Test.

    Steps
    -----
    1. Load ``data_path``; generate via :func:`generate_exp3_data` if absent.
    2. Compute the primary endpoint: Cohen's d for human |delta| (named) vs
       |delta| (aggregate).  Falsification criterion: d >= 0.5.
    3. Compute per-game (3 games) Cohen's d for humans.
    4. Compute named-vs-aggregate Cohen's d for each LLM type.
    5. Fit OLS regression on human data:
       ``abs_delta ~ C(opponent_info) + C(game_id) + C(opponent_info):C(game_id)``
       and test the ``opponent_info`` coefficient at p < 0.05.
    6. Run path-dependence tests for repeated rounds in each condition ×
       agent-type subset.
    7. Build a results table indexed by agent_type × condition × game_id with
       mean_abs_delta and Cohen's d columns.
    8. Persist results to ``results_dir/exp3_results.json`` and
       ``results_dir/exp3_summary.csv``.
    9. Print a formatted summary with the falsification verdict.

    Parameters
    ----------
    data_path:
        Path to the contest-game CSV.  Generated automatically when absent.
    results_dir:
        Directory that receives the JSON and CSV output files.

    Returns
    -------
    dict
        Full results dict containing Cohen's d values, regression outputs,
        path-dep tests, the results table, and the falsification assessment.
    """
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load or generate contest data
    # ------------------------------------------------------------------ #
    if not os.path.exists(data_path):
        print(f"[exp3] '{data_path}' not found — generating...")
        generate_exp3_data(data_path)

    print(f"[exp3] Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(
        f"[exp3] {len(df)} rows loaded | "
        f"agent types: {df['agent_type'].unique().tolist()} | "
        f"conditions (opponent_info): {df['opponent_info'].unique().tolist()}"
    )

    df['abs_delta'] = df['delta'].abs()

    conditions: list = df['opponent_info'].unique().tolist()
    games: list = df['game_id'].unique().tolist()

    # ------------------------------------------------------------------ #
    # 2. PRIMARY ENDPOINT — human Cohen's d: named vs aggregate
    # ------------------------------------------------------------------ #
    human_df = df[df['agent_type'] == 'human'].copy()

    named_abs = (
        human_df.loc[human_df['opponent_info'] == 'named', 'abs_delta']
        .dropna()
        .values
    )
    aggregate_abs = (
        human_df.loc[human_df['opponent_info'] == 'aggregate', 'abs_delta']
        .dropna()
        .values
    )
    primary_cohens_d = float(compute_cohens_d(named_abs, aggregate_abs))
    print(
        f"[exp3] Primary endpoint — "
        f"human Cohen's d (named vs aggregate): {primary_cohens_d:.4f}"
    )

    # ------------------------------------------------------------------ #
    # 3. Per-game Cohen's d (humans)
    # ------------------------------------------------------------------ #
    per_game_cohens_d_human: dict = {}
    for game in games:
        g_df = human_df[human_df['game_id'] == game]
        named_g = g_df.loc[g_df['opponent_info'] == 'named', 'abs_delta'].dropna().values
        agg_g = g_df.loc[g_df['opponent_info'] == 'aggregate', 'abs_delta'].dropna().values
        if len(named_g) > 0 and len(agg_g) > 0:
            per_game_cohens_d_human[str(game)] = float(compute_cohens_d(named_g, agg_g))

    # ------------------------------------------------------------------ #
    # 4. Per-LLM Cohen's d (named vs aggregate)
    # ------------------------------------------------------------------ #
    llm_cohens_d: dict = {}
    for agent_type in LLM_TYPES:
        at_df = df[df['agent_type'] == agent_type]
        named_at = at_df.loc[at_df['opponent_info'] == 'named', 'abs_delta'].dropna().values
        agg_at = at_df.loc[at_df['opponent_info'] == 'aggregate', 'abs_delta'].dropna().values
        if len(named_at) > 0 and len(agg_at) > 0:
            llm_cohens_d[agent_type] = float(compute_cohens_d(named_at, agg_at))

    # ------------------------------------------------------------------ #
    # 5. OLS regression on human data
    # ------------------------------------------------------------------ #
    print("[exp3] Fitting OLS regression on human data...")
    regression_results: dict = {}
    if len(human_df) > 0:
        try:
            model = smf.ols(
                'abs_delta ~ C(opponent_info) + C(game_id) + C(opponent_info):C(game_id)',
                data=human_df,
            ).fit()

            # Identify all opponent_info level coefficients (e.g. C(opponent_info)[T.named])
            named_coef_keys = [
                k for k in model.pvalues.index
                if 'opponent_info' in k and '[T.' in k
            ]
            named_pvals = {k: float(model.pvalues[k]) for k in named_coef_keys}
            named_coefs = {k: float(model.params[k]) for k in named_coef_keys}
            named_significant = any(pv < ALPHA for pv in named_pvals.values())

            regression_results = {
                'r_squared': float(model.rsquared),
                'adj_r_squared': float(model.rsquared_adj),
                'f_pvalue': float(model.f_pvalue),
                'n_obs': int(model.nobs),
                'named_condition_coefs': named_coefs,
                'named_condition_pvalues': named_pvals,
                'named_condition_significant': named_significant,
                'model_summary': model.summary().as_text(),
            }
            print(
                f"[exp3] Regression R²={model.rsquared:.4f} | "
                f"named_condition significant (p<{ALPHA}): {named_significant}"
            )
        except Exception as exc:
            print(f"[exp3] Regression failed: {exc}")
            regression_results = {'error': str(exc)}

    # ------------------------------------------------------------------ #
    # 6. Path-dependence tests per condition × agent type
    # ------------------------------------------------------------------ #
    print("[exp3] Running path-dependence tests per condition...")
    path_dep_by_condition: dict = {}
    for cond in conditions:
        cond_results: dict = {}
        for agent_type in AGENT_TYPES:
            sub_df = df[
                (df['opponent_info'] == cond) & (df['agent_type'] == agent_type)
            ].copy()
            if sub_df.empty or not (sub_df['round_idx'] > 0).any():
                continue
            print(
                f"[exp3]   condition='{cond}', agent='{agent_type}' "
                f"({len(sub_df)} rows)..."
            )
            cond_results[agent_type] = run_all_structural_tests(sub_df)
        if cond_results:
            path_dep_by_condition[str(cond)] = cond_results

    # ------------------------------------------------------------------ #
    # 7. Results table: agent_type × condition × game × mean_abs_delta × d
    #    Cohen's d (per row) reflects named vs aggregate for that
    #    agent_type × game_id pairing.
    # ------------------------------------------------------------------ #
    table_rows = []
    for agent_type in AGENT_TYPES:
        at_df = df[df['agent_type'] == agent_type]
        for game in games:
            game_at_df = at_df[at_df['game_id'] == game]
            # Compute Cohen's d once per (agent_type, game) pair
            named_vals = (
                game_at_df.loc[game_at_df['opponent_info'] == 'named', 'abs_delta']
                .dropna()
                .values
            )
            agg_vals = (
                game_at_df.loc[game_at_df['opponent_info'] == 'aggregate', 'abs_delta']
                .dropna()
                .values
            )
            d_val = (
                float(compute_cohens_d(named_vals, agg_vals))
                if len(named_vals) > 0 and len(agg_vals) > 0
                else float('nan')
            )
            for cond in conditions:
                sub = game_at_df[game_at_df['opponent_info'] == cond]
                if sub.empty:
                    continue
                table_rows.append({
                    'agent_type': agent_type,
                    'condition': str(cond),
                    'game_id': str(game),
                    'mean_abs_delta': float(sub['abs_delta'].mean()),
                    'cohens_d': d_val,
                    'n': int(len(sub)),
                })
    results_table = pd.DataFrame(table_rows)

    # ------------------------------------------------------------------ #
    # 8. Falsification assessment
    # ------------------------------------------------------------------ #
    falsification_passed = bool(primary_cohens_d >= COHENS_D_THRESHOLD)
    print(
        f"[exp3] Falsification assessment: "
        f"human Cohen's d = {primary_cohens_d:.4f} "
        f">= {COHENS_D_THRESHOLD} → "
        f"{'PASSED' if falsification_passed else 'FAILED'}"
    )

    # ------------------------------------------------------------------ #
    # 9. Assemble results dict
    # ------------------------------------------------------------------ #
    results: dict = {
        'experiment': 'exp3_moderator',
        'n_total': int(len(df)),
        'agent_types': AGENT_TYPES,
        'conditions': [str(c) for c in conditions],
        'games': [str(g) for g in games],
        'primary_cohens_d_human': primary_cohens_d,
        'cohens_d_threshold': COHENS_D_THRESHOLD,
        'falsification_passed': falsification_passed,
        'per_game_cohens_d_human': per_game_cohens_d_human,
        'llm_cohens_d': llm_cohens_d,
        'regression': regression_results,
        'path_dep_by_condition': path_dep_by_condition,
        'results_table': results_table.to_dict(orient='records'),
    }

    # ------------------------------------------------------------------ #
    # 10. Persist outputs
    # ------------------------------------------------------------------ #
    results_path = os.path.join(results_dir, 'exp3_results.json')
    with open(results_path, 'w') as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    print(f"[exp3] Results saved to '{results_path}'")

    summary_path = os.path.join(results_dir, 'exp3_summary.csv')
    results_table.to_csv(summary_path, index=False)
    print(f"[exp3] Summary saved to '{summary_path}'")

    # ------------------------------------------------------------------ #
    # 11. Printed summary table
    # ------------------------------------------------------------------ #
    print("\n=== Experiment 3: Moderator Test — Results Table ===")
    print(results_table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    verdict = 'PASSED ✓' if falsification_passed else 'FAILED ✗'
    print(f"\nPrimary endpoint — human Cohen's d (named vs aggregate) : {primary_cohens_d:.4f}")
    print(f"Falsification criterion (d >= {COHENS_D_THRESHOLD})             : {verdict}")

    if regression_results and 'named_condition_significant' in regression_results:
        sig = regression_results['named_condition_significant']
        print(f"Regression named_condition significant (p < {ALPHA})     : {sig}")

    print("\nPer-game Cohen's d (human, named vs aggregate):")
    for game, d in per_game_cohens_d_human.items():
        print(f"  {game}: {d:.4f}")

    print("\nLLM Cohen's d (named vs aggregate):")
    for at, d in llm_cohens_d.items():
        print(f"  {at}: {d:.4f}")

    return results


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results = run_exp3()
    print('Exp 3 complete. Results saved to results/')

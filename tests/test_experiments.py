import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data.generate_data import generate_exp1_data, generate_exp2_data, generate_exp3_data

_EXP1_REQUIRED_COLUMNS = [
    'game_id',
    'agent_type',
    'y',
    'y_strategic',
    'delta',
    'abs_delta',
    'standardized_delta',
    'paraphrase_id',
    'opponent_info',
    'framing',
]


def test_generate_exp1_data_schema():
    df = generate_exp1_data(n_human=20, n_llm=20)
    assert isinstance(df, pd.DataFrame), "generate_exp1_data should return a DataFrame"
    missing = [col for col in _EXP1_REQUIRED_COLUMNS if col not in df.columns]
    assert not missing, f"Missing columns in exp1 data: {missing}"


def test_generate_exp2_data_novelty_flag():
    df = generate_exp2_data(n_human=20, n_llm=20)
    assert isinstance(df, pd.DataFrame), "generate_exp2_data should return a DataFrame"
    assert 'novelty_certified' in df.columns, (
        "'novelty_certified' column missing from exp2 data"
    )
    assert df['novelty_certified'].all(), (
        "All rows in exp2 data should have novelty_certified=True"
    )


def test_generate_exp3_data_conditions():
    df = generate_exp3_data(n_human=20, n_llm=20)
    assert isinstance(df, pd.DataFrame), "generate_exp3_data should return a DataFrame"
    opponent_info_values = set(df['opponent_info'].unique())
    assert 'named' in opponent_info_values, (
        "exp3 data should contain 'named' condition in opponent_info"
    )
    assert 'aggregate' in opponent_info_values, (
        "exp3 data should contain 'aggregate' condition in opponent_info"
    )


def test_delta_computation():
    for name, df in [
        ('exp1', generate_exp1_data(n_human=20, n_llm=20)),
        ('exp2', generate_exp2_data(n_human=20, n_llm=20)),
        ('exp3', generate_exp3_data(n_human=20, n_llm=20)),
    ]:
        errors = np.abs(df['delta'] - (df['y'] - df['y_strategic']))
        assert (errors < 1e-9).all(), (
            f"{name}: delta != y - y_strategic for {(errors >= 1e-9).sum()} rows"
        )


def test_human_delta_nonzero():
    df = generate_exp1_data(n_human=50, n_llm=0)
    human_df = df[df['agent_type'] == 'human']
    canonical_games = human_df['game_id'].unique()
    for game_id in canonical_games:
        mean_abs_delta = human_df.loc[human_df['game_id'] == game_id, 'abs_delta'].mean()
        assert mean_abs_delta > 0.5, (
            f"Human mean |delta| for game '{game_id}' should be > 0.5, got {mean_abs_delta:.4f}"
        )


def test_llm_delta_smaller_than_human():
    df = generate_exp1_data(n_human=50, n_llm=50)
    canonical_games = df['game_id'].unique()

    for game_id in canonical_games:
        game_df = df[df['game_id'] == game_id]
        human_mean = game_df.loc[game_df['agent_type'] == 'human', 'abs_delta'].mean()
        llm_mean = game_df.loc[game_df['agent_type'] != 'human', 'abs_delta'].mean()
        assert human_mean > llm_mean, (
            f"Game '{game_id}': human mean |delta| ({human_mean:.4f}) should exceed "
            f"LLM mean |delta| ({llm_mean:.4f})"
        )

# Visualization for game theory structural tests
# Using matplotlib and seaborn for all plots

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

_AGENT_PALETTE = {
    'human': '#2196F3',
    'gpt-base': '#FF9800',
    'gpt-reasoning': '#9C27B0',
}
_DEFAULT_DPI = 150


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experiment 1 – delta distributions
# ---------------------------------------------------------------------------

def plot_delta_distributions(
    df: pd.DataFrame,
    output_path: str,
    title: str = 'Delta Distributions',
) -> None:
    """Facet KDE plot of delta per game_id, grouped by agent_type."""
    _ensure_dir(output_path)

    agent_types = df['agent_type'].unique().tolist()
    palette = {a: _AGENT_PALETTE.get(a, '#888888') for a in agent_types}

    game_ids = sorted(df['game_id'].unique())
    n_cols = min(4, len(game_ids))
    n_rows = int(np.ceil(len(game_ids) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    for idx, game_id in enumerate(game_ids):
        ax = axes_flat[idx]
        subset = df[df['game_id'] == game_id]
        for agent in agent_types:
            data = subset.loc[subset['agent_type'] == agent, 'delta']
            if len(data) >= 2:
                sns.kdeplot(
                    data,
                    ax=ax,
                    label=agent,
                    color=palette[agent],
                    fill=True,
                    alpha=0.35,
                )
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.9, alpha=0.7)
        ax.set_title(game_id, fontsize=10)
        ax.set_xlabel('δ (y − y*)')
        ax.set_ylabel('Density')

    # Hide unused axes
    for idx in range(len(game_ids), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    handles = [
        mpatches.Patch(color=palette[a], label=a, alpha=0.7) for a in agent_types
    ]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Structural test heatmap
# ---------------------------------------------------------------------------

def plot_structural_test_heatmap(
    results: dict,
    output_path: str,
    title: str = 'Structural Test Results',
) -> None:
    """Binary heatmap: rows=agent_type, cols=structural test outcomes.

    Args:
        results: Nested dict keyed by agent_type, each containing boolean
                 values for 'cond_dep_structured', 'asym_exceeds_threshold',
                 'path_dependent', and 'paraphrase_robust'.
    """
    _ensure_dir(output_path)

    cols = [
        'cond_dep_structured',
        'asym_exceeds_threshold',
        'path_dependent',
        'paraphrase_robust',
    ]
    col_labels = [
        'Cond. Dep.\nStructured',
        'Asym. Exceeds\nThreshold',
        'Path\nDependent',
        'Paraphrase\nRobust',
    ]
    agent_types = list(results.keys())

    matrix = np.array(
        [[int(results[a].get(col, False)) for col in cols] for a in agent_types],
        dtype=float,
    )

    cmap = mcolors.ListedColormap(['#E53935', '#43A047'])  # red=0, green=1
    fig, ax = plt.subplots(figsize=(len(cols) * 2, max(2, len(agent_types) * 1.2) + 1))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(agent_types)))
    ax.set_yticklabels(agent_types, fontsize=10)

    for i, agent in enumerate(agent_types):
        for j, col in enumerate(cols):
            val = results[agent].get(col, False)
            ax.text(j, i, '✓' if val else '✗', ha='center', va='center', fontsize=14,
                    color='white', fontweight='bold')

    legend_handles = [
        mpatches.Patch(color='#43A047', label='Passes threshold'),
        mpatches.Patch(color='#E53935', label='Fails threshold'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.35, 1.05))
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cohen's d bar chart
# ---------------------------------------------------------------------------

def plot_cohens_d_comparison(
    effect_sizes: dict,
    output_path: str,
    title: str = "Cohen's d by Game and Agent",
) -> None:
    """Bar chart of Cohen's d by game_id and agent_type.

    Args:
        effect_sizes: Dict (or DataFrame-convertible) with keys
                      'game_id', 'agent_type', 'cohens_d'.
    """
    _ensure_dir(output_path)

    df = pd.DataFrame(effect_sizes) if isinstance(effect_sizes, dict) else effect_sizes

    agent_types = df['agent_type'].unique().tolist()
    palette = {a: _AGENT_PALETTE.get(a, '#888888') for a in agent_types}

    fig, ax = plt.subplots(figsize=(max(8, len(df['game_id'].unique()) * 1.5), 5))
    sns.barplot(
        data=df,
        x='game_id',
        y='cohens_d',
        hue='agent_type',
        palette=palette,
        ax=ax,
        edgecolor='white',
        linewidth=0.5,
    )
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label="d = 0.5 (threshold)")
    ax.set_xlabel('Game')
    ax.set_ylabel("Cohen's d")
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='x', rotation=30)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Path dependence scatter
# ---------------------------------------------------------------------------

def plot_path_dependence(
    df: pd.DataFrame,
    output_path: str,
    title: str = 'Path Dependence: y_t vs y_{t-1}',
) -> None:
    """Scatter of y_t vs y_{t-1} with per-agent OLS regression line.

    Expects columns: agent_type, session_id, round_number, y.
    """
    _ensure_dir(output_path)

    df = (
        df.sort_values(['agent_type', 'session_id', 'round_number'])
        .copy()
    )
    df['y_lag'] = df.groupby(['agent_type', 'session_id'])['y'].shift(1)
    df = df.dropna(subset=['y_lag'])

    agent_types = sorted(df['agent_type'].unique())
    palette = {a: _AGENT_PALETTE.get(a, '#888888') for a in agent_types}

    fig, ax = plt.subplots(figsize=(7, 6))

    for agent in agent_types:
        sub = df[df['agent_type'] == agent]
        color = palette[agent]
        ax.scatter(sub['y_lag'], sub['y'], alpha=0.3, s=15, color=color, label=agent)

        # OLS regression line
        x_vals = sub['y_lag'].values
        y_vals = sub['y'].values
        if len(x_vals) >= 2:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
            ax.plot(x_range, np.polyval(coeffs, x_range), color=color, linewidth=2)

    ax.set_xlabel('y_{t−1}')
    ax.set_ylabel('y_t')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Individuation effect bar chart
# ---------------------------------------------------------------------------

def plot_individuation_effect(
    df: pd.DataFrame,
    output_path: str,
    title: str = 'Individuation Effect on |delta|',
) -> None:
    """Grouped bar chart of mean abs_delta by game_id × opponent_info.

    Expects columns: game_id, opponent_info, abs_delta, agent_type.
    One subplot per agent_type.
    """
    _ensure_dir(output_path)

    agent_types = sorted(df['agent_type'].unique())
    n_agents = len(agent_types)

    fig, axes = plt.subplots(
        1, n_agents,
        figsize=(6 * n_agents, 5),
        sharey=True,
        constrained_layout=True,
    )
    if n_agents == 1:
        axes = [axes]

    opponent_palette = {'named': '#1976D2', 'aggregate': '#EF6C00'}

    for ax, agent in zip(axes, agent_types):
        sub = df[df['agent_type'] == agent]
        agg = (
            sub.groupby(['game_id', 'opponent_info'], as_index=False)['abs_delta']
            .mean()
        )
        palette = {
            cond: opponent_palette.get(cond, '#888888')
            for cond in agg['opponent_info'].unique()
        }
        sns.barplot(
            data=agg,
            x='game_id',
            y='abs_delta',
            hue='opponent_info',
            palette=palette,
            ax=ax,
            edgecolor='white',
            linewidth=0.5,
        )
        ax.set_title(agent, fontsize=11)
        ax.set_xlabel('Game')
        ax.set_ylabel('Mean |δ|')
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment-level convenience functions
# ---------------------------------------------------------------------------

def generate_all_exp1_plots(
    df: pd.DataFrame,
    results: dict,
    results_dir: str = 'results',
) -> list:
    """Generate and save all Experiment 1 plots.

    Returns:
        List of saved file paths.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    delta_path = str(Path(results_dir) / 'exp1_delta_distributions.png')
    plot_delta_distributions(df, delta_path, title='Exp 1: Delta Distributions (Canonical Games)')
    saved.append(delta_path)

    heatmap_path = str(Path(results_dir) / 'exp1_structural_heatmap.png')
    plot_structural_test_heatmap(results, heatmap_path, title='Exp 1: Structural Test Results')
    saved.append(heatmap_path)

    return saved


def generate_all_exp2_plots(
    df: pd.DataFrame,
    results: dict,
    results_dir: str = 'results',
) -> list:
    """Generate and save all Experiment 2 plots.

    Returns:
        List of saved file paths.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    delta_path = str(Path(results_dir) / 'exp2_delta_distributions.png')
    plot_delta_distributions(df, delta_path, title='Exp 2: Delta Distributions (Novel Games)')
    saved.append(delta_path)

    return saved


def generate_all_exp3_plots(
    df: pd.DataFrame,
    results: dict,
    results_dir: str = 'results',
) -> list:
    """Generate and save all Experiment 3 plots.

    Returns:
        List of saved file paths.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    saved = []

    cohens_d_path = str(Path(results_dir) / 'exp3_cohens_d.png')
    plot_cohens_d_comparison(results, cohens_d_path, title="Exp 3: Cohen's d by Game and Agent")
    saved.append(cohens_d_path)

    individuation_path = str(Path(results_dir) / 'exp3_individuation_effect.png')
    plot_individuation_effect(df, individuation_path, title='Exp 3: Individuation Effect on |δ|')
    saved.append(individuation_path)

    return saved

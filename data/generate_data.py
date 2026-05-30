"""Data generation for Experiments 1, 2, and 3.

Experiment 1 – Canonical games
    5 canonical games × 5 paraphrases × 2 framing conditions × 2
    opponent-info conditions.  stag_hunt and public_goods are repeated
    over 5 rounds (path-dependent); all other games are one-shot.
    Human and two LLM agents (gpt-base, gpt-reasoning) are simulated.

Experiment 2 – Novel games
    3 novel games × 5 paraphrases × 2 framing conditions × 2
    opponent-info conditions, one-shot only.  novelty_certified=True is
    appended to every record.

Experiment 3 – Individuation gradient
    public_goods, stag_hunt, and beauty_contest × 2 opponent-info
    conditions × 5 repeated rounds, neutral framing, canonical paraphrase
    only.  Human and two LLM agents simulated.

All CSV files are written to the same directory as this script.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow execution from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import simulate_human_decision, simulate_llm_decision
from src.games import GameDef, get_all_games, get_canonical_games, get_game, get_novel_games

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FRAMINGS: list[str] = ["neutral", "generous"]
OPPONENT_INFOS: list[str] = ["aggregate", "named"]
N_PARAPHRASES: int = 5
LLM_MODELS: list[str] = ["gpt-base", "gpt-reasoning"]

MULTI_ROUND_GAMES: set[str] = {"stag_hunt", "public_goods"}
N_ROUNDS_MULTI: int = 5

EXP3_GAME_IDS: list[str] = ["public_goods", "stag_hunt", "beauty_contest"]

DATA_DIR: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _game_scale(game: GameDef) -> float:
    """Return the action-space range used to compute standardised delta."""
    return game.action_max - game.action_min


def _collect_condition(
    game: GameDef,
    n_human: int,
    n_llm: int,
    n_rounds: int,
    framing: str,
    opponent_info: str,
    paraphrase_id: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Simulate all agents for a single experimental condition.

    Rounds are played sequentially; the mean decision from round *r*
    becomes the ``prev_decision`` anchor for round *r+1* (path dependence).

    Parameters
    ----------
    game:
        Game definition.
    n_human:
        Observations per round for the human agent.
    n_llm:
        Observations per round for each LLM agent.
    n_rounds:
        Number of repeated rounds (1 for one-shot tasks).
    framing, opponent_info, paraphrase_id:
        Condition metadata forwarded to the simulators.
    rng:
        Shared NumPy random generator (mutated in place).

    Returns
    -------
    list[dict]
        Flat list of observation records from all agents across all rounds.
    """
    records: list[dict] = []

    # Per-agent previous-round decision anchors
    prev: dict[str, Optional[float]] = {
        "human": None,
        **{m: None for m in LLM_MODELS},
    }

    for round_idx in range(n_rounds):
        # Human
        h_recs = simulate_human_decision(
            game,
            n_obs=n_human,
            framing=framing,
            opponent_info=opponent_info,
            paraphrase_id=paraphrase_id,
            round_idx=round_idx,
            prev_decision=prev["human"],
            rng=rng,
        )
        prev["human"] = float(np.mean([r["y"] for r in h_recs]))
        records.extend(h_recs)

        # LLM agents
        for model_name in LLM_MODELS:
            l_recs = simulate_llm_decision(
                game,
                model_name=model_name,
                n_obs=n_llm,
                framing=framing,
                opponent_info=opponent_info,
                paraphrase_id=paraphrase_id,
                round_idx=round_idx,
                prev_decision=prev[model_name],
                rng=rng,
            )
            prev[model_name] = float(np.mean([r["y"] for r in l_recs]))
            records.extend(l_recs)

    return records


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def generate_exp1_data(
    n_human: int = 200,
    n_llm: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the Experiment 1 dataset (canonical games).

    Design
    ------
    * 5 canonical games
    * 5 paraphrase variants (paraphrase_id ∈ {0, …, 4})
    * 2 framing conditions (``'neutral'``, ``'generous'``)
    * 2 opponent-info conditions (``'aggregate'``, ``'named'``)
    * stag_hunt and public_goods: 5 repeated rounds; all others: 1 round
    * Human agent + gpt-base + gpt-reasoning

    Additional column
    -----------------
    standardized_delta
        ``delta / (action_max − action_min)`` – puts deviations from
        the strategic benchmark on a common 0-to-1 scale across games.

    Parameters
    ----------
    n_human:
        Observations per condition per round for the human agent.
    n_llm:
        Observations per condition per round for each LLM agent.
    seed:
        Master random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        All observations concatenated with columns: game_id, agent_type,
        framing, stake, opponent_info, paraphrase_id, round_idx, y,
        y_strategic, delta, abs_delta, standardized_delta.

    Side effect
    -----------
    Writes ``exp1_data.csv`` to the data directory.
    """
    rng = np.random.default_rng(seed)
    canonical_games = get_canonical_games()
    all_records: list[dict] = []

    for game in canonical_games:
        n_rounds = N_ROUNDS_MULTI if game.game_id in MULTI_ROUND_GAMES else 1
        scale = _game_scale(game)

        for paraphrase_id in range(N_PARAPHRASES):
            for framing in FRAMINGS:
                for opponent_info in OPPONENT_INFOS:
                    cond_records = _collect_condition(
                        game=game,
                        n_human=n_human,
                        n_llm=n_llm,
                        n_rounds=n_rounds,
                        framing=framing,
                        opponent_info=opponent_info,
                        paraphrase_id=paraphrase_id,
                        rng=rng,
                    )
                    for rec in cond_records:
                        rec["standardized_delta"] = rec["delta"] / scale
                    all_records.extend(cond_records)

    df = pd.DataFrame(all_records)
    out_path = DATA_DIR / "exp1_data.csv"
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------

def generate_exp2_data(
    n_human: int = 200,
    n_llm: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the Experiment 2 dataset (novel games).

    Design
    ------
    * 3 novel games (novel_threshold, novel_cascade, novel_fractional)
    * 5 paraphrase variants
    * 2 framing conditions
    * 2 opponent-info conditions
    * One-shot only (no repeated rounds)
    * Human agent + gpt-base + gpt-reasoning

    Additional column
    -----------------
    novelty_certified
        Boolean flag (always ``True``) confirming these are novel game
        structures not present in standard training corpora.

    Parameters
    ----------
    n_human:
        Observations per condition for the human agent.
    n_llm:
        Observations per condition for each LLM agent.
    seed:
        Master random seed.

    Returns
    -------
    pd.DataFrame
        All observations with columns: game_id, agent_type, framing, stake,
        opponent_info, paraphrase_id, round_idx, y, y_strategic, delta,
        abs_delta, novelty_certified.

    Side effect
    -----------
    Writes ``exp2_data.csv`` to the data directory.
    """
    rng = np.random.default_rng(seed)
    novel_games = get_novel_games()
    all_records: list[dict] = []

    for game in novel_games:
        for paraphrase_id in range(N_PARAPHRASES):
            for framing in FRAMINGS:
                for opponent_info in OPPONENT_INFOS:
                    cond_records = _collect_condition(
                        game=game,
                        n_human=n_human,
                        n_llm=n_llm,
                        n_rounds=1,
                        framing=framing,
                        opponent_info=opponent_info,
                        paraphrase_id=paraphrase_id,
                        rng=rng,
                    )
                    for rec in cond_records:
                        rec["novelty_certified"] = True
                    all_records.extend(cond_records)

    df = pd.DataFrame(all_records)
    out_path = DATA_DIR / "exp2_data.csv"
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------

def generate_exp3_data(
    n_human: int = 300,
    n_llm: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the Experiment 3 dataset (individuation gradient).

    Design
    ------
    * 3 games forming the individuation gradient:
      public_goods, stag_hunt, beauty_contest
    * 2 opponent-info conditions (``'aggregate'``, ``'named'``)
    * 5 repeated rounds per participant × game (path-dependent)
    * Neutral framing only, canonical paraphrase (paraphrase_id=0)
    * Human agent + gpt-base + gpt-reasoning

    Parameters
    ----------
    n_human:
        Observations per condition per round for the human agent.
    n_llm:
        Observations per condition per round for each LLM agent.
    seed:
        Master random seed.

    Returns
    -------
    pd.DataFrame
        All observations with columns: game_id, agent_type, framing, stake,
        opponent_info, paraphrase_id, round_idx, y, y_strategic, delta,
        abs_delta.

    Side effect
    -----------
    Writes ``exp3_data.csv`` to the data directory.
    """
    rng = np.random.default_rng(seed)
    exp3_games = [get_game(gid) for gid in EXP3_GAME_IDS]
    all_records: list[dict] = []

    for game in exp3_games:
        for opponent_info in OPPONENT_INFOS:
            cond_records = _collect_condition(
                game=game,
                n_human=n_human,
                n_llm=n_llm,
                n_rounds=N_ROUNDS_MULTI,
                framing="neutral",
                opponent_info=opponent_info,
                paraphrase_id=0,
                rng=rng,
            )
            all_records.extend(cond_records)

    df = pd.DataFrame(all_records)
    out_path = DATA_DIR / "exp3_data.csv"
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all three data generators and report dataset shapes."""
    print("Generating Experiment 1 data (canonical games) …")
    df1 = generate_exp1_data()
    print(f"  exp1_data shape: {df1.shape}")

    print("Generating Experiment 2 data (novel games) …")
    df2 = generate_exp2_data()
    print(f"  exp2_data shape: {df2.shape}")

    print("Generating Experiment 3 data (individuation gradient) …")
    df3 = generate_exp3_data()
    print(f"  exp3_data shape: {df3.shape}")


if __name__ == "__main__":
    main()

"""Agent simulators for human behavioral models and LLM baseline models.

Human agents are simulated using behavioural-economics primitives:
    * QRE-inspired noisy best response
    * Social preferences (inequality aversion, warm glow)
    * Framing and context sensitivity

LLM agents play near-Nash with light noise and weaker social-context
sensitivity than humans.

Both simulators share the same output schema so that downstream analysis
can treat observations uniformly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.games import GameDef


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    """Return *rng* as-is or create a fresh default generator."""
    return rng if rng is not None else np.random.default_rng()


def _build_records(
    game: GameDef,
    y_values: np.ndarray,
    agent_type: str,
    framing: str,
    stake: float,
    opponent_info: str,
    paraphrase_id: int,
    round_idx: int,
) -> list[dict]:
    """Assemble the output list of dicts from an array of decision values.

    Parameters
    ----------
    game:
        The game definition providing ``game_id`` and ``y_strategic``.
    y_values:
        Array of simulated decisions, one per observation.
    agent_type:
        ``'human'`` or a model name string.
    framing, stake, opponent_info, paraphrase_id, round_idx:
        Condition metadata written verbatim into every record.

    Returns
    -------
    list[dict]
        One dict per observation with keys: game_id, agent_type, framing,
        stake, opponent_info, paraphrase_id, round_idx, y, y_strategic,
        delta, abs_delta.
    """
    y_strategic = game.y_strategic
    records = []
    for y in y_values:
        delta = float(y) - y_strategic
        records.append(
            {
                "game_id": game.game_id,
                "agent_type": agent_type,
                "framing": framing,
                "stake": stake,
                "opponent_info": opponent_info,
                "paraphrase_id": paraphrase_id,
                "round_idx": round_idx,
                "y": float(y),
                "y_strategic": y_strategic,
                "delta": delta,
                "abs_delta": abs(delta),
            }
        )
    return records


def _apply_path_dependence(
    base_values: np.ndarray, prev_decision: float
) -> np.ndarray:
    """Blend *base_values* with a scalar previous decision.

    y = 0.4 × prev_decision + 0.6 × base_draw
    """
    return 0.4 * prev_decision + 0.6 * base_values


def _apply_paraphrase_noise(
    values: np.ndarray, paraphrase_id: int, rng: np.random.Generator
) -> np.ndarray:
    """Add tiny i.i.d. noise proportional to *paraphrase_id*.

    noise ~ N(0, paraphrase_id × 0.3)
    """
    if paraphrase_id == 0:
        return values
    std = paraphrase_id * 0.3
    return values + rng.normal(0.0, std, size=len(values))


# ---------------------------------------------------------------------------
# Human simulator
# ---------------------------------------------------------------------------

def simulate_human_decision(
    game: GameDef,
    n_obs: int,
    framing: str = "neutral",
    stake: float = 1.0,
    opponent_info: str = "aggregate",
    paraphrase_id: int = 0,
    round_idx: int = 0,
    prev_decision: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[dict]:
    """Simulate human decisions using behavioural-economics primitives.

    Behavioural properties encoded per game
    ----------------------------------------
    ultimatum
        Beta(3, 2) × 100, shifted +5 for 'generous' framing, −5 for
        'strategic' framing.  Humans typically offer ~40 units.
    public_goods
        N(10, 4) clipped to [0, 20].  Named-opponent condition adds +2
        (social-identity effect).
    stag_hunt
        Bernoulli with p_cooperate=0.65 at baseline; named opponent adds
        +0.10.
    travelers_dilemma
        N(75, 15) clipped to [2, 100].
    beauty_contest
        N(30, 10) clipped to [0, 100].
    novel games
        N(y_strategic + 8, 5) – positive bias above the strategic benchmark.

    Cross-cutting modifiers
    -----------------------
    Framing (ultimatum only)
        'generous' → +5; 'strategic' → −5.
    Named opponent
        public_goods: +2;  stag_hunt: +0.10 on p_cooperate.
    Paraphrase noise
        Additive N(0, paraphrase_id × 0.3) applied after game-specific draw.
    Path dependence
        y = 0.4 × prev_decision + 0.6 × base_draw when prev_decision is not None.

    Parameters
    ----------
    game:
        Game definition (provides id, action bounds, strategic benchmark).
    n_obs:
        Number of independent observations to generate.
    framing:
        Narrative framing condition: ``'neutral'``, ``'generous'``, or
        ``'strategic'``.
    stake:
        Payoff multiplier (recorded as metadata; does not alter the
        distribution in this simulator).
    opponent_info:
        ``'aggregate'`` (statistical summary) or ``'named'`` (identified
        individual).
    paraphrase_id:
        Index of the question paraphrase (0 = canonical wording).
    round_idx:
        Repetition index within a multi-round game (0-indexed).
    prev_decision:
        Scalar decision from the previous round; activates path dependence
        when not None.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    list[dict]
        One dict per observation; keys: game_id, agent_type, framing,
        stake, opponent_info, paraphrase_id, round_idx, y, y_strategic,
        delta, abs_delta.
    """
    rng = _make_rng(rng)
    gid = game.game_id
    lo, hi = game.action_min, game.action_max
    named = opponent_info == "named"

    # ------------------------------------------------------------------
    # Stag hunt: purely Bernoulli — handled separately
    # ------------------------------------------------------------------
    if gid == "stag_hunt":
        p_base = 0.65 + (0.10 if named else 0.0)
        if prev_decision is not None:
            p_base = 0.4 * prev_decision + 0.6 * p_base
        # Paraphrase nudges the probability slightly
        if paraphrase_id > 0:
            p_base += rng.normal(0.0, paraphrase_id * 0.3)
        p_cooperate = float(np.clip(p_base, 0.0, 1.0))
        y_values = rng.binomial(1, p_cooperate, size=n_obs).astype(float)
        return _build_records(
            game, y_values, "human", framing, stake,
            opponent_info, paraphrase_id, round_idx,
        )

    # ------------------------------------------------------------------
    # Continuous / clipped games
    # ------------------------------------------------------------------
    if gid == "ultimatum":
        framing_shift = {"generous": 5.0, "strategic": -5.0}.get(framing, 0.0)
        base = rng.beta(3.0, 2.0, size=n_obs) * 100.0 + framing_shift

    elif gid == "public_goods":
        named_shift = 2.0 if named else 0.0
        base = rng.normal(10.0 + named_shift, 4.0, size=n_obs)

    elif gid == "travelers_dilemma":
        base = rng.normal(75.0, 15.0, size=n_obs)

    elif gid == "beauty_contest":
        base = rng.normal(30.0, 10.0, size=n_obs)

    else:
        # Novel games: positive bias above strategic benchmark
        base = rng.normal(game.y_strategic + 8.0, 5.0, size=n_obs)

    # Cross-cutting modifiers
    base = _apply_paraphrase_noise(base, paraphrase_id, rng)
    if prev_decision is not None:
        base = _apply_path_dependence(base, prev_decision)

    y_values = np.clip(base, lo, hi)
    return _build_records(
        game, y_values, "human", framing, stake,
        opponent_info, paraphrase_id, round_idx,
    )


# ---------------------------------------------------------------------------
# LLM simulator
# ---------------------------------------------------------------------------

def simulate_llm_decision(
    game: GameDef,
    model_name: str = "gpt-base",
    n_obs: int = 100,
    framing: str = "neutral",
    stake: float = 1.0,
    opponent_info: str = "aggregate",
    paraphrase_id: int = 0,
    round_idx: int = 0,
    prev_decision: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[dict]:
    """Simulate LLM decisions as near-Nash responses with small noise.

    Behavioural properties encoded per game class
    ----------------------------------------------
    Canonical games
        N(y_strategic, 3) – tight cluster around the theoretical benchmark.
    Novel games
        N(y_strategic + 4, 8) – wider spread reflecting greater uncertainty
        about the equilibrium in unfamiliar game structures.

    Cross-cutting modifiers
    -----------------------
    Named opponent
        Adds only +0.3 (weaker social-context sensitivity than humans).
    Framing effect
        Additive N(0, 0.5) noise – the LLM is minimally affected by surface
        framing.
    Paraphrase noise
        Additive N(0, paraphrase_id × 0.3), same as the human simulator.
    Path dependence
        y = 0.4 × prev_decision + 0.6 × base_draw when prev_decision is not None.

    Parameters
    ----------
    game:
        Game definition.
    model_name:
        Identifier string written into every output record (e.g.
        ``'gpt-base'``, ``'gpt-reasoning'``).
    n_obs:
        Number of independent observations to generate.
    framing:
        Narrative framing condition.
    stake:
        Payoff multiplier (metadata only).
    opponent_info:
        ``'aggregate'`` or ``'named'``.
    paraphrase_id:
        Index of the question paraphrase.
    round_idx:
        Repetition index within a multi-round game.
    prev_decision:
        Scalar previous-round decision; activates path dependence.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    list[dict]
        One dict per observation; same schema as :func:`simulate_human_decision`.
    """
    rng = _make_rng(rng)
    gid = game.game_id
    lo, hi = game.action_min, game.action_max
    y_strategic = game.y_strategic

    # Noise parameters
    mean_shift = 4.0 if not game.canonical else 0.0
    spread = 8.0 if not game.canonical else 3.0
    named_shift = 0.3 if opponent_info == "named" else 0.0

    base_mean = y_strategic + mean_shift + named_shift
    framing_noise = rng.normal(0.0, 0.5, size=n_obs)

    # ------------------------------------------------------------------
    # Stag hunt: Bernoulli derived from Gaussian cooperation inclination
    # ------------------------------------------------------------------
    if gid == "stag_hunt":
        # Draw cooperation inclination from N(y_strategic, spread) then clip
        # to [0, 1] to form p_cooperate.  Near-Nash: y_strategic=0 → mostly defect.
        raw = rng.normal(base_mean, spread, size=n_obs) + framing_noise
        if prev_decision is not None:
            raw = _apply_path_dependence(raw, prev_decision)
        raw = _apply_paraphrase_noise(raw, paraphrase_id, rng)
        p_cooperate_vec = np.clip(raw, 0.0, 1.0)
        y_values = rng.binomial(1, p_cooperate_vec).astype(float)
        return _build_records(
            game, y_values, model_name, framing, stake,
            opponent_info, paraphrase_id, round_idx,
        )

    # ------------------------------------------------------------------
    # Continuous / clipped games
    # ------------------------------------------------------------------
    base = rng.normal(base_mean, spread, size=n_obs) + framing_noise
    base = _apply_paraphrase_noise(base, paraphrase_id, rng)
    if prev_decision is not None:
        base = _apply_path_dependence(base, prev_decision)

    y_values = np.clip(base, lo, hi)
    return _build_records(
        game, y_values, model_name, framing, stake,
        opponent_info, paraphrase_id, round_idx,
    )

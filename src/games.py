"""Game definitions for canonical and novel strategic games.

Each game is represented by a ``GameDef`` dataclass that captures the
game's identity, action space, strategic benchmark (Nash / SPE / BNE),
and a reference payoff function evaluated from a single player's
perspective.

Games
-----
Canonical
    ultimatum, public_goods, stag_hunt, travelers_dilemma, beauty_contest

Novel (bespoke payoff structures)
    novel_threshold, novel_cascade, novel_fractional
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GameDef:
    """Complete description of a strategic game for simulation and analysis.

    Parameters
    ----------
    game_id:
        Unique machine-readable identifier (e.g. ``'ultimatum'``).
    game_family:
        Broad family label used for grouping (e.g. ``'ultimatum'``,
        ``'public_goods'``, ``'coordination'``).
    canonical:
        ``True`` for textbook games with well-established empirical baselines;
        ``False`` for the novel bespoke games in Experiment 2.
    y_strategic:
        The theoretically optimal action under the stated equilibrium concept
        (Nash / SPE / BNE).  This is the value that a perfectly rational,
        self-interested agent should play.  Also accessible as ``benchmark``
        for compatibility with downstream analysis code.
    action_min:
        Lower bound of the continuous (or discrete) action space.
    action_max:
        Upper bound of the action space.
    description:
        Human-readable one-sentence description of the game.
    payoff_fn:
        Single-player payoff function ``f(y, **kwargs) -> float``.  The
        default assumes opponents play their Nash action.  Not used in
        equality comparisons so that ``GameDef`` instances remain hashable
        and comparable by their data fields alone.
    """

    game_id: str
    game_family: str
    canonical: bool
    y_strategic: float
    action_min: float
    action_max: float
    description: str
    payoff_fn: Optional[Callable[..., float]] = field(
        default=None, compare=False, repr=False
    )

    @property
    def benchmark(self) -> float:
        """Alias for ``y_strategic`` used by downstream analysis code."""
        return self.y_strategic

    @property
    def benchmark_y_strategic(self) -> float:
        """Backward-compatible alias for ``y_strategic``."""
        return self.y_strategic


# ---------------------------------------------------------------------------
# Game registry
# ---------------------------------------------------------------------------

def _ultimatum_payoff(y: float) -> float:
    """Proposer's payoff (keeps 100 − y) assuming the responder accepts."""
    return 100.0 - float(y)


def _public_goods_payoff(y: float, n: int = 4, mpcr: float = 0.4,
                         endowment: float = 20.0,
                         others_avg: float = 0.0) -> float:
    """Own payoff from contributing *y* when opponents contribute *others_avg*.

    payoff = MPCR × (y + (N−1) × others_avg) + (endowment − y)
    """
    return mpcr * (y + (n - 1) * others_avg) + (endowment - y)


def _stag_hunt_payoff(y: float, other: float = 0.0) -> float:
    """Expected payoff in the 2×2 stag hunt.

    Payoff matrix (row=self, col=other):  (C,C)=4, (D,D)=1, (C,D)=0, (D,C)=0.
    *y* and *other* are the cooperation probabilities ∈ {0, 1}.
    """
    y, other = float(y), float(other)
    return 4.0 * y * other + 1.0 * (1.0 - y) * (1.0 - other)


def _travelers_dilemma_payoff(y: float, other: float = 80.0,
                               bonus: float = 2.0) -> float:
    """Traveler's dilemma payoff.

    If *y* == *other*: both receive *y*.
    If *y* < *other*: claimant receives y + bonus; over-claimer receives y − bonus.
    If *y* > *other*: claimant receives other − bonus (self is the over-claimer).
    """
    y, other = float(y), float(other)
    if y == other:
        return y
    if y < other:
        return y + bonus
    return other - bonus


def _beauty_contest_payoff(y: float, avg: float = 33.33) -> float:
    """Negative distance from the 2/3-of-average target (lower is worse)."""
    return -abs(y - (2.0 / 3.0) * avg)


def _novel_threshold_payoff(y: float, other: float = 30.0,
                             prize: float = 30.0,
                             threshold: float = 60.0,
                             cost_rate: float = 0.5) -> float:
    """Payoff in the asymmetric threshold effort game.

    payoff = prize  if  y + other >= threshold  else 0  −  cost_rate × y
    """
    y, other = float(y), float(other)
    prize_term = prize if (y + other) >= threshold else 0.0
    return prize_term - cost_rate * y


def _novel_cascade_payoff(y: float, bne: float = 50.0) -> float:
    """Stylised cascade-auction payoff: quadratic loss around BNE bid of 50."""
    return -((float(y) - bne) ** 2) / 100.0


def _novel_fractional_payoff(y: float, noise: float = 0.0) -> float:
    """Payoff in the fractional coordination game.

    payoff = −(y − 0.618)² + noise
    """
    return -((float(y) - 0.618) ** 2) + noise


_GAME_REGISTRY: dict[str, GameDef] = {
    # ------------------------------------------------------------------
    # Canonical games
    # ------------------------------------------------------------------
    "ultimatum": GameDef(
        game_id="ultimatum",
        game_family="ultimatum",
        canonical=True,
        y_strategic=0.0,
        action_min=0.0,
        action_max=100.0,
        description=(
            "Proposer splits 100 units; y is the share offered to the "
            "responder.  SPE predicts y=0; humans typically offer ~40."
        ),
        payoff_fn=_ultimatum_payoff,
    ),
    "public_goods": GameDef(
        game_id="public_goods",
        game_family="public_goods",
        canonical=True,
        y_strategic=0.0,
        action_min=0.0,
        action_max=20.0,
        description=(
            "N=4 public-goods game with endowment 20 and MPCR=0.4.  "
            "y is own contribution.  Nash predicts y=0; humans contribute 8-12."
        ),
        payoff_fn=_public_goods_payoff,
    ),
    "stag_hunt": GameDef(
        game_id="stag_hunt",
        game_family="coordination",
        canonical=True,
        y_strategic=0.0,
        action_min=0.0,
        action_max=1.0,
        description=(
            "2×2 coordination game; y∈{0,1} where 1=cooperate.  "
            "Risk-dominant Nash is y=0 (defect); humans cooperate ~65% of the time."
        ),
        payoff_fn=_stag_hunt_payoff,
    ),
    "travelers_dilemma": GameDef(
        game_id="travelers_dilemma",
        game_family="traveler",
        canonical=True,
        y_strategic=2.0,
        action_min=2.0,
        action_max=100.0,
        description=(
            "Traveler's dilemma with claim range [2, 100].  "
            "Nash predicts y=2; humans typically claim 80-90."
        ),
        payoff_fn=_travelers_dilemma_payoff,
    ),
    "beauty_contest": GameDef(
        game_id="beauty_contest",
        game_family="beauty_contest",
        canonical=True,
        y_strategic=0.0,
        action_min=0.0,
        action_max=100.0,
        description=(
            "p=2/3 beauty-contest / Keynesian guessing game, range [0, 100].  "
            "Nash predicts y=0; level-k (k=2) predicts ~22."
        ),
        payoff_fn=_beauty_contest_payoff,
    ),
    # ------------------------------------------------------------------
    # Novel games
    # ------------------------------------------------------------------
    "novel_threshold": GameDef(
        game_id="novel_threshold",
        game_family="novel_threshold",
        canonical=False,
        y_strategic=30.0,
        action_min=0.0,
        action_max=50.0,
        description=(
            "Two-player asymmetric threshold effort game; y=effort ∈ [0,50].  "
            "Payoff=30 if e1+e2≥60 else 0, minus 0.5×e.  Symmetric Nash: y=30."
        ),
        payoff_fn=_novel_threshold_payoff,
    ),
    "novel_cascade": GameDef(
        game_id="novel_cascade",
        game_family="novel_auction",
        canonical=False,
        y_strategic=50.0,
        action_min=0.0,
        action_max=100.0,
        description=(
            "Sequential cascade auction; y=bid ∈ [0, 100].  "
            "BNE predicts y=50."
        ),
        payoff_fn=_novel_cascade_payoff,
    ),
    "novel_fractional": GameDef(
        game_id="novel_fractional",
        game_family="novel_coord",
        canonical=False,
        y_strategic=0.618,
        action_min=0.0,
        action_max=1.0,
        description=(
            "Fractional coordination game; y ∈ [0, 1].  "
            "Payoff=−(y−0.618)²+noise.  Strategic optimum: y=0.618."
        ),
        payoff_fn=_novel_fractional_payoff,
    ),
}


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_game(game_id: str) -> GameDef:
    """Return the ``GameDef`` for the given *game_id*.

    Parameters
    ----------
    game_id:
        One of the registered game identifiers.

    Returns
    -------
    GameDef
        The corresponding game definition.

    Raises
    ------
    KeyError
        If *game_id* is not in the registry.
    """
    if game_id not in _GAME_REGISTRY:
        raise KeyError(
            f"Unknown game_id {game_id!r}.  "
            f"Available: {sorted(_GAME_REGISTRY)}"
        )
    return _GAME_REGISTRY[game_id]


def get_canonical_games() -> list[GameDef]:
    """Return all canonical game definitions in registration order.

    Returns
    -------
    list[GameDef]
        The five canonical game definitions (ultimatum, public_goods,
        stag_hunt, travelers_dilemma, beauty_contest).
    """
    return [g for g in _GAME_REGISTRY.values() if g.canonical]


def get_novel_games() -> list[GameDef]:
    """Return all novel (non-canonical) game definitions.

    Returns
    -------
    list[GameDef]
        The three novel game definitions (novel_threshold, novel_cascade,
        novel_fractional).
    """
    return [g for g in _GAME_REGISTRY.values() if not g.canonical]


def get_all_games() -> list[GameDef]:
    """Return every registered game definition.

    Returns
    -------
    list[GameDef]
        All eight game definitions in registration order.
    """
    return list(_GAME_REGISTRY.values())

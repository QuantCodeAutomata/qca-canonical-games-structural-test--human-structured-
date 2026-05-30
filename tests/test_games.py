import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games import get_game, get_canonical_games, get_novel_games, get_all_games


def test_get_canonical_games():
    games = get_canonical_games()
    assert len(games) >= 5, f"Expected at least 5 canonical games, got {len(games)}"
    for g in games:
        assert g.canonical is True, f"Game {g.game_id} should have canonical=True"
        assert isinstance(g.benchmark, (int, float)), (
            f"Game {g.game_id} benchmark should be numeric, got {type(g.benchmark)}"
        )


def test_get_novel_games():
    games = get_novel_games()
    assert len(games) >= 3, f"Expected at least 3 novel games, got {len(games)}"
    for g in games:
        assert g.canonical is False, f"Game {g.game_id} should have canonical=False"


def test_game_benchmark_range():
    for g in get_all_games():
        assert g.action_min <= g.y_strategic <= g.action_max, (
            f"Game {g.game_id}: y_strategic={g.y_strategic} not in "
            f"[{g.action_min}, {g.action_max}]"
        )


def test_game_id_uniqueness():
    all_games = get_all_games()
    ids = [g.game_id for g in all_games]
    assert len(ids) == len(set(ids)), "Duplicate game_ids found"


def test_get_game_returns_correct():
    game = get_game('ultimatum')
    assert game.game_id == 'ultimatum', (
        f"Expected game_id 'ultimatum', got '{game.game_id}'"
    )

"""random_baseline の最小健全性テスト（エンジン実体が必要）。

実行: poetry run pytest -q
cg 共有ライブラリ (libcg.so 等) がロードできる環境でのみ通る。
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"


def test_deck_is_legal():
    from ptcg import engine
    from ptcg.cards import read_deck_csv

    deck = read_deck_csv(AGENTS_DIR / "random_baseline" / "deck.csv")
    assert len(deck) == 60
    ok, err = engine.check_deck(deck)
    assert ok, f"deck illegal: errorType={err}"


def test_agent_loads_without_file_trap():
    from ptcg import engine

    agent = engine.load_agent(AGENTS_DIR / "random_baseline" / "main.py")
    assert callable(agent)


def test_agent_survives_a_game():
    from ptcg import engine
    from ptcg.agents.base import RandomAgent
    from ptcg.cards import read_deck_csv

    deck = read_deck_csv(AGENTS_DIR / "random_baseline" / "deck.csv")
    agent = engine.load_agent(AGENTS_DIR / "random_baseline" / "main.py")
    result = engine.play_game(agent, RandomAgent(deck), deck, deck)
    assert result in (0, 1, 2)


def test_estimator_classifies_known_decks():
    from ptcg.cards import read_deck_csv
    from ptcg.meta import classify

    decks_dir = Path(__file__).resolve().parents[1] / "decks"
    cases = {
        "sample_abomasnow": "abomasnow",
        "archaludon_cinderace": "archaludon",
        "greattusk_crustle_lo": "crustle_lo",
    }
    for stem, truth in cases.items():
        ids = set(read_deck_csv(decks_dir / f"{stem}.csv"))
        guess = classify(ids)
        assert guess.name == truth, (stem, guess.name, guess.scores)
        assert guess.confident, (stem, guess.confidence)


def test_vendored_archetypes_in_sync():
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "src" / "ptcg" / "meta" / "archetypes.py").read_text(encoding="utf-8")
    vend = (repo / "agents" / "meta_aware" / "archetypes.py").read_text(encoding="utf-8")
    body = vend.split("\n", 1)[1] if vend.startswith("# AUTO-VENDORED") else vend
    assert body == src, "agents/meta_aware/archetypes.py が古い。`poetry run python scripts/vendor.py meta_aware` を再実行"


def test_pool_eval_runs():
    from ptcg import engine
    from ptcg.cards import read_deck_csv
    from ptcg.eval import default_pool, evaluate_pool

    agent = engine.load_agent(AGENTS_DIR / "greedy_first" / "main.py")
    deck = read_deck_csv(AGENTS_DIR / "greedy_first" / "deck.csv")
    pool = default_pool()
    assert len(pool) >= 1
    res = evaluate_pool("greedy_first", pool, games_per_opponent=1, agent=agent, agent_deck=deck)
    assert len(res.members) == len(pool)
    assert res.aggregate.games == len(pool)

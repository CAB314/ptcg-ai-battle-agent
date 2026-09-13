"""style.py / deck_metrics.py の健全性テスト（エンジン実体が必要）。

実行: poetry run pytest -q
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _lookups():
    from ptcg.engine import engine_attacks, engine_card_data

    cards = {c.cardId: c for c in engine_card_data()}
    attacks = {a.attackId: a for a in engine_attacks()}
    return cards, attacks


def test_lo_deck_classified_as_control():
    from ptcg.cards import read_deck_csv
    from ptcg.meta.style import classify_style

    deck = read_deck_csv(REPO / "decks" / "greattusk_crustle_lo.csv")
    cards, attacks = _lookups()
    label, f = classify_style(deck, cards, attacks)
    assert label == "control"
    assert f.n_mill >= 4  # Great Tusk x4 の Land Collapse（Unicodeアポストロフィ正規化の回帰テスト）
    assert f.n_disrupt >= 4  # Xerosic x4


def test_mill_keyword_detection_survives_unicode_apostrophe():
    from ptcg.meta.style import style_features

    cards, attacks = _lookups()
    f = style_features([58], cards, attacks)  # Great Tusk 単体
    assert f.n_mill == 1


def test_deck_math_formulas():
    from deck_metrics import mulligan_rate, opener_access, prized_rate

    # 織田尭: 1枚差しの初手アクセス=11.67%
    assert abs(opener_access(1) - 0.1167) < 0.001
    # lastlegume: 1枚差しのサイド落ち=ちょうど10%
    assert abs(prized_rate(1) - 0.10) < 1e-9
    # JustInBasil Appendix4: たね9枚のマリガン率 ≈ 30%
    assert abs(mulligan_rate(9) - 0.2998) < 0.001

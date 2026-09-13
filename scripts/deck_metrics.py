"""デッキ数学メトリクス + 配分バリデータ（docs/ptcg-usable-excerpts.md #3, #4 の実装）。

数式はすべて超幾何分布（出典: JustInBasil Appendix4 / TheMathTCG / 織田尭 / エイル / lastlegume）:
- マリガン率 = C(60−B,7)/C(60,7)（B=たね数）
- 初手アクセス率 P(≥1 in 7) = 1 − C(60−K,7)/C(60,7)（K=投入枚数）
- サイド落ち率 P(≥1 prized) = 1 − C(60−K,6)/C(60,6)（1枚差し=10%）・期待落ち枚数 = 0.1K

バリデータ（足切り/警告の目安）:
- たね ≥6（エイル: マリガン期待値<1）・マリガン率 ≲10%
- 進化ライン ≤2（ポケカ族: 事故りにくい）
- 基本エネ 8–14（Smogon。特殊エネ主体デッキは情報表示のみ）

使い方:
    poetry run python scripts/deck_metrics.py                 # decks/*.csv 全部
    poetry run python scripts/deck_metrics.py decks/iono.csv agents/greattusk_lo_playbook/deck.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ptcg.cards import id_to_name, read_deck_csv  # noqa: E402
from ptcg.engine import engine_attacks, engine_card_data  # noqa: E402
from ptcg.meta.style import classify_style  # noqa: E402


def mulligan_rate(n_basics: int) -> float:
    return comb(60 - n_basics, 7) / comb(60, 7) if n_basics < 60 else 0.0


def opener_access(k: int) -> float:
    return 1.0 - comb(60 - k, 7) / comb(60, 7)


def prized_rate(k: int) -> float:
    return 1.0 - comb(60 - k, 6) / comb(60, 6)


def analyze(path: Path, cards: dict, attacks: dict, names: dict) -> None:
    deck = read_deck_csv(path)
    counts = Counter(deck)
    basics = [cid for cid in deck if getattr(cards.get(cid), "basic", False)]
    n_basic_energy = sum(1 for cid in deck if int(getattr(cards.get(cid), "cardType", -1)) == 5)
    n_special_energy = sum(1 for cid in deck if int(getattr(cards.get(cid), "cardType", -1)) == 6)
    n_pokemon = sum(1 for cid in deck if int(getattr(cards.get(cid), "cardType", -1)) == 0)
    label, f = classify_style(deck, cards, attacks)
    mull = mulligan_rate(len(basics))

    print(f"\n=== {path} ===")
    print(
        f"  構成: ポケ{n_pokemon} (たね{f.n_basic}/1進化{f.n_stage1}/2進化{f.n_stage2}, "
        f"ライン{f.n_evo_lines}, ルールボックス{f.n_rule_box}) / エネ{f.n_energy} "
        f"(基本{n_basic_energy}+特殊{n_special_energy}) / トレーナー{60 - n_pokemon - f.n_energy}"
    )
    print(
        f"  スタイル: {label} (control={f.scores.get('control', 0):.1f}, aggro={f.scores.get('aggro', 0):.1f}, "
        f"平均打点{f.avg_attacker_damage:.0f}/最大{f.top_damage}, 妨害{f.n_disrupt}/ミル{f.n_mill}/呼出{f.n_gust}/回復{f.n_heal})"
    )
    exp_mull = mull / (1.0 - mull) if mull < 1.0 else float("inf")
    print(f"  マリガン率: {mull:6.2%}  (たね {len(basics)} 枚, 期待マリガン回数 {exp_mull:.2f})")

    warns = []
    if len(basics) < 6:
        warns.append(f"たね{len(basics)}枚 <6（エイル基準: マリガン期待値>1）")
    if exp_mull >= 1.0:
        warns.append(f"期待マリガン回数{exp_mull:.2f} ≥1（エイル基準の足切り）")
    if f.n_evo_lines > 2:
        warns.append(f"進化ライン{f.n_evo_lines} >2（初動不成立の分散大）")
    if n_basic_energy > 0 and not 8 <= n_basic_energy <= 14:
        warns.append(f"基本エネ{n_basic_energy}枚（目安8–14）")
    if n_basic_energy == 0 and n_special_energy < 8:
        warns.append(f"エネ計{n_special_energy}枚（特殊のみ・少なめ）: エネ供給の再現性に注意")
    for w in warns:
        print(f"  [warn] {w}")
    if not warns:
        print("  [ok] 配分バリデータ全通過")

    ones = sorted(cid for cid, k in counts.items() if k == 1)
    if ones:
        print("  1枚差しのサイド落ちリスク（P=10%/枚）:")
        for cid in ones:
            print(f"    - [{cid}] {names.get(cid, '?')}")
    print("  初手アクセス率（枚数別）: " + ", ".join(
        f"{k}枚={opener_access(k):.1%}" for k in sorted({v for v in counts.values()})
    ))
    print(f"  サイド落ち期待: 4枚投入→期待{0.1 * 4:.1f}枚落ち / P(≥1)={prized_rate(4):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decks", nargs="*", help="deck.csv のパス（省略時は decks/*.csv）")
    args = ap.parse_args()
    paths = [Path(p) for p in args.decks] or sorted((REPO_ROOT / "decks").glob("*.csv"))
    cards = {c.cardId: c for c in engine_card_data()}
    attacks = {a.attackId: a for a in engine_attacks()}
    names = id_to_name()
    for p in paths:
        analyze(p, cards, attacks, names)


if __name__ == "__main__":
    main()

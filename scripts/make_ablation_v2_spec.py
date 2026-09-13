#!/usr/bin/env python
"""現メタ加重のデッキアブレーション候補スペックを作る（2026-08-05）。

候補は勘で決めず、**上位 TR Mewtwo パイロット3名の最頻リストとの差分**から作る。
3名（THIRD PTCG Club 58.1% / flg 56.4% / kashiwashira 55.4%）が全員一致で
やっている変更＝強い事前分布。単一カード単位で分離した候補と、
上位3名のリスト丸ごとの候補（上限の目安）を両方入れる。

最重要仮説: **Battle Cage(1264)** =「ベンチにダメカンを置かせない」スタジアム。
Grimmsnarl ex の Shadow Bullet（180＋ベンチ30）と Froslass の特性チップを封じる。
Grimmsnarl は現メタ 38.1% で我々の最悪対面（実戦 32.8%）。上位3名中2名が採用。

出力: docs/data/ablation_v2_candidates.json
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

BASE_DECK = REPO_ROOT / "agents/trmewtwo_ppo2/deck.csv"
OUT = REPO_ROOT / "docs/data/ablation_v2_candidates.json"

# 上位3名との差分（実測。scripts の解析結果より）。+は増やす、-は減らす。
CONSENSUS = {          # 3名全員一致
    5: -3,    # Basic {P} Energy
    434: +2,  # TR Mimikyu（70HP以下のたね＝Poffinで釣れる。TR4体条件を早く満たす）
    1227: +2, # Lillie's Determination（手札シャッフル→6ドロー）
    463: -2,  # TR Murkrow
    1119: -2, # Energy Search
    431: -1,  # TR Mewtwo ex 3→2（**我々のアブレーションの結論と逆**）
    432: -1,  # TR Wobbuffet
    1086: +1, # Buddy-Buddy Poffin
    1121: +2, # Ultra Ball
}


def apply(base: Counter, delta: dict[int, int]) -> list[int]:
    d = Counter(base)
    for cid, n in delta.items():
        d[cid] = max(0, d.get(cid, 0) + n)
        if d[cid] == 0:
            del d[cid]
    return sorted(c for c, n in d.items() for _ in range(n))


def main() -> int:
    base_l = [int(x) for x in BASE_DECK.read_text().split()]
    base = Counter(base_l)
    assert sum(base.values()) == 60, sum(base.values())

    names = {}
    for r in csv.DictReader(open(REPO_ROOT / "data/EN_Card_Data.csv")):
        names.setdefault(int(r["Card ID"]), r["Card Name"])

    cands: list[dict] = []

    def add(name, delta, note):
        deck = apply(base, delta)
        if len(deck) != 60:
            print(f"  ★{name}: {len(deck)}枚になったのでスキップ")
            return
        over = [c for c, n in Counter(deck).items() if n > 4 and c > 20]
        if over:
            print(f"  ★{name}: 4枚超過 {[(c, names.get(c)) for c in over]} → スキップ")
            return
        cands.append({"name": name, "deck": deck, "note": note})

    # --- 最重要仮説: 対 Grimmsnarl(38.1%) の Battle Cage ---
    add("mw_cage3", {1264: +3, 5: -3},
        "Battle Cage x3 / 基本超-3。ベンチへのダメカンを封じ Shadow Bullet(180+ベンチ30) と Froslass を無効化")
    add("mw_cage2_factory0", {1264: +2, 1257: -2},
        "Battle Cage x2 / TR Factory-2。スタジアム枠の入れ替え（上位2名がこの方向）")

    # --- エンジン加速（Mewtwo ex は TR4体で初めて攻撃できる）---
    add("mw_engine", {434: +2, 1086: +1, 463: -2, 432: -1},
        "Mimikyu+2 Poffin+1 / Murkrow-2 Wobbuffet-1。TR4体条件を早く満たす")
    add("mw_ultraball2", {1121: +2, 5: -2}, "Ultra Ball+2 / 基本超-2。サーチのみ分離")
    add("mw_lillie2", {1227: +2, 1119: -2}, "Lillie's Determination+2 / Energy Search-2。ドローのみ分離")

    # --- 我々の過去の結論との衝突を検証 ---
    add("mw_mewtwo2", {431: -1, 5: +1},
        "Mewtwo ex 3→2。**上位3名全員が2枚**。我々の7/31アブレーションは3枚を+1.06ppと結論しており矛盾する")

    # --- 合意パッケージ全部（増減差 -2 を Battle Cage で埋める）---
    add("mw_consensus", {**CONSENSUS, 1264: +2},
        "上位3名の全員一致の変更をすべて適用（枚数差 -2 は Battle Cage で埋める）")
    add("mw_consensus_cage", {**CONSENSUS, 1264: +4, 1257: -2},
        "合意パッケージ + Battle Cage x4 / TR Factory-2（上位2名の構成に最も近い）")

    # --- 上位パイロットのリストを丸ごと（到達可能な上限の目安）---
    for team, delta in (
        ("third", {5: -3, 1264: +3, 434: +2, 1227: +2, 463: -2, 1119: -2, 1121: +2, 1257: -2,
                   1: +1, 1175: +1, 431: -1, 432: -1, 1086: +1, 1217: -1, 1218: -1, 1097: +1}),
        ("flg", {5: -3, 434: +3, 1086: +2, 1220: +2, 1227: +2, 463: -2, 1119: -2, 431: -1,
                 432: -1, 1219: -1, 1097: -1, 1121: +1, 1257: +1}),
    ):
        add(f"mw_{team}_full", delta,
            f"上位パイロット {team} の最頻リストをそのまま（差分適用）")

    field = json.loads((REPO_ROOT / "docs/data/ablation_v2_field.json").read_text())

    spec = {
        "comment": ("2026-08-05。基準=現行ラダー主力 trmewtwo_ppo2 のデッキ。"
                    "方策も同一（agents/trmewtwo_ppo2 を配備経路で読む）。"
                    "相手プールは 8/3 実測メタ加重（カバー率95.5%）。"
                    "候補は上位パイロット3名との差分から設計。"),
        "base": {"name": "trmewtwo_ppo2 現行(942.7)", "deck": base_l},
        "candidates": cands,
        "field": field,
    }
    OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n候補 {len(cands)} 件:")
    for c in cands:
        print(f"  {c['name']:20s} {c['note'][:78]}")
    print(f"\n[ok] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

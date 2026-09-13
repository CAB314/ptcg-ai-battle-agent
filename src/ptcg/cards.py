"""カードデータの読み込みユーティリティ。

配布 CSV (EN_Card_Data.csv / JP_Card_Data.csv) と、エンジン由来の構造化データ
(engine.engine_card_data) の両方を扱う。CSV は人間可読・全カード一覧向き、
エンジンデータは attacks/skills/ex フラグ等のロジック向き。
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from .engine import REPO_ROOT

CARD_ID_COL = "Card ID"
CARD_NAME_COL = "Card Name"


def card_csv_path(lang: str = "EN") -> Path:
    path = REPO_ROOT / f"data/{lang}_Card_Data.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} がありません（配布データ未展開？）")
    return path


@lru_cache(maxsize=None)
def load_card_rows(lang: str = "EN") -> tuple[dict, ...]:
    """CSV の各行を dict のタプルで返す（lru_cache のため immutable）。"""
    with open(card_csv_path(lang), encoding="utf-8") as f:
        return tuple(dict(row) for row in csv.DictReader(f))


@lru_cache(maxsize=None)
def id_to_name(lang: str = "EN") -> dict:
    out: dict[int, str] = {}
    for row in load_card_rows(lang):
        try:
            out[int(row[CARD_ID_COL])] = row[CARD_NAME_COL]
        except (KeyError, ValueError):
            continue
    return out


@lru_cache(maxsize=None)
def name_to_id(lang: str = "EN") -> dict:
    return {name: cid for cid, name in id_to_name(lang).items()}


def read_deck_csv(path: str | Path) -> list[int]:
    """deck.csv を int のリストで読む（空行は無視）。"""
    text = Path(path).read_text(encoding="utf-8")
    return [int(line) for line in text.splitlines() if line.strip()]


def describe_deck(path: str | Path, lang: str = "EN") -> str:
    """デッキを『枚数 x カード名 (ID)』の一覧に整形する（目視確認用）。"""
    from collections import Counter

    names = id_to_name(lang)
    counts = Counter(read_deck_csv(path))
    lines = [f"{sum(counts.values())} cards total"]
    for cid, c in counts.most_common():
        lines.append(f"  {c:>2} x {names.get(cid, '?')} ({cid})")
    return "\n".join(lines)

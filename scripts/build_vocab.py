#!/usr/bin/env python
"""カード/ワザ静的テーブル cards.npz / attacks.npz を生成する。

エンジン（cg）から ptcg.ml.vocab.build_tables で抽出し、data/ml/ に保存する。
学習・PPO actor・提出エージェントは全てこの npz を読む（特徴量パスの cg 非依存化）。
エンジン更新（カード追加等）の際は再生成して VOCAB_VERSION を上げること。

使い方:
    poetry run python scripts/build_vocab.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from ptcg.ml.vocab import build_tables, load_tables, save_tables

OUT_DIR = REPO_ROOT / "data" / "ml"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cards_path = OUT_DIR / "cards.npz"
    attacks_path = OUT_DIR / "attacks.npz"
    tables = build_tables()
    save_tables(tables, cards_path, attacks_path)
    # 読み戻し検証（numpy のみ経路）
    loaded = load_tables(cards_path, attacks_path)
    assert loaded is not None, "load_tables が None を返した"
    n_cards = int(loaded["cards"][:, -1].sum())
    n_atks = int(loaded["atk_known"].sum())
    print(f"[ok] {cards_path} ({n_cards} cards) / {attacks_path} ({n_atks} attacks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

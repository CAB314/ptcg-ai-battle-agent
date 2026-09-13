#!/usr/bin/env python
"""Daily Top Episodes から実メタの正確な60枚デッキを抽出する。

各 episode JSON の steps[0][0].visualize[0].action == [P0の60枚, P1の60枚] に
両者の完全デッキが入っている（カードID）。これをアーキタイプ別に集計し、各上位
アーキタイプの **最頻(modal)デッキリスト** を decks/meta_<slug>.csv に書き出す。
名前→ID変換不要・全カードIDが揃うので合法性も担保できる。

アーキタイプ判定 = 「2枚以上採用のポケモンのうち最高HP（Mega>Stage2>Stage1>Basic でタイブレーク）」
をエースとし、その名前をラベル化（上位ログ解析と同じヒューリスティック）。

使い方:
    poetry run python scripts/mine_decklists.py --episodes-dir data/episodes/2026-07-13 --top 6
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from ptcg import engine
from ptcg.eval.harness import wilson_interval

DECKS_DIR = REPO_ROOT / "decks"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--top", type=int, default=6, help="書き出す上位アーキタイプ数")
    ap.add_argument("--min-games", type=int, default=50, help="採用する最小試合数")
    ap.add_argument("--prefix", default="meta_", help="出力デッキ名の接頭辞")
    args = ap.parse_args()

    cd = {c.cardId: c for c in engine.engine_card_data()}

    def is_pokemon(cid):
        c = cd.get(cid)
        return c is not None and getattr(c, "cardType", None) == 0

    def ace_label(deck):
        # 2枚以上のポケモンのうち最高HP（Mega>Stage2>Stage1>Basic, 同点はid）
        counts = Counter(deck)
        best = None
        for cid, n in counts.items():
            if n < 2 or not is_pokemon(cid):
                continue
            c = cd[cid]
            key = (c.hp or 0, bool(c.megaEx), bool(c.stage2), bool(c.stage1), cid)
            if best is None or key > best[0]:
                best = (key, cid)
        if best is None:
            return "no_ace"
        return cd[best[1]].name

    files = sorted(glob.glob(f"{args.episodes_dir}/**/*.json", recursive=True))
    if not files:
        print(f"no episode json under {args.episodes_dir}", file=sys.stderr)
        return 1
    print(f"parsing {len(files)} episodes ...")

    lists_by_arch: dict[str, Counter] = defaultdict(Counter)  # arch -> Counter(deck tuple)
    games = Counter()
    wins = Counter()
    for k, f in enumerate(files, 1):
        if k % 1000 == 0:
            print(f"  {k}/{len(files)}")
        try:
            d = json.load(open(f, encoding="utf-8"))
            action = d["steps"][0][0]["visualize"][0]["action"]
            rew = d.get("rewards") or []
        except Exception:
            continue
        if len(action) != 2 or len(rew) != 2 or rew[0] is None or rew[1] is None or rew[0] == rew[1]:
            continue
        winner = 0 if rew[0] > rew[1] else 1
        for seat in (0, 1):
            deck = [int(x) for x in action[seat]]
            if len(deck) != 60:
                continue
            arch = ace_label(deck)
            lists_by_arch[arch][tuple(sorted(deck))] += 1
            games[arch] += 1
            if seat == winner:
                wins[arch] += 1

    print("\n=== top archetypes (deck-slot share) ===")
    ranked = games.most_common()
    written = 0
    for arch, n in ranked:
        if arch in ("no_ace",) or n < args.min_games:
            continue
        p, lo, hi = wilson_interval(wins[arch], n)
        modal_deck, modal_n = lists_by_arch[arch].most_common(1)[0]
        n_distinct = len(lists_by_arch[arch])
        print(f"  {arch[:34]:34s} games={n:5d} wr={p*100:4.1f}% "
              f"[{lo*100:.0f}-{hi*100:.0f}]  modal-list={modal_n}/{n} ({n_distinct} variants)")
        if written < args.top:
            name = f"{args.prefix}{_slug(arch)}"
            deck = list(modal_deck)
            ok, err = engine.check_deck(deck)
            if ok:
                (DECKS_DIR / f"{name}.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")
                print(f"      -> wrote decks/{name}.csv (legal, modal list)")
                written += 1
            else:
                print(f"      !! modal list illegal (errorType {err}) — skipped")
    print(f"\nwrote {written} meta decklists to {DECKS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

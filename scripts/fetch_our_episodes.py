#!/usr/bin/env python
"""自チームのラダーエピソードを取得し、対面別の実戦成績を集計する。

日次 Top Episodes データセットは「平均レートの高いエピソード」だけを収録するため、
自チーム（レート850前後）の対戦は 1 件も入っていない。ここでは **公式 CLI** の
シミュレーション大会向けコマンドで自分のエピソードとリプレイを取得する。

  kaggle competitions episodes <submission_id>   … 提出のエピソード一覧
  kaggle competitions replay   <episode_id>      … リプレイ本体

大会の data-description も「You can download replay files from other teams from the
Leaderboard」と明記しており、これが想定された経路。
（初版は Kaggle の内部 RPC を直接叩いていたが、ToS のクロール禁止に触れうるうえ
レート制限で長時間ブロックされるため、2026-08-02 に公式CLIへ全面的に置き換えた。）

リプレイの steps[0][0].visualize[0].action に両者の60枚デッキ（カードID）が入る。
アーキタイプ判定は mine_decklists.py と同一ヒューリスティック。

使い方:
    poetry run python scripts/fetch_our_episodes.py --submission-ids 55175745 55175751
    poetry run python scripts/fetch_our_episodes.py --submission-ids 55175745 --no-fetch
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from ptcg import engine

CACHE = REPO_ROOT / "data" / "episodes_ours"
OUR_TEAM = "cabbage patch"


def _kaggle(*args: str) -> str:
    env = dict(os.environ)
    env.setdefault("KAGGLE_CONFIG_DIR", str(REPO_ROOT / ".kaggle"))
    p = subprocess.run(["poetry", "run", "kaggle", "competitions", *args],
                       cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} 失敗: {p.stderr.strip()[:300]}")
    return p.stdout


def list_episodes(submission_id: int) -> list[dict]:
    out = _kaggle("episodes", str(submission_id), "--csv")
    return [r for r in csv.DictReader(io.StringIO(out)) if r.get("id")]


def get_replay(episode_id: int, dest: Path) -> Path:
    _kaggle("replay", str(episode_id), "-p", str(dest))
    p = dest / f"episode-{episode_id}-replay.json"
    if p.exists():
        return p
    cands = list(dest.glob(f"*{episode_id}*.json"))
    if not cands:
        raise FileNotFoundError(f"リプレイが保存されていない: {episode_id}")
    return cands[0]


def make_ace_label():
    cd = {c.cardId: c for c in engine.engine_card_data()}

    def is_pokemon(cid):
        c = cd.get(cid)
        return c is not None and getattr(c, "cardType", None) == 0

    def ace_label(deck):
        counts = Counter(deck)
        best = None
        for cid, n in counts.items():
            if n < 2 or not is_pokemon(cid):
                continue
            c = cd[cid]
            key = (c.hp or 0, bool(c.megaEx), bool(c.stage2), bool(c.stage1), cid)
            if best is None or key > best[0]:
                best = (key, cid)
        return cd[best[1]].name if best else "no_ace"

    return ace_label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission-ids", type=int, nargs="+", required=True)
    ap.add_argument("--no-fetch", action="store_true", help="キャッシュのみで集計")
    ap.add_argument("--sleep", type=float, default=0.5, help="リプレイ取得の間隔秒（レート制限対策）")
    args = ap.parse_args()

    ace_label = make_ace_label()

    for sid in args.submission_ids:
        cdir = CACHE / str(sid)
        cdir.mkdir(parents=True, exist_ok=True)
        meta_path = cdir / "_episodes.json"

        if not args.no_fetch:
            eps = list_episodes(sid)
            meta_path.write_text(json.dumps(eps), encoding="utf-8")
        elif meta_path.exists():
            eps = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            print(f"[{sid}] キャッシュ無し（--no-fetch を外す）", file=sys.stderr)
            continue

        eps = sorted(eps, key=lambda e: e.get("createTime", ""))
        pub = [e for e in eps if "VALIDATION" not in (e.get("type") or "")]
        print(f"\n===== submission {sid}: {len(pub)} episodes "
              f"({pub[0]['createTime'][:16]} 〜 {pub[-1]['createTime'][:16]}) =====")

        if not args.no_fetch:
            need = [e for e in pub
                    if "COMPLETED" in (e.get("state") or "")
                    and not (cdir / f"episode-{e['id']}-replay.json").exists()]
            print(f"リプレイ取得: {len(need)} 件（既取得 {len(pub) - len(need)}）")
            for i, e in enumerate(need, 1):
                try:
                    get_replay(int(e["id"]), cdir)
                except Exception as ex:
                    print(f"  [warn] ep {e['id']}: {ex}", file=sys.stderr)
                if i % 20 == 0:
                    print(f"  {i}/{len(need)}")
                time.sleep(args.sleep)

        mu_n: Counter = Counter()
        mu_w: Counter = Counter()
        for e in pub:
            rp = cdir / f"episode-{e['id']}-replay.json"
            if not rp.exists():
                continue
            try:
                rep = json.loads(rp.read_text(encoding="utf-8"))
                names = [a.get("Name") for a in rep["info"]["Agents"]]
                if OUR_TEAM not in names:
                    continue
                my_idx = names.index(OUR_TEAM)
                decks = rep["steps"][0][0]["visualize"][0]["action"]
                opp = ace_label(decks[1 - my_idx])
                reward = (rep.get("rewards") or [0, 0])[my_idx]
            except Exception:
                opp, reward = "parse_error", 0
            mu_n[opp] += 1
            if (reward or 0) > 0:
                mu_w[opp] += 1

        tot_n = sum(mu_n.values())
        tot_w = sum(mu_w.values())
        print(f"  合計 {tot_w}/{tot_n} = {tot_w / max(tot_n, 1):.1%}")
        for opp, n in mu_n.most_common():
            print(f"    vs {opp:28s}: {mu_w[opp]:3d}/{n:3d} = {mu_w[opp] / n:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

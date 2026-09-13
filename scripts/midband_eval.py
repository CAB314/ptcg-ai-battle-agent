#!/usr/bin/env python
"""中位帯模擬field評価 — 実遭遇分布（docs/data/midband_field.json）で加重勝率を測る。

内部プールfieldがLBを予測しなかった（hydrapple 72.6%→735 / ogerpon 68.4%→1037）反省から、
「我々の提出が実際に戦う相手構成」で候補を評価する。自アーキのセルは除外して正規化。

    poetry run python scripts/midband_eval.py --me runs/opp_pool_current/dragapult \
        --arch dragapult --tag dragapult_r3 --games 4000 --workers 44
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from selfplay_judge import CHUNK, play


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--me", required=True, help="方策ディレクトリ（weights.npz/config.json）")
    ap.add_argument("--deck", default=None, help="自デッキ（省略時 <me>/deck.csv）")
    ap.add_argument("--arch", default=None, help="自アーキ名（fieldの同名セルを除外）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--field", default="docs/data/midband_field.json")
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=44)
    args = ap.parse_args()

    field = json.loads((REPO_ROOT / args.field).read_text())["field"]
    if args.arch:
        field = [c for c in field if args.arch not in c["id"]]
    tot_w = sum(c["w"] for c in field)
    me = str((REPO_ROOT / args.me).resolve())
    my_deck = args.deck or f"{args.me}/deck.csv"

    out = {"me": args.me, "tag": args.tag, "matchups": {}}
    num = var = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for c in field:
            n_i = max(CHUNK, round(args.games * c["w"] / tot_w))
            arg = str((REPO_ROOT / c["arg"]).resolve()) if c["kind"] == "np" else c["arg"]
            kind = "np" if c["kind"] == "np" else "agent"
            r = play(me, kind, arg, my_deck, c["deck"], n_i, ex)
            out["matchups"][c["id"]] = {**r, "w": c["w"]}
            num += c["w"] / tot_w * r["wr"]
            var += (c["w"] / tot_w) ** 2 * r["wr"] * (1 - r["wr"]) / r["games"]
            print(f"[{args.tag}] vs {c['id']:22s} wr={r['wr']:.3f} (n={r['games']})", flush=True)
    out["midband_wr"] = num
    out["se"] = math.sqrt(var)
    print(f"[{args.tag}] 中位帯模擬field = {num:.4f} ± {out['se']:.4f}", flush=True)
    odir = REPO_ROOT / "runs" / "midband_eval"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / f"{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

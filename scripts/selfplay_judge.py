#!/usr/bin/env python
"""段C判定: 6アーキ自己強化の v0 基準測定（docs/plan-selfplay-6arch.md §4）。

測定は2種類。いずれも自側 = NpPolicy argmax + ガードマスク（配備と同一経路）。
  H2H  : 同一デッキ・方策のみ差で v0 と対戦（判定1: 勝率 >= 54% @ n=1,200）
  field: 他5アーキの v0 + grimmsnarl_ppo2 + trmewtwo_ppo2 に 8/5 実測シェア加重で対戦
         （判定2: v0 の field 勝率に対し Δ >= +1.5pp。判定3: 対面別の退行 -5pp 以内）

使い方:
    # v0 の基準値（PPO完了前に先行取得できる）
    poetry run python scripts/selfplay_judge.py --arch lopunny --policy v0 --field-games 6000
    # v1 の判定（H2H + field）
    poetry run python scripts/selfplay_judge.py --arch lopunny \
        --policy runs/ppo_opp_lopunny_v1/policy/best_gauntlet --tag v1 \
        --h2h-games 1200 --field-games 6000

出力: runs/selfplay_judge/<arch>_<tag>.json（対面別 + 加重集約 + SE）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

from setup_selfplay_run import ARCHS, SHARE, FROZEN

CHUNK = 100  # C++シミュレータのリーク対策: 1子プロセスあたりの対戦数

# ---- 子プロセス: 1マッチアップ N 戦 → 最終行に JSON ----
_CHILD = r'''
import json, os, sys
os.nice(19)
os.environ.setdefault("OMP_NUM_THREADS", "1")
repo, me_dir, opp_kind, opp_arg, my_deck, opp_deck, games, seat0 = sys.argv[1:9]
games, seat0 = int(games), int(seat0)
sys.path.insert(0, repo + "/src")
from ptcg.ml.vocab import load_tables
from ptcg.ml.np_forward import NpPolicy
from ptcg.ml.rl.actor import _np_agent_argmax, _rule_agent
tables = load_tables(repo + "/data/ml/cards.npz", repo + "/data/ml/attacks.npz")
me_pol = NpPolicy.load(me_dir + "/weights.npz", me_dir + "/config.json")
assert me_pol is not None, "me policy load failed: " + me_dir
me = _np_agent_argmax(me_pol, tables)
if opp_kind == "np":
    op_pol = NpPolicy.load(opp_arg + "/weights.npz", opp_arg + "/config.json")
    assert op_pol is not None, "opp policy load failed: " + opp_arg
    op = _np_agent_argmax(op_pol, tables)
else:
    op = _rule_agent(opp_arg)
my_d = [int(x) for x in open(repo + "/" + my_deck).read().split()]
op_d = [int(x) for x in open(repo + "/" + opp_deck).read().split()]
from ptcg.engine import ensure_cg_importable
ensure_cg_importable()
from cg.game import battle_finish, battle_select, battle_start
w = l = d = 0
for g in range(games):
    seat = (seat0 + g) % 2
    decks = (my_d, op_d) if seat == 0 else (op_d, my_d)
    obs, start = battle_start(list(decks[0]), list(decks[1]))
    assert start.errorPlayer < 0, "deck error"
    steps = 0
    try:
        while obs["current"]["result"] < 0 and steps < 4000:
            mv = me(obs) if obs["current"]["yourIndex"] == seat else op(obs)
            obs = battle_select(mv)
            steps += 1
        r = obs["current"]["result"]
    finally:
        battle_finish()
    if r == 2: d += 1
    elif r == seat: w += 1
    else: l += 1
print(json.dumps({"w": w, "l": l, "d": d}))
'''


def play(me_dir: str, opp_kind: str, opp_arg: str, my_deck: str, opp_deck: str,
         games: int, workers: ThreadPoolExecutor) -> dict:
    """games 戦を CHUNK 毎の子プロセスに分割して並列実行。"""
    def one(n: int, seat0: int) -> dict:
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"),
                   PTCG_CG_DIR=str(REPO_ROOT / "data/sample_submission/sample_submission/cg"))
        env.pop("PTCG_AGENT_DIR", None)
        r = subprocess.run([sys.executable, "-c", _CHILD, str(REPO_ROOT), me_dir,
                            opp_kind, opp_arg, my_deck, opp_deck, str(n), str(seat0)],
                           capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
        if r.returncode != 0:
            raise RuntimeError(f"child failed: {r.stderr[-500:]}")
        return json.loads(r.stdout.strip().splitlines()[-1])

    futs = []
    left, seat0 = games, 0
    while left > 0:
        n = min(CHUNK, left)
        futs.append(workers.submit(one, n, seat0))
        left -= n
        seat0 ^= 1
    tot = {"w": 0, "l": 0, "d": 0}
    for f in futs:
        c = f.result()
        for k in tot:
            tot[k] += c[k]
    n = tot["w"] + tot["l"] + tot["d"]
    return {**tot, "games": n, "wr": tot["w"] / n if n else 0.0}


def field_pool(arch: str, froot: str = FROZEN) -> list[dict]:
    label = Path(froot).name.split("_")[-1]  # opp_frozen_v0 → v0
    pool = []
    for o in ARCHS:
        if o == arch:
            continue
        pool.append({"id": f"{label}@{o}", "kind": "np", "arg": str(REPO_ROOT / froot / o),
                     "deck": f"{froot}/{o}/deck.csv", "w": SHARE[o]})
    for name, key in (("grimmsnarl_ppo2", "grimmsnarl"), ("trmewtwo_ppo2", "trmewtwo")):
        pool.append({"id": name, "kind": "agent", "arg": name,
                     "deck": f"agents/{name}/deck.csv", "w": SHARE[key]})
    return pool


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--policy", required=True, help="'v0' か policyディレクトリ")
    ap.add_argument("--tag", default=None, help="出力名（省略時 v0/policy名）")
    ap.add_argument("--frozen-root", default=FROZEN,
                    help="H2H 相手と field の基準プール（v2判定では runs/opp_frozen_v1）")
    ap.add_argument("--h2h-games", type=int, default=0)
    ap.add_argument("--field-games", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    frozen = str(REPO_ROOT / args.frozen_root / args.arch)
    me_dir = frozen if args.policy == "v0" else str(REPO_ROOT / args.policy)
    tag = args.tag or ("v0" if args.policy == "v0" else Path(args.policy).name)
    my_deck = f"{args.frozen_root}/{args.arch}/deck.csv"
    out = {"arch": args.arch, "tag": tag, "policy": me_dir, "matchups": {}}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        if args.h2h_games > 0:
            r = play(me_dir, "np", frozen, my_deck, my_deck, args.h2h_games, ex)
            out["h2h_vs_v0"] = r
            print(f"[{args.arch}/{tag}] H2H vs v0: {r['wr']:.3f} (n={r['games']})", flush=True)
        if args.field_games > 0:
            pool = field_pool(args.arch, args.frozen_root)
            tot_w = sum(c["w"] for c in pool)
            num, den, var = 0.0, 0.0, 0.0
            for c in pool:
                n_i = max(CHUNK, round(args.field_games * c["w"] / tot_w))
                r = play(me_dir, c["kind"], c["arg"], my_deck, c["deck"], n_i, ex)
                out["matchups"][c["id"]] = {**r, "w_share": c["w"]}
                num += c["w"] * r["wr"]
                den += c["w"]
                var += (c["w"] / tot_w) ** 2 * r["wr"] * (1 - r["wr"]) / r["games"]
                print(f"[{args.arch}/{tag}] vs {c['id']:18s} wr={r['wr']:.3f} (n={r['games']})",
                      flush=True)
            out["field_wr"] = num / den
            out["field_se"] = math.sqrt(var)
            print(f"[{args.arch}/{tag}] field(加重) = {out['field_wr']:.4f} "
                  f"± {out['field_se']:.4f}", flush=True)

    odir = REPO_ROOT / "runs" / "selfplay_judge"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / f"{args.arch}_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[out] runs/selfplay_judge/{args.arch}_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

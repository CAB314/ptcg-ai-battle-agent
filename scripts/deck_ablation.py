#!/usr/bin/env python
"""デッキアブレーション第1段: ゼロショット・スクリーニング（GPU不要・純CPU）。

学習済み方策を固定したまま **デッキだけ** 差し替え、実メタ加重の相手分布に対して
大量対戦させる。

統計設計（重要・当初の想定を実測で訂正済み）:
  エンジン `battle_start(deck0, deck1)` に**シード引数が無く**、山札シャッフルは内部で
  ランダムに決まる。したがって「共通乱数によるペア比較」は**実装不可能**であり、
  本比較は実質 **非ペア（独立2標本）** である（揃えられるのは先攻/後攻の並びのみ）。
  → 1候補1万戦・基準はその3倍にすると、差の95%CIは約 ±1.1pp。
     **検出できるのは概ね1.5pp以上の効果**。それ未満は第2段でも判別しない前提で読む。

    poetry run python scripts/deck_ablation.py --policy runs/ppo_grimmsnarl_v1/policy/best_gauntlet_cluster \
        --candidates docs/data/ablation_candidates.json --games 10000 --workers 24

出力: runs/ablation/<stamp>/results.json（候補ごとの勝率・基準との差・CI）

注意: 方策は基準リストで学習しているため、この段階には「不慣れバイアス」が残る。
      有望候補は第2段（転移学習＋対照群）で確認すること。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.nice(15)

import _bootstrap  # noqa: F401,E402
from _bootstrap import REPO_ROOT  # noqa: E402

# 子プロセス: 指定デッキで N 戦して RESULT を返す（gauntlet の子と同型。engine のゲーム跨ぎ
# リークを避けるため必ず独立プロセスで動かす）
_CHILD = r'''
import json, sys, os
sys.path.insert(0, r"%(repo)s/src")
policy_dir, my_deck_json, opp_kind, opp_arg, opp_deck, games, seed0 = sys.argv[1:8]
games, seed0 = int(games), int(seed0)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from ptcg.ml.vocab import load_tables
from ptcg.ml.np_forward import NpPolicy
from ptcg.ml.rl.actor import _np_agent_argmax, _rule_agent
REPO = r"%(repo)s"

def load_np(d):
    """方策ディレクトリ or エージェントディレクトリから「打ち手」を作る（2026-08-05 改訂）。

    agents/<name>/ に main.py があるならそれを **配備時とまったく同じ経路** で読む
    （engine.load_agent が PTCG_AGENT_DIR と vendor モジュールを分離してくれる）。
    これでリーサルガード・アンチループ・ルール縮退まで含めて再現でき、かつ
    **特徴量の世代が違う過去の提出物も相手/基準として使える**。
    main.py が無い（runs/.../policy/latest のような素の重み）場合は src の版で読む。
    デッキは呼び出し側が battle_start に渡すので、main.py のデッキ読みは使われない。
    """
    if os.path.exists(os.path.join(d, "main.py")):
        from ptcg.engine import load_agent
        return load_agent(os.path.join(d, "main.py"))
    p = NpPolicy.load(d + "/weights.npz", d + "/config.json")
    if p is None:
        raise SystemExit("方策をロードできない: " + d)
    return _np_agent_argmax(p, tables)

tables = load_tables(REPO + "/data/ml/cards.npz", REPO + "/data/ml/attacks.npz")
me = load_np(policy_dir)
if opp_kind == "np":
    op = load_np(opp_arg)
else:
    op = _rule_agent(opp_arg)
my_deck_l = json.loads(my_deck_json)
opp_deck_l = [int(x) for x in open(REPO + "/" + opp_deck).read().split()]
from ptcg.engine import ensure_cg_importable
ensure_cg_importable()
from cg.game import battle_finish, battle_select, battle_start
w = l = d = 0
for g in range(games):
    seat = (seed0 + g) %% 2   # 共通乱数: 基準/候補で同じ seed0 → 同じ先後の並び
    decks = (my_deck_l, opp_deck_l) if seat == 0 else (opp_deck_l, my_deck_l)
    obs, start = battle_start(list(decks[0]), list(decks[1]))
    steps = 0
    try:
        while obs["current"]["result"] < 0 and steps < 4000:
            mv = me(obs) if obs["current"]["yourIndex"] == seat else op(obs)
            obs = battle_select(mv)
            steps += 1
        r = obs["current"]["result"]
    finally:
        battle_finish()
    if r == seat: w += 1
    elif r == 2: d += 1
    else: l += 1
print("RESULT " + json.dumps({"w": w, "l": l, "d": d}))
'''


def _run_chunk(args) -> dict:
    policy_dir, deck, opp_kind, opp_arg, opp_deck, games, seed0 = args
    code = _CHILD % {"repo": str(REPO_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c", code, policy_dir, json.dumps(deck), opp_kind, opp_arg, opp_deck,
         str(games), str(seed0)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "OMP_NUM_THREADS": "1"},
    )
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[len("RESULT "):])
    return {"w": 0, "l": 0, "d": 0, "error": (proc.stderr or "")[-300:] or "RESULT行なし"}


def wilson(w: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, w / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    dn = 1 + z * z / n
    return ((p + z * z / (2 * n) - c) / dn, (p + z * z / (2 * n) + c) / dn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="評価に使う方策ディレクトリ（weights.npz/config.json）")
    ap.add_argument("--candidates", required=True, help="候補定義 JSON")
    ap.add_argument("--games", type=int, default=10000, help="1候補あたり総試合数")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=25, help="1サブプロセスの試合数（engineリーク対策）")
    ap.add_argument("--base-games-mult", type=float, default=3.0,
                    help="基準群の試合数倍率（全比較の分母になるので厚くする）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.candidates).read_text())
    base = spec["base"]          # {"name":..., "deck":[60枚]}
    cands = spec["candidates"]   # [{"name":..., "deck":[...], "note":...}, ...]
    field = spec["field"]        # [[deck_csv, weight], ...] 実メタ加重
    policy = str((REPO_ROOT / args.policy).resolve()) if not Path(args.policy).is_absolute() else args.policy

    missing = [f["id"] for f in field
               if f["kind"] == "np" and not (REPO_ROOT / f["arg"] / "weights.npz").exists()]
    if missing:
        raise SystemExit(f"相手方策のアセットが無い: {missing}（--candidates の arg を確認）")
    for f in field:
        if not (REPO_ROOT / f["deck"]).exists():
            raise SystemExit(f"相手デッキが無い: {f['deck']}")
    tot_w = sum(f["w"] for f in field)
    plan = []  # (arm_index, opp_deck, games, seed0)
    arms = [base] + cands
    for ai, arm in enumerate(arms):
        g_arm = args.games * (args.base_games_mult if ai == 0 else 1.0)
        for f in field:
            n = max(args.chunk, int(round(g_arm * f["w"] / tot_w / args.chunk) * args.chunk))
            arg = str((REPO_ROOT / f["arg"]).resolve()) if f["kind"] == "np" else f["arg"]
            for s in range(0, n, args.chunk):
                # seed0 は先攻/後攻の並びのみ揃える（山札シャッフルは engine 内部で不可制御）
                plan.append((ai, f["id"], f["kind"], arg, f["deck"], args.chunk, s))
    print(f"候補 {len(cands)} + 基準1 = {len(arms)}群 / 相手{len(field)}種 / "
          f"計 {sum(p[5] for p in plan):,}戦 / {len(plan):,}サブプロセス / {args.workers}並列", flush=True)

    agg = [{"w": 0, "l": 0, "d": 0, "per_opp": {}} for _ in arms]
    errs: list = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_chunk, (policy, arms[ai]["deck"], kind, arg, dk, g, s)): (ai, oid)
                for ai, oid, kind, arg, dk, g, s in plan}
        for fut in as_completed(futs):
            ai, opp = futs[fut]
            r = fut.result()
            if r.get("error"):
                errs.append((ai, opp, r["error"]))
            a = agg[ai]
            for k in ("w", "l", "d"):
                a[k] += r.get(k, 0)
            po = a["per_opp"].setdefault(opp, {"w": 0, "l": 0, "d": 0})
            for k in ("w", "l", "d"):
                po[k] += r.get(k, 0)
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(plan)} chunk  {el/60:.1f}分経過  "
                      f"残り約{el/done*(len(plan)-done)/60:.0f}分", flush=True)

    def wr(a):
        n = a["w"] + a["l"] + a["d"]
        return a["w"] / max(n, 1), n

    b_wr, b_n = wr(agg[0])
    rows = []
    for ai, arm in enumerate(arms):
        p, n = wr(agg[ai])
        lo, hi = wilson(agg[ai]["w"], n)
        # 差の標準誤差（独立2標本。engine にシードが無くペア化できないため厳密にこれ）
        se = math.sqrt(p * (1 - p) / max(n, 1) + b_wr * (1 - b_wr) / max(b_n, 1))
        rows.append({"name": arm["name"], "note": arm.get("note", ""), "n": n,
                     "wr": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                     "delta": round(p - b_wr, 4), "delta_se": round(se, 4),
                     "z": round((p - b_wr) / se, 2) if se > 0 and ai else 0.0,
                     "per_opp": {k: round(v["w"] / max(v["w"] + v["l"] + v["d"], 1), 4)
                                 for k, v in agg[ai]["per_opp"].items()}})

    out = Path(args.out) if args.out else REPO_ROOT / "runs" / "ablation" / time.strftime("%Y%m%d_%H%M") / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": args.policy, "games_per_arm": args.games,
                               "elapsed_s": round(time.time() - t0), "rows": rows}, indent=1, ensure_ascii=False))

    print(f"\n=== 結果（{time.time()-t0:.0f}秒）基準 {base['name']}: {b_wr:.2%} (n={b_n:,}) ===")
    print(f"{'候補':34s} {'勝率':>7s} {'95%CI':>16s} {'Δ':>7s} {'z':>6s}  備考")
    for r in sorted(rows[1:], key=lambda x: -x["delta"]):
        star = " ***" if abs(r["z"]) >= 2.58 else (" **" if abs(r["z"]) >= 1.96 else "")
        print(f"{r['name']:34s} {r['wr']:6.2%} [{r['ci'][0]:5.2%},{r['ci'][1]:5.2%}] "
              f"{r['delta']:+6.2%} {r['z']:+6.2f}{star}  {r['note']}")
    if errs:
        rate = len(errs) / max(len(plan), 1)
        print(f"\n★エラーチャンク {len(errs)}/{len(plan)} ({rate:.1%})  例: {errs[0][2][:200]}")
        if rate > 0.01:
            print("★エラー率が1%超。結果は信用できない。")
    print(f"\n[ok] {out}")
    print("** = p<0.05 / *** = p<0.01（独立2標本。engine にシード指定が無くペア化不能）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

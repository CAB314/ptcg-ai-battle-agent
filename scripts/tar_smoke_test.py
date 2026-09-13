#!/usr/bin/env python
"""tar 展開スモーク — 「スモーク緑なのに本番でモデル欠落/無限ループERROR」の検出。

build_submission.py が作った tar.gz を一時ディレクトリへ展開し、リポジトリの
sys.path を混ぜず素の exec で main.py をロードして対戦させる。フェーズ:
  1. vs ランダム（従来スモーク相当）
  2. **ミラー自己対戦**（Kaggleバリデーションと同一構成。2026-07-17 の alakazam_bc
     ERROR=ミラーで無限ゲーム18,167ステップ、の再発防止。STEP_CAP 超過で fail）
  3. weights.npz を退避した縮退複製で 1と2（numpy 不在相当の完走確認）

**各フェーズは独立サブプロセスで実行する**。エンジンにはゲームを跨いで内部レジストリが
蓄積し `FixedList capacity:7` の C++ 例外でプロセス死する既知リークがあり（実測:
vsランダム3戦→同一プロセスでミラー戦、で再現）、Kaggle は1ゲーム=1プロセスなので
本番には存在しない経路。ローカルだけ多ゲーム/プロセスにすると偽陽性クラッシュになる。

使い方:
    poetry run python scripts/tar_smoke_test.py alakazam_bc --games 3 --mirror-games 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import SUBMISSION_DIR

STEP_CAP = 3000  # 通常ゲームは~250ステップ。超過=無限ループ（バリデーションERRORの原因）

_CHILD = r'''
import json, random, sys, os
workdir, games, mirror, step_cap = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4])
os.chdir(workdir)
sys.path.insert(0, workdir)
os.environ.pop("PTCG_AGENT_DIR", None)
src = open("main.py", encoding="utf-8").read()
ns = {}
exec(compile(src, "<tar-agent>", "exec"), ns)  # 素の名前空間（__file__なし）
agent = ns["agent"]
deck = [int(x) for x in open("deck.csv").read().splitlines() if x.strip()]
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start

def rand_move(sel):
    n = len(sel.option)
    k = max(min(sel.maxCount, n), sel.minCount)
    return random.sample(range(n), k)

crashes = illegal = 0
max_steps = 0
loop_fail = False
for gi in range(games):
    obs, start = battle_start(list(deck), list(deck))
    assert start.errorPlayer < 0, "deck error"
    me = gi % 2
    steps = 0
    try:
        while obs["current"]["result"] < 0 and steps < step_cap:
            who = obs["current"]["yourIndex"]
            o = to_observation_class(obs)
            if mirror or who == me:
                try:
                    mv = agent(obs)
                except Exception:
                    crashes += 1
                    mv = rand_move(o.select)
                n = len(o.select.option)
                ok = (isinstance(mv, list) and len(set(mv)) == len(mv)
                      and all(isinstance(x, int) and 0 <= x < n for x in mv)
                      and o.select.minCount <= len(mv) <= o.select.maxCount)
                if not ok:
                    illegal += 1
                    mv = rand_move(o.select)
            else:
                mv = rand_move(o.select)
            obs = battle_select(mv)
            steps += 1
        if steps >= step_cap:
            loop_fail = True
        max_steps = max(max_steps, steps)
    finally:
        battle_finish()
stats = ns.get("_STATS") or {}
print("RESULT " + json.dumps({
    "crashes": crashes, "illegal": illegal, "max_steps": max_steps,
    "loop_fail": loop_fail, "stats": stats,
}))
'''


def _run_phase(workdir: Path, games: int, label: str, mirror: bool = False,
               min_model_rate: float | None = None) -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(workdir), str(games), "1" if mirror else "0", str(STEP_CAP)],
        capture_output=True, text=True, timeout=1800,
    )
    line = next((ln for ln in (proc.stdout or "").splitlines() if ln.startswith("RESULT ")), None)
    if proc.returncode != 0 or line is None:
        tailerr = (proc.stderr or "").strip().splitlines()[-3:]
        print(f"[FAIL] {label}: 子プロセス異常終了 rc={proc.returncode}（エンジンクラッシュ等）")
        for ln in tailerr:
            print(f"       {ln}")
        return False
    r = json.loads(line[len("RESULT "):])
    stats = r.get("stats") or {}
    total = sum(v for v in stats.values() if isinstance(v, int)) or 1
    model_rate = stats.get("model", 0) / total
    print(f"[{label}] {games}戦 crashes={r['crashes']} illegal={r['illegal']} max_steps={r['max_steps']} "
          f"model発動率={model_rate:.1%} (model={stats.get('model', 0)}, rules={stats.get('rules', 0)}, "
          f"antiloop={stats.get('antiloop', 0)})")
    if r["loop_fail"]:
        print(f"[FAIL] {label}: {STEP_CAP} ステップ未決着のゲームあり（無限ループ疑い）")
    ok = r["crashes"] == 0 and r["illegal"] == 0 and not r["loop_fail"]
    # モデルが実際に発火しているかをリリースゲートに入れる（2026-08-02 追加）。
    # 資産の版ズレ等でモデル層が全滅してもクラッシュはしないため、従来は
    # 「ルールだけで戦う縮退エージェント」が ALL GREEN で通ってしまった。
    if min_model_rate is not None and model_rate < min_model_rate:
        print(f"[FAIL] {label}: model発動率 {model_rate:.1%} < {min_model_rate:.0%}"
              f"（モデル層が働いていない。weights/vocab の版ズレを疑う）")
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--mirror-games", type=int, default=8,
                    help="ミラー自己対戦のゲーム数（バリデーション構成の再現）")
    ap.add_argument("--min-model-rate", type=float, default=0.95,
                    help="ML エージェントに要求する model 発動率の下限（提出ゲート）")
    args = ap.parse_args()
    tar_path = SUBMISSION_DIR / f"{args.agent}.tar.gz"
    if not tar_path.exists():
        print(f"[FAIL] {tar_path} が無い（先に build_submission.py を実行）")
        return 1
    results = []
    with tempfile.TemporaryDirectory(prefix="tar_smoke_") as td:
        base = Path(td) / "a"
        base.mkdir()
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(base)
        # npz 資産を持つ＝ML エージェントなら、モデルが実際に発火することを緑条件にする
        has_npz = any(base.glob("*.npz"))
        gate = args.min_model_rate if has_npz else None
        results.append(_run_phase(base, args.games, "tar展開", min_model_rate=gate))
        results.append(_run_phase(base, max(args.mirror_games, 1), "ミラー自己対戦",
                                  mirror=True, min_model_rate=gate))
        degraded = Path(td) / "b"
        shutil.copytree(base, degraded)
        removed = False
        for w in degraded.glob("*.npz"):
            w.unlink()
            removed = True
        if removed:
            results.append(_run_phase(degraded, 1, "縮退(weights無し)"))
            results.append(_run_phase(degraded, 2, "縮退ミラー", mirror=True))
        else:
            print("[note] npz 資産なし（純ルールエージェント）→ 縮退テスト省略")
    ok = all(results)
    print("\n=== " + ("TAR SMOKE GREEN ✅" if ok else "赤あり ❌") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

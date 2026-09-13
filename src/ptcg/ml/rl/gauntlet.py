"""ガントレット — policy/latest の周期評価と best 追跡・自動巻き戻しトリガ。

- 各マッチアップを**独立サブプロセス**で実行（エンジンのゲーム跨ぎリーク対策）。
- 自側 = policy_dir の NpPolicy（argmax + ガードマスク = 配備と同一経路）。
- composite = 全マッチアップ合算勝率。best より 3pp 以上低い評価が2回連続で
  run/ROLLBACK を書く（learner が次イテレーション境界で best を復元）。
- best 更新時は policy を runs/<run>/policy/best_gauntlet/ へコピー。

書き込み: gauntlet.jsonl（評価履歴）/ best_gauntlet.json（best記録）/ ROLLBACK。
league_state.json には触れない（書き手は supervisor のみ）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

# 子プロセス: 1マッチアップ N 戦 → "RESULT {json}" を出力
_CHILD = r'''
import json, sys, os
sys.path.insert(0, r"%(repo)s/src")
policy_dir, opp_kind, opp_arg, opp_deck, my_deck, games = sys.argv[1:7]
games = int(games)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from ptcg.ml.vocab import load_tables
from ptcg.ml.np_forward import NpPolicy
from ptcg.ml.rl.actor import _np_agent_argmax, _rule_agent
REPO = r"%(repo)s"
tables = load_tables(REPO + "/data/ml/cards.npz", REPO + "/data/ml/attacks.npz")
me_pol = NpPolicy.load(policy_dir + "/weights.npz", policy_dir + "/config.json")
me = _np_agent_argmax(me_pol, tables)
if opp_kind == "np":
    op_pol = NpPolicy.load(opp_arg + "/weights.npz", opp_arg + "/config.json")
    op = _np_agent_argmax(op_pol, tables)
else:
    op = _rule_agent(opp_arg)
my_deck_l = [int(x) for x in open(REPO + "/" + my_deck).read().split()]
opp_deck_l = [int(x) for x in open(REPO + "/" + opp_deck).read().split()]
from ptcg.engine import ensure_cg_importable
ensure_cg_importable()
from cg.game import battle_finish, battle_select, battle_start
w = l = d = 0
for g in range(games):
    seat = g %% 2
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


def default_matchups(run: Path, my_deck: str, games_scale: float = 1.0) -> list[dict]:
    anchor = str(run / "policy" / "anchor")
    g = lambda n: max(20, int(n * games_scale))  # noqa: E731
    return [
        {"id": "vs_bc_frozen", "kind": "np", "arg": anchor, "deck": my_deck, "games": g(200)},
        {"id": "vs_glethal2@kangaskhan", "kind": "agent", "arg": "greedy_lethal2",
         "deck": "decks/meta_mega_kangaskhan_ex.csv", "games": g(100)},
        {"id": "vs_glethal2@mewtwo", "kind": "agent", "arg": "greedy_lethal2",
         "deck": "decks/meta_team_rocket_s_mewtwo_ex.csv", "games": g(100)},
        {"id": "vs_glethal2@alakazam", "kind": "agent", "arg": "greedy_lethal2",
         "deck": "decks/meta_alakazam.csv", "games": g(100)},
        {"id": "vs_kangaskhan_pb2", "kind": "agent", "arg": "kangaskhan_playbook2",
         "deck": "agents/kangaskhan_playbook2/deck.csv", "games": g(100)},
        {"id": "vs_greattusk_lo", "kind": "agent", "arg": "greattusk_lo_playbook",
         "deck": "agents/greattusk_lo_playbook/deck.csv", "games": g(100)},
    ]


def run_matchup(policy_dir: str, m: dict, my_deck: str, chunk: int = 25) -> dict:
    """restart-every 相当: chunk 戦ごとにサブプロセスを分割して合算。"""
    total = {"w": 0, "l": 0, "d": 0}
    remaining = m["games"]
    while remaining > 0:
        n = min(chunk, remaining)
        code = _CHILD % {"repo": str(REPO_ROOT)}
        opp_arg = m["arg"]
        proc = subprocess.run(
            [sys.executable, "-c", code, policy_dir, m["kind"], opp_arg, m["deck"], my_deck, str(n)],
            capture_output=True, text=True, timeout=1800,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "OMP_NUM_THREADS": "1",
                 "NICE_ADJ": "19"},
        )
        line = next((ln for ln in (proc.stdout or "").splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            return {**total, "error": (proc.stderr or "")[-200:]}
        r = json.loads(line[len("RESULT "):])
        for k in ("w", "l", "d"):
            total[k] += r[k]
        remaining -= n
    return total


def evaluate(run: Path, my_deck: str, games_scale: float = 1.0, workers: int = 4) -> dict:
    """policy/latest を全マッチアップ評価して結果 dict を返す（並列サブプロセス）。"""
    from concurrent.futures import ThreadPoolExecutor

    # 評価中に latest が更新されないよう固定コピーを取る
    eval_dir = run / "policy" / ".eval_snapshot"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    shutil.copytree(run / "policy" / "latest", eval_dir)
    version = json.loads((eval_dir / "version.json").read_text())["version"]
    matchups = default_matchups(run, my_deck, games_scale)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_matchup, str(eval_dir), m, my_deck): m for m in matchups}
        for fut in list(futs):
            results[futs[fut]["id"]] = fut.result()
    total_w = sum(r["w"] for r in results.values())
    total_n = sum(r["w"] + r["l"] + r["d"] for r in results.values())
    composite = total_w / max(total_n, 1)
    out = {"t": round(time.time(), 1), "version": version, "composite": round(composite, 4),
           "matchups": {k: {**v, "wr": round(v["w"] / max(v["w"] + v["l"] + v["d"], 1), 4)}
                        for k, v in results.items()}}
    return out


def _history(run: Path) -> list[float]:
    p = run / "gauntlet.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            c = json.loads(line).get("composite")
        except Exception:
            continue
        if c is not None:
            out.append(float(c))
    return out


def _degraded(hist: list[float], cur: float, margin: float) -> bool:
    """退化判定。best（単発評価＝±1.5-2pp のノイズを含む）を基準にすると、たまたま高く
    出た1回が恒久的なバーになり「常に best-3pp」→巻き戻し→β倍増/lr半減の空回りになる
    （7/22 deck2・7/27 deck2 で実測）。そこでロバスト統計を2本使う:
      - 直近の典型値（直近6回の中央値）から margin 下回る = 最近の自分より落ちた
      - 歴代高水準（上位5件の中央値）から 2×margin 下回る = 破滅的退化の絶対床
    """
    if len(hist) < 4:
        return False
    recent = sorted(hist[-6:])
    ref_recent = recent[len(recent) // 2]
    top = sorted(hist, reverse=True)[:5]
    ref_high = sorted(top)[len(top) // 2]
    return cur < ref_recent - margin or cur < ref_high - 2 * margin


def gauntlet_step(run: Path, my_deck: str, games_scale: float = 1.0,
                  rollback_margin: float = 0.03) -> dict:
    hist = _history(run)
    res = evaluate(run, my_deck, games_scale)
    best_file = run / "best_gauntlet.json"
    best = json.loads(best_file.read_text()) if best_file.exists() else None
    if best is None or res["composite"] >= best["composite"]:
        # best 更新: policy コピー + 記録 + 連続悪化カウンタリセット
        bdir = run / "policy" / "best_gauntlet"
        if bdir.exists():
            shutil.rmtree(bdir)
        shutil.copytree(run / "policy" / ".eval_snapshot", bdir)
        res["bad_streak"] = 0
        best_file.write_text(json.dumps({**res, "path": str(bdir)}))
        res["is_best"] = True
    else:
        streak = int(best.get("bad_streak", 0))
        streak = streak + 1 if _degraded(hist, res["composite"], rollback_margin) else 0
        res["is_best"] = False
        if streak >= 2:
            (run / "ROLLBACK").write_text(json.dumps(
                {"reason": f"composite {res['composite']} がロバスト基準を {rollback_margin} 下回る評価×2"}))
            res["rollback_triggered"] = True
            streak = 0  # 巻き戻しの効果が出る前に再発火しないようリセット
        best["bad_streak"] = streak
        best_file.write_text(json.dumps(best))
        res["bad_streak"] = streak
    with open(run / "gauntlet.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    return res

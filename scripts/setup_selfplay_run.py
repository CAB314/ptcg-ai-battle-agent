#!/usr/bin/env python
"""6アーキタイプ自己強化 run の初期化（docs/plan-selfplay-6arch.md §1 の実装）。

configs/ppo_opp_<arch>.json を生成し、ensure_run で run を初期化した上で、
league_state.json を相互対戦仕様に書き換える:
  buckets: latest 0.40 / snapshot 0.25 / fixed 0.35
  fixed  : 他5アーキの v0（np_fixed・現行特徴量）+ grimmsnarl_ppo2 / trmewtwo_ppo2
           （rule_agent 経路。v1特徴量なので np_fixed に入れると NpPolicy.load が
             None を返し actor が即死する — 2026-08-06 スモークで確認済み）
  各セル w = 2026-08-05 実測メタシェア（league._pfsp_pick の静的重み）

注意: PPOConfig の league_* / deck_mix は league_state に反映されない死に設定。
リーグ構成はこのスクリプトで league_state.json を直接書くのが唯一の経路。

使い方（リポジトリ直下・venv有効化済み）:
    PYTHONPATH=src python scripts/setup_selfplay_run.py --arch lopunny --preflight
    PYTHONPATH=src python scripts/setup_selfplay_run.py --all --preflight
    PYTHONPATH=src python scripts/setup_selfplay_run.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

ARCHS = ["lopunny", "alakazam", "kangaskhan", "ogerpon", "dragapult", "garchomp",
         "lucario", "hydrapple"]  # lucario/hydrapple は 2026-08-10 追加（round 3 から）

# 2026-08-09 実測メタシェア（runs/daily_data/meta_shares_2026-08-09.json、
# Alakazam 系 ace 245/743 は 66 へ統合済み）。バケット内の相対重みなので正規化不要。
# 旧値（8/5）: grimmsnarl .3002 / alakazam .1916 / lopunny .1227 / kangaskhan .1209 …
SHARE = {
    "grimmsnarl": 0.321,
    "alakazam": 0.179,
    "lopunny": 0.123,
    "kangaskhan": 0.096,
    "dragapult": 0.082,
    "ogerpon": 0.048,
    "lucario": 0.043,
    "thwackey": 0.020,
    "garchomp": 0.020,
    "hydrapple": 0.017,
    "trmewtwo": 0.013,
}

ARCH_DECK = {
    "grimmsnarl": "decks/meta_marnie_s_grimmsnarl_ex.csv",
    "alakazam": "decks/meta_alakazam.csv",
    "lopunny": "decks/meta_mega_lopunny_ex.csv",
    "kangaskhan": "decks/meta_mega_kangaskhan_ex.csv",
    "ogerpon": "decks/meta_teal_mask_ogerpon_ex.csv",
    "dragapult": "decks/meta_dragapult_ex.csv",
    "thwackey": "decks/meta_thwackey.csv",
    "garchomp": "decks/meta_cynthia_s_garchomp_ex.csv",
    "trmewtwo": "decks/meta_team_rocket_s_mewtwo_ex.csv",
    "lucario": "decks/meta_mega_lucario_ex.csv",
    "hydrapple": "decks/meta_hydrapple_ex.csv",
}

BUCKETS = {"latest": 0.40, "snapshot": 0.25, "fixed": 0.35}
FROZEN = "runs/opp_frozen_v0"


def round_params(rnd: int, arch: str) -> dict:
    """ラウンドごとの相手プール/anchor/run名。round N は v(N-1) を基準に vN を作る。"""
    if rnd == 1:
        return {"froot": "runs/opp_frozen_v0", "label": "v0",
                "anchor": f"runs/bc_opp_{arch}/best.pt",
                "run": f"ppo_opp_{arch}_v1", "cfg": f"ppo_opp_{arch}.json"}
    if rnd == 2:
        # anchor は v1 の npz を scripts/npz_to_anchor.py で .pt 化したもの
        return {"froot": "runs/opp_frozen_v1", "label": "v1",
                "anchor": f"runs/anchors_v1/{arch}.pt",
                "run": f"ppo_opp_{arch}_v2", "cfg": f"ppo_opp2_{arch}.json"}
    if rnd == 3:
        # 8アーキ相互強化。プール = opp_pool_current（各アーキの現行採用版）。
        # anchor は anchors_cur/（v2/v1 は npz→pt 変換、新規2種は BC best.pt のコピー）
        return {"froot": "runs/opp_pool_current", "label": "cur",
                "anchor": f"runs/anchors_cur/{arch}.pt",
                "run": f"ppo_opp_{arch}_r3", "cfg": f"ppo_opp3_{arch}.json"}
    raise SystemExit(f"未対応 round: {rnd}")


def meta_decks() -> list[list]:
    tot = sum(SHARE.values())
    return [[ARCH_DECK[k], round(v / tot, 4)]
            for k, v in sorted(SHARE.items(), key=lambda x: -x[1])]


def fixed_cells(arch: str, rp: dict) -> list[dict]:
    cells = []
    for o in ARCHS:
        if o == arch:
            continue
        cells.append({
            "id": f"{rp['label']}@{o}", "kind": "np_fixed",
            "path": str((REPO_ROOT / rp["froot"] / o).resolve()),
            "deck": f"{rp['froot']}/{o}/deck.csv",
            "ema": 0.5, "games": 0, "w": SHARE[o],
        })
    for name, key in (("grimmsnarl_ppo2", "grimmsnarl"), ("trmewtwo_ppo2", "trmewtwo")):
        cells.append({
            "id": name, "kind": "rule_agent", "agent": name,
            "deck": f"agents/{name}/deck.csv",
            "ema": 0.5, "games": 0, "w": SHARE[key],
        })
    return cells


def make_config(arch: str, rp: dict) -> Path:
    from ptcg.ml.rl.config import PPOConfig

    cfg = PPOConfig().__dict__ | {
        "run_name": rp["run"],
        "anchor_ckpt": rp["anchor"],
        "deck_csv": f"{rp['froot']}/{arch}/deck.csv",
    }
    p = REPO_ROOT / "configs" / rp["cfg"]
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    return p


def setup(arch: str, dry: bool, rp: dict) -> Path:
    from ptcg.ml.rl.league import load_state, save_state

    run = REPO_ROOT / "runs" / rp["run"]
    cfg_path = make_config(arch, rp)
    if dry:
        state = {"snapshots": []}
    else:
        from run_actors import ensure_run

        ensure_run(run, str(cfg_path))
        state = load_state(run / "league_state.json")
    state["buckets"] = dict(BUCKETS)
    state["fixed"] = fixed_cells(arch, rp)
    state.setdefault("decks", {"weak": [], "uniform": []})["meta"] = meta_decks()
    state.setdefault("deck_mix", {"meta": 0.55, "weak": 0.25, "uniform": 0.20})
    state.setdefault("policy_version", 0)
    state.setdefault("pfsp_floor", 0.05)
    if dry:
        print(json.dumps({"arch": arch, "buckets": state["buckets"],
                          "fixed": state["fixed"], "meta": state["decks"]["meta"]},
                         ensure_ascii=False, indent=1))
    else:
        save_state(state, run / "league_state.json")
        print(f"[setup] {run.name}: fixed×{len(state['fixed'])} buckets={state['buckets']}")
    return run


def preflight(arch: str, rp: dict) -> bool:
    """fixed 各セルを actor.py と同一経路でロードし、policy/latest と1戦ずつ回す。"""
    os.environ.setdefault("PTCG_CG_DIR",
                          str(REPO_ROOT / "data/sample_submission/sample_submission/cg"))
    from ptcg.engine import ensure_cg_importable
    from ptcg.ml.np_forward import NpPolicy
    from ptcg.ml.rl.actor import _np_agent_argmax, _rule_agent
    from ptcg.ml.rl.league import load_state
    from ptcg.ml.vocab import load_tables

    run = REPO_ROOT / "runs" / rp["run"]
    tables = load_tables(REPO_ROOT / "data/ml/cards.npz", REPO_ROOT / "data/ml/attacks.npz")
    assert tables is not None, "data/ml テーブルがロードできない"
    me_pol = NpPolicy.load(run / "policy/latest/weights.npz", run / "policy/latest/config.json")
    assert me_pol is not None, "policy/latest がロードできない"
    me = _np_agent_argmax(me_pol, tables)
    cfg = json.loads((run / "config.json").read_text())
    my_deck = [int(x) for x in (REPO_ROOT / cfg["deck_csv"]).read_text().split()]

    ensure_cg_importable()
    from cg.game import battle_finish, battle_select, battle_start

    ok = True
    for c in load_state(run / "league_state.json")["fixed"]:
        try:
            if c["kind"] == "np_fixed":
                p = Path(c["path"])
                pol = NpPolicy.load(p / "weights.npz", p / "config.json")
                assert pol is not None, "NpPolicy.load が None"
                opp = _np_agent_argmax(pol, tables)
            else:
                opp = _rule_agent(c["agent"])
            opp_deck = [int(x) for x in (REPO_ROOT / c["deck"]).read_text().split()]
            obs, start = battle_start(list(my_deck), list(opp_deck))
            assert start.errorPlayer < 0, f"deck error p{start.errorPlayer}"
            steps = 0
            try:
                while obs["current"]["result"] < 0 and steps < 4000:
                    mv = me(obs) if obs["current"]["yourIndex"] == 0 else opp(obs)
                    obs = battle_select(mv)
                    steps += 1
                r = obs["current"]["result"]
            finally:
                battle_finish()
            print(f"  [preflight {arch}] {c['id']:20s} ok result={r} steps={steps} w={c['w']}")
        except Exception as e:
            print(f"  [preflight {arch}] {c['id']:20s} ★NG {type(e).__name__}: {e}")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--round", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    targets = ARCHS if args.all else ([args.arch] if args.arch else [])
    if not targets:
        ap.error("--arch か --all を指定")
    bad = []
    for a in targets:
        rp = round_params(args.round, a)
        setup(a, args.dry_run, rp)
        if args.preflight and not args.dry_run:
            if not preflight(a, rp):
                bad.append(a)
    if bad:
        print(f"★preflight 失敗: {bad}")
        return 1
    print("[setup] 完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

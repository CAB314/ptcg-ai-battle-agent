#!/usr/bin/env python
"""kangaskhan の対Dragapult+Hydrapple カリキュラムrun 初期化（2026-08-16）。

fixed = dragapultパイロット段階 + hydrappleパイロット段階 + Grimm維持を含む通常セル。
anchor = dhシード済み重み。PFSPの (1-EMA)²×w で攻略済み段は自動退場。

    PYTHONPATH=src python scripts/setup_dh_curriculum.py --preflight
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

FIXED = [
    # dragapult 段階（計 w .28 — 新王者22%に厚く）
    {"id": "greedy@dragapult", "kind": "rule_agent", "agent": "greedy_lethal2",
     "deck": "decks/meta_dragapult_ex.csv", "w": 0.05},
    {"id": "dragapult_playbook", "kind": "rule_agent", "agent": "dragapult_playbook",
     "deck": "agents/dragapult_playbook/deck.csv", "w": 0.07},
    {"id": "dragapult_r3", "kind": "np_fixed", "path": "runs/opp_pool_current/dragapult",
     "deck": "runs/opp_pool_current/dragapult/deck.csv", "w": 0.16},
    # hydrapple 段階（計 w .22）
    {"id": "greedy@hydrapple", "kind": "rule_agent", "agent": "greedy_lethal2",
     "deck": "decks/meta_hydrapple_ex.csv", "w": 0.04},
    {"id": "hydrapple_v0", "kind": "np_fixed", "path": "runs/opp_frozen_new_v0/hydrapple",
     "deck": "runs/opp_frozen_new_v0/hydrapple/deck.csv", "w": 0.06},
    {"id": "hydrapple_k2", "kind": "np_fixed", "path": "runs/opp_frozen_k2/hydrapple",
     "deck": "runs/opp_frozen_k2/hydrapple/deck.csv", "w": 0.12},
    # 維持セル（Grimm 17% を厚めに）
    {"id": "grimmsnarl_ppo2", "kind": "rule_agent", "agent": "grimmsnarl_ppo2",
     "deck": "agents/grimmsnarl_ppo2/deck.csv", "w": 0.20},
    {"id": "cur@lopunny", "kind": "np_fixed", "path": "runs/opp_pool_current/lopunny",
     "deck": "runs/opp_pool_current/lopunny/deck.csv", "w": 0.12},
    {"id": "cur@alakazam", "kind": "np_fixed", "path": "runs/opp_pool_current/alakazam",
     "deck": "runs/opp_pool_current/alakazam/deck.csv", "w": 0.11},
    {"id": "cur@ogerpon", "kind": "np_fixed", "path": "runs/opp_pool_current/ogerpon",
     "deck": "runs/opp_pool_current/ogerpon/deck.csv", "w": 0.04},
    {"id": "cur@lucario", "kind": "np_fixed", "path": "runs/opp_pool_current/lucario",
     "deck": "runs/opp_pool_current/lucario/deck.csv", "w": 0.03},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--run-name", default="ppo_opp_kangaskhan_dh")
    ap.add_argument("--anchor", default="runs/init_kangaskhan_dhseed/best.pt",
                    help="既定 = kang_r3 素のまま（シード無し）。シード版は bc_opp_*_dhseed2 等を指定")
    args = ap.parse_args()

    from ptcg.ml.rl.config import PPOConfig
    from ptcg.ml.rl.league import load_state, save_state

    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_actors import ensure_run

    cfg = PPOConfig().__dict__ | {
        "run_name": args.run_name,
        "anchor_ckpt": args.anchor,
        "deck_csv": "runs/opp_pool_current/kangaskhan/deck.csv",
    }
    cfg_path = REPO_ROOT / "configs" / f"{args.run_name.replace('ppo_opp_', 'ppo_opp_')}.json"
    cfg_path = REPO_ROOT / "configs" / f"{args.run_name}.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))

    run = REPO_ROOT / "runs" / args.run_name
    ensure_run(run, str(cfg_path))
    state = load_state(run / "league_state.json")
    state["buckets"] = {"latest": 0.40, "snapshot": 0.25, "fixed": 0.35}
    fixed = [dict(c) for c in FIXED]
    for c in fixed:
        if c["kind"] == "np_fixed":
            c["path"] = str((REPO_ROOT / c["path"]).resolve())
        c.setdefault("ema", 0.5)
        c.setdefault("games", 0)
    state["fixed"] = fixed
    save_state(state, run / "league_state.json")
    print(f"[setup] {run.name}: fixed×{len(fixed)} (dpt段3 + hyd段3 + 維持5)")

    if args.preflight:
        import os
        os.environ.setdefault("PTCG_CG_DIR",
                              str(REPO_ROOT / "data/sample_submission/sample_submission/cg"))
        from pathlib import Path

        from ptcg.engine import ensure_cg_importable
        from ptcg.ml.np_forward import NpPolicy
        from ptcg.ml.rl.actor import _np_agent_argmax, _rule_agent
        from ptcg.ml.vocab import load_tables

        tables = load_tables(REPO_ROOT / "data/ml/cards.npz", REPO_ROOT / "data/ml/attacks.npz")
        me_pol = NpPolicy.load(run / "policy/latest/weights.npz", run / "policy/latest/config.json")
        assert me_pol is not None
        me = _np_agent_argmax(me_pol, tables)
        my_deck = [int(x) for x in (REPO_ROOT / cfg["deck_csv"]).read_text().split()]
        ensure_cg_importable()
        from cg.game import battle_finish, battle_select, battle_start

        ok = True
        for c in state["fixed"]:
            try:
                if c["kind"] == "np_fixed":
                    p = Path(c["path"])
                    pol = NpPolicy.load(p / "weights.npz", p / "config.json")
                    assert pol is not None, "NpPolicy.load None"
                    opp = _np_agent_argmax(pol, tables)
                else:
                    opp = _rule_agent(c["agent"])
                opp_deck = [int(x) for x in (REPO_ROOT / c["deck"]).read_text().split()]
                obs, start = battle_start(list(my_deck), list(opp_deck))
                assert start.errorPlayer < 0
                steps = 0
                try:
                    while obs["current"]["result"] < 0 and steps < 4000:
                        mv = me(obs) if obs["current"]["yourIndex"] == 0 else opp(obs)
                        obs = battle_select(mv)
                        steps += 1
                    r = obs["current"]["result"]
                finally:
                    battle_finish()
                print(f"  [preflight] {c['id']:20s} ok result={r} steps={steps} w={c['w']}")
            except Exception as e:
                print(f"  [preflight] {c['id']:20s} ★NG {type(e).__name__}: {e}")
                ok = False
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

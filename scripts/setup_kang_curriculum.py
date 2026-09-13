#!/usr/bin/env python
"""対Kangaskhan弱点克服run の初期化（BCシード + パイロット強度カリキュラム）。

league 構成:
  latest 0.40 / snapshot 0.25 / fixed 0.35
  fixed = Kangaskhanパイロット段階(греedy→playbook→BC v0→現行r3, 計w0.45)
        + 通常プール(忘却防止, 計w0.55)
  PFSP が (1-EMA)²×w なので「勝てるようになった段階は自動で退場し、
  まだ勝てない段階に露出が移る」= 自然なカリキュラム進行。
anchor = BCシード済み重み（注入した勝ち筋を KL で保護する）。

使い方（the cluster・venv有効）:
    PYTHONPATH=src python scripts/setup_kang_curriculum.py --arch hydrapple --preflight
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

KANG_LADDER = [
    {"id": "glethal2@kang", "kind": "rule_agent", "agent": "greedy_lethal2",
     "deck": "decks/meta_mega_kangaskhan_ex.csv", "w": 0.06},
    {"id": "kang_playbook2", "kind": "rule_agent", "agent": "kangaskhan_playbook2",
     "deck": "agents/kangaskhan_playbook2/deck.csv", "w": 0.08},
    {"id": "kang_v0", "kind": "np_fixed", "path": "runs/opp_frozen_v0/kangaskhan",
     "deck": "runs/opp_frozen_v0/kangaskhan/deck.csv", "w": 0.12},
    {"id": "kang_cur", "kind": "np_fixed", "path": "runs/opp_pool_current/kangaskhan",
     "deck": "runs/opp_pool_current/kangaskhan/deck.csv", "w": 0.19},
]

RETENTION = [
    {"id": "grimmsnarl_ppo2", "kind": "rule_agent", "agent": "grimmsnarl_ppo2",
     "deck": "agents/grimmsnarl_ppo2/deck.csv", "w": 0.20},
    {"id": "cur@lopunny", "kind": "np_fixed", "path": "runs/opp_pool_current/lopunny",
     "deck": "runs/opp_pool_current/lopunny/deck.csv", "w": 0.12},
    {"id": "cur@alakazam", "kind": "np_fixed", "path": "runs/opp_pool_current/alakazam",
     "deck": "runs/opp_pool_current/alakazam/deck.csv", "w": 0.10},
    {"id": "cur@dragapult", "kind": "np_fixed", "path": "runs/opp_pool_current/dragapult",
     "deck": "runs/opp_pool_current/dragapult/deck.csv", "w": 0.05},
    {"id": "trmewtwo_ppo2", "kind": "rule_agent", "agent": "trmewtwo_ppo2",
     "deck": "agents/trmewtwo_ppo2/deck.csv", "w": 0.03},
]

COUNTERPART = {"hydrapple": "ogerpon", "ogerpon": "hydrapple"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=("hydrapple", "ogerpon"))
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2),
                    help="2 = 第2周: 下位段を減らし kang_cur を主敵化、anchor は第1周の成果")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    arch = args.arch
    if args.stage == 2:
        # 第1周でgreedy/playbook段は攻略済み（gauntlet 81%/85%）。露出を最強段へ移す
        KANG_LADDER[0]["w"] = 0.02   # glethal2@kang
        KANG_LADDER[1]["w"] = 0.05   # kang_playbook2
        KANG_LADDER[2]["w"] = 0.10   # kang_v0
        KANG_LADDER[3]["w"] = 0.28   # kang_cur


    from ptcg.ml.rl.config import PPOConfig
    from ptcg.ml.rl.league import load_state, save_state

    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_actors import ensure_run

    suffix = "k" if args.stage == 1 else "k2"
    anchor = (f"runs/bc_opp_{arch}_kseed/best.pt" if args.stage == 1
              else f"runs/anchors_k/{arch}.pt")  # 第1周best のnpz→pt変換を事前に置く
    cfg = PPOConfig().__dict__ | {
        "run_name": f"ppo_opp_{arch}_{suffix}",
        "anchor_ckpt": anchor,
        "deck_csv": f"runs/opp_pool_current/{arch}/deck.csv",
    }
    cfg_path = REPO_ROOT / "configs" / f"ppo_opp_{arch}_{suffix}.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))

    run = REPO_ROOT / "runs" / f"ppo_opp_{arch}_{suffix}"
    ensure_run(run, str(cfg_path))
    state = load_state(run / "league_state.json")
    state["buckets"] = {"latest": 0.40, "snapshot": 0.25, "fixed": 0.35}
    cp = COUNTERPART[arch]
    fixed = [dict(c) for c in KANG_LADDER + RETENTION] + [
        {"id": f"cur@{cp}", "kind": "np_fixed", "path": f"runs/opp_pool_current/{cp}",
         "deck": f"runs/opp_pool_current/{cp}/deck.csv", "w": 0.05},
    ]
    for c in fixed:
        if c["kind"] == "np_fixed":
            c["path"] = str((REPO_ROOT / c["path"]).resolve())
        c.setdefault("ema", 0.5)
        c.setdefault("games", 0)
    state["fixed"] = fixed
    save_state(state, run / "league_state.json")
    print(f"[setup] {run.name}: fixed×{len(fixed)} (kang段階4 + 通常{len(fixed)-4})")

    if args.preflight:
        import os
        os.environ.setdefault("PTCG_CG_DIR",
                              str(REPO_ROOT / "data/sample_submission/sample_submission/cg"))
        from ptcg.engine import ensure_cg_importable
        from ptcg.ml.np_forward import NpPolicy
        from ptcg.ml.rl.actor import _np_agent_argmax, _rule_agent
        from ptcg.ml.vocab import load_tables
        from pathlib import Path

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
                print(f"  [preflight {arch}] {c['id']:18s} ok result={r} steps={steps} w={c['w']}")
            except Exception as e:
                print(f"  [preflight {arch}] {c['id']:18s} ★NG {type(e).__name__}: {e}")
                ok = False
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""温度プローブ: argmax でなく温度サンプリングにしたとき勝率が動くかを測る。

「勝率≈0 は決定論の轍（同じ負け筋の反復）か、真の相性の壁か」を判別する。
自側 = 温度サンプリング（actor の学習時と同じ経路）、相手 = argmax。

    poetry run python scripts/temp_probe.py --me runs/opp_frozen_v1/ogerpon \
        --opp runs/opp_frozen_v1/kangaskhan --temp 1.0 --games 50 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PTCG_CG_DIR", str(REPO_ROOT / "data/sample_submission/sample_submission/cg"))

from ptcg.ml.features import featurize
from ptcg.ml.np_forward import NpPolicy
from ptcg.ml.rl.actor import _np_agent_argmax
from ptcg.ml.rl.masks import guard_mask
from ptcg.ml.rl.sample import sample_single
from ptcg.ml.vocab import load_tables


def sample_agent(policy, tables, temp: float, rng):
    def agent(obs):
        sel = obs.get("select") or {}
        n = len(sel.get("option") or [])
        k_min, k_max = int(sel.get("minCount", 0)), int(sel.get("maxCount", 1))
        fallback = list(range(max(min(k_max, n), k_min)))
        feats = featurize(obs, tables)
        if feats is None:
            return fallback
        mask, noop_allow = guard_mask(feats, int(sel.get("type", -1)))
        if k_max <= 1:
            out = policy.score(feats)
            if out is None:
                return fallback
            opt_logits, noop_logit, _ = out
            K = len(opt_logits)
            act, _, _ = sample_single(opt_logits, mask, noop_logit,
                                      noop_allow and bool(feats["noop_allowed"]),
                                      rng, temperature=temp)
            return [] if act == K else [act]
        mv = policy.choose(feats, extra_mask=mask)
        return mv if isinstance(mv, list) and len(mv) >= k_min else fallback
    return agent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--me", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tables = load_tables(REPO_ROOT / "data/ml/cards.npz", REPO_ROOT / "data/ml/attacks.npz")
    me_pol = NpPolicy.load(f"{a.me}/weights.npz", f"{a.me}/config.json")
    op_pol = NpPolicy.load(f"{a.opp}/weights.npz", f"{a.opp}/config.json")
    assert me_pol is not None and op_pol is not None
    rng = np.random.default_rng(a.seed)
    me = sample_agent(me_pol, tables, a.temp, rng)
    op = _np_agent_argmax(op_pol, tables)
    my_deck = [int(x) for x in open(f"{a.me}/deck.csv").read().split()]
    op_deck = [int(x) for x in open(f"{a.opp}/deck.csv").read().split()]

    from ptcg.engine import ensure_cg_importable

    ensure_cg_importable()
    from cg.game import battle_finish, battle_select, battle_start

    w = l = d = 0
    for g in range(a.games):
        seat = g % 2
        decks = (my_deck, op_deck) if seat == 0 else (op_deck, my_deck)
        obs, start = battle_start(list(decks[0]), list(decks[1]))
        assert start.errorPlayer < 0
        steps = 0
        try:
            while obs["current"]["result"] < 0 and steps < 4000:
                mv = me(obs) if obs["current"]["yourIndex"] == seat else op(obs)
                obs = battle_select(mv)
                steps += 1
            r = obs["current"]["result"]
        finally:
            battle_finish()
        if r == 2:
            d += 1
        elif r == seat:
            w += 1
        else:
            l += 1
    print(json.dumps({"temp": a.temp, "w": w, "l": l, "d": d, "seed": a.seed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

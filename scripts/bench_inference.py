#!/usr/bin/env python
"""配備方策の numpy 推論レイテンシ計測（レポート用の一次データ）。

提出物と同一経路（featurize → guard_mask → NpPolicy.choose）を実戦ゲーム中に計測する。
マイクロベンチではなく実対戦なので、オプション数の分布が本番と同じになる。

    poetry run python scripts/bench_inference.py --agent kangaskhan_dh --games 10
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", default="runs/selfplay_dh/kangaskhan")
    ap.add_argument("--opponent", default="greedy_lethal2")
    ap.add_argument("--games", type=int, default=10)
    args = ap.parse_args()

    from ptcg.cards import read_deck_csv
    from ptcg.engine import play_game
    from ptcg.ml.features import featurize
    from ptcg.ml.rl.masks import guard_mask
    from ptcg.ml.vocab import load_tables
    from ptcg.ml.np_forward import NpPolicy
    from ptcg.ml.rl.actor import _rule_agent

    pdir = REPO_ROOT / args.policy_dir
    pol = NpPolicy.load(pdir / "weights.npz", pdir / "config.json")
    tables = load_tables(REPO_ROOT / "data/ml/cards.npz", REPO_ROOT / "data/ml/attacks.npz")
    assert tables is not None, "data/ml/{cards,attacks}.npz が読めない"
    deck = read_deck_csv(pdir / "deck.csv")

    lat: list[float] = []
    n_model = n_fallback = 0

    def timed_agent(obs_dict):
        nonlocal n_model, n_fallback
        sel = obs_dict.get("select") or {}
        n = len(sel.get("option") or [])
        k = max(min(int(sel.get("maxCount", 1)), n), int(sel.get("minCount", 0)))
        t0 = time.perf_counter()
        feats = featurize(obs_dict, tables)
        mv = None
        if feats is not None:
            mask, _noop = guard_mask(feats, int(sel.get("type", -1)))
            cand = pol.choose(feats, extra_mask=mask)
            if isinstance(cand, list) and len(cand) >= int(sel.get("minCount", 0)):
                ok = (all(0 <= x < n for x in cand) and len(set(cand)) == len(cand)
                      and len(cand) <= int(sel.get("maxCount", n)))
                if ok:
                    mv = cand
        lat.append((time.perf_counter() - t0) * 1000.0)
        if mv is None:
            n_fallback += 1
            return list(range(k))
        n_model += 1
        return mv

    opp = _rule_agent(args.opponent)
    for g in range(args.games):
        play_game(timed_agent, opp, deck, deck) if g % 2 == 0 else play_game(opp, timed_agent, deck, deck)

    lat.sort()
    print(f"policy={args.policy_dir} games={args.games} decisions={len(lat)} "
          f"model={n_model} fallback={n_fallback} ({n_model / max(len(lat), 1):.1%} model rate)")
    print(f"  mean {st.mean(lat):.2f} ms | p50 {lat[len(lat)//2]:.2f} | p95 {lat[int(len(lat)*.95)]:.2f} "
          f"| p99 {lat[int(len(lat)*.99)]:.2f} | max {max(lat):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

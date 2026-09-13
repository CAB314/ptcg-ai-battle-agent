"""PPO actor ワーカー — 1プロセス=1相手スペック=N戦（リーク対策で自然終了）。

自席は **提出物と同一の numpy forward** でサンプリング（配備との方策同一性の保険）。
軌跡は traj.PackWriter で inbox へ。weights はゲーム境界で version.json を見て hot-reload。

起動: python -m ptcg.ml.rl.actor <run_dir> <actor_id> <opp_spec_json>
（supervisor = scripts/run_actors.py が抽選済みスペックを渡して respawn する）
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..features import featurize
from ..np_forward import NpPolicy
from ..vocab import load_tables
from .config import PPOConfig
from .league import K_AGENT, K_MIRROR, K_NP_FIXED, K_SNAPSHOT
from .masks import guard_mask
from .sample import sample_single
from .traj import PackWriter

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_policy(policy_dir: Path):
    ver = 0
    vj = policy_dir / "version.json"
    if vj.exists():
        try:
            ver = int(json.loads(vj.read_text())["version"])
        except Exception:
            ver = 0
    pol = NpPolicy.load(policy_dir / "weights.npz", policy_dir / "config.json")
    return pol, ver


def _np_agent_argmax(policy, tables):
    """スナップショット/凍結BC 用の argmax エージェント（greedy フォールバック付き）。"""
    def agent(obs_dict):
        sel = obs_dict.get("select") or {}
        n = len(sel.get("option") or [])
        k = max(min(int(sel.get("maxCount", 1)), n), int(sel.get("minCount", 0)))
        feats = featurize(obs_dict, tables)
        if feats is not None:
            mask, noop = guard_mask(feats, int(sel.get("type", -1)))
            mv = policy.choose(feats, extra_mask=mask)
            if isinstance(mv, list) and len(mv) >= int(sel.get("minCount", 0)):
                ok = all(0 <= x < n for x in mv) and len(set(mv)) == len(mv) and len(mv) <= int(sel.get("maxCount", n))
                if ok:
                    return mv
        return list(range(k))
    return agent


def _rule_agent(agent_name: str):
    from ptcg.engine import load_agent

    os.environ["PTCG_AGENT_DIR"] = str(REPO_ROOT / "agents" / agent_name)
    return load_agent(REPO_ROOT / "agents" / agent_name / "main.py")


def run_actor(run_dir: str, actor_id: int, spec: dict) -> int:
    os.nice(19)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run = Path(run_dir)
    cfg = PPOConfig.load(run / "config.json")
    stop_file = run / "STOP"
    tables = load_tables(REPO_ROOT / "data/ml/cards.npz", REPO_ROOT / "data/ml/attacks.npz")
    # tables=None（版不一致・ファイル欠落）だと featurize が毎回 None を返し、
    # add_step が一度も呼ばれず inbox が空のまま learner が永久待機する。
    # sbatch の watchdog は learner の「死亡」しか見ないので止まらず、GPU課金だけ溶ける。
    assert tables is not None, (
        "cards.npz/attacks.npz がロードできない（VOCAB_VERSION 不一致か未配置）。"
        "scripts/build_vocab.py を実行し data/ml/ を同期すること")
    policy_dir = run / "policy" / "latest"
    policy, version = _load_policy(policy_dir)
    assert policy is not None, "policy latest がロードできない"
    my_deck = [int(x) for x in (REPO_ROOT / cfg.deck_csv).read_text().split()]

    # 相手の準備
    kind = spec["kind"]
    opp_np = None
    opp_agent = None
    if kind == K_MIRROR:
        opp_deck = my_deck
    else:
        opp_deck = [int(x) for x in (REPO_ROOT / spec["deck"]).read_text().split()]
        if kind in (K_SNAPSHOT, K_NP_FIXED):
            p = Path(spec["path"])
            opp_np = NpPolicy.load(p / "weights.npz", p / "config.json")
            assert opp_np is not None, f"相手 policy ロード失敗: {p}"
            opp_agent = _np_agent_argmax(opp_np, tables)
        elif kind == K_AGENT:
            opp_agent = _rule_agent(spec["agent"])

    from ptcg.engine import ensure_cg_importable

    ensure_cg_importable()
    from cg.game import battle_finish, battle_select, battle_start

    rng = np.random.default_rng((os.getpid() * 7919 + actor_id) % 2**32)
    writer = PackWriter(run / "inbox", actor_id, version,
                        flush_games=8, flush_secs=45.0)
    games_done = 0
    wins = 0
    draws = 0
    turn_guard = {"turn": -1, "count": 0, "game": 0}

    def my_move(obs_dict, ep_seat):
        sel = obs_dict.get("select") or {}
        n = len(sel.get("option") or [])
        k_min = int(sel.get("minCount", 0))
        k_max = int(sel.get("maxCount", 1))
        k_fallback = max(min(k_max, n), k_min)
        cur = obs_dict.get("current") or {}
        # アンチループ（配備 main.py と同じ思想。学習中の暴走ゲームも遮断）
        t = int(cur.get("turn") or 0)
        if t < turn_guard["turn"]:
            turn_guard.update({"turn": t, "count": 0, "game": 0})
        if t != turn_guard["turn"]:
            turn_guard["turn"] = t
            turn_guard["count"] = 0
        turn_guard["count"] += 1
        turn_guard["game"] += 1
        if turn_guard["count"] > 40 or turn_guard["game"] > 800:
            for i, o in enumerate(sel.get("option") or []):
                if int(o.get("type", -1)) == 14:  # END
                    return [i]
            return list(range(k_fallback))
        feats = featurize(obs_dict, tables)
        if feats is None:
            return list(range(k_fallback))
        mask, noop_allow = guard_mask(feats, int(sel.get("type", -1)))
        out = policy.score(feats)
        if out is None:
            return list(range(k_fallback))
        opt_logits, noop_logit, value = out
        K = len(opt_logits)
        if k_max <= 1:
            act, logp, ent = sample_single(
                opt_logits, mask, noop_logit, noop_allow and bool(feats["noop_allowed"]),
                rng, temperature=cfg.sample_temperature,
            )
            writer.add_step(feats, mask, noop_allow, act, logp, value, True, ep_seat)
            return [] if act == K else [act]
        mv = policy.choose(feats, extra_mask=mask)
        if not isinstance(mv, list):
            mv = list(range(k_fallback))
        writer.add_step(feats, mask, noop_allow, min(mv[0] if mv else K, K), 0.0, value, False, ep_seat)
        return mv

    while games_done < cfg.actor_games_per_proc and not stop_file.exists():
        # weights hot-reload（ゲーム境界）
        pol2, ver2 = _load_policy(policy_dir)
        if pol2 is not None and ver2 != version:
            policy, version = pol2, ver2
            writer.flush()
            writer.set_version(version)
        turn_guard.update({"turn": -1, "count": 0, "game": 0})
        my_seat = games_done % 2
        decks = (my_deck, opp_deck) if my_seat == 0 else (opp_deck, my_deck)
        obs, start = battle_start(list(decks[0]), list(decks[1]))
        if start.errorPlayer >= 0:
            raise RuntimeError("deck error")
        base = games_done * 2
        steps = 0
        try:
            while obs["current"]["result"] < 0 and steps < 4000:
                who = obs["current"]["yourIndex"]
                if kind == K_MIRROR:
                    mv = my_move(obs, base + who)
                elif who == my_seat:
                    mv = my_move(obs, base + my_seat)
                else:
                    mv = opp_agent(obs)
                obs = battle_select(mv)
                steps += 1
            result = obs["current"]["result"]
        finally:
            battle_finish()
        # 終端報酬
        def rw(seat):
            if result == 2:
                return cfg.draw_reward
            return 1.0 if result == seat else -1.0
        if kind == K_MIRROR:
            writer.end_episode(base + 0, rw(0))
            writer.end_episode(base + 1, rw(1))
        else:
            writer.end_episode(base + my_seat, rw(my_seat))
        my_result = 2 if result == 2 else (1 if result == my_seat else 0)
        writer.end_game(my_result, 0, 0)
        wins += 1 if my_result == 1 else 0
        draws += 1 if my_result == 2 else 0
        games_done += 1
    writer.flush()
    # supervisor へ相手別戦績を返す（リーグ EMA 更新用。hash() はプロセス毎に変わるためファイル渡し）
    res_dir = run / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / f"r{int(time.time() * 1000)}_{actor_id}.json").write_text(json.dumps(
        {"opp": spec["id"], "wins": wins, "draws": draws, "games": games_done}
    ))
    return 0


def main() -> int:
    run_dir, actor_id, spec_json = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    return run_actor(run_dir, actor_id, json.loads(spec_json))


if __name__ == "__main__":
    raise SystemExit(main())

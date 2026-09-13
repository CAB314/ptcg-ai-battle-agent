"""相手リーグ（league_state.json）— 抽選・PFSP-lite・EMA勝率。

バケット: latest（最新方策とのミラー）/ snapshot（過去の自分）/ fixed（凍結BC・ルール勢）。
PFSP-lite: バケット内の相手セル i を q_i ∝ max(floor, 1−EMA勝率_i)² で抽選
（勝ち切った相手も floor 分は残して回帰を検知する）。

state の更新（EMA・スナップショット追加）は learner 側が行い、supervisor/actor は
読み取りと抽選のみ（ファイルは atomic rename で更新）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# 相手スペックの kind
K_MIRROR = "mirror"          # 最新方策と同一インスタンスで両席（両席とも軌跡採取）
K_SNAPSHOT = "np_snapshot"   # 過去スナップショットの NpPolicy（argmax）
K_NP_FIXED = "np_fixed"      # 凍結 BC 等の NpPolicy（argmax）
K_AGENT = "rule_agent"       # agents/<name> のルール勢（load_agent。1プロセス1種まで）


def default_state(repo_root: str, anchor_export: str, my_deck: str) -> dict:
    """初期 league_state（スナップショット0個・固定勢5種・デッキ分布は7/15メタ）。"""
    decks = {
        # 実メタ比例（**2026-07-31 実測シェア**に更新。週次で手動更新）
        # 旧版は 7/15 時点（Alakazam 0.42 / Grimmsnarl 0.12）で、7月下旬のレジーム転換
        # （Grimmsnarl 0.62 / Alakazam 0.045）と正反対になっていた。この分布で学習すると
        # 実戦の 62% を占めるミラーがほとんど回らない。
        "meta": [
            ["decks/meta_marnie_s_grimmsnarl_ex.csv", 0.62],
            ["decks/meta_mega_kangaskhan_ex.csv", 0.09],
            ["decks/meta_thwackey.csv", 0.05],
            ["decks/meta_mega_lopunny_ex.csv", 0.05],
            ["decks/meta_alakazam.csv", 0.045],
            ["decks/meta_team_rocket_s_mewtwo_ex.csv", 0.04],
            ["decks/meta_dragapult_ex.csv", 0.04],
            ["decks/meta_cynthia_s_garchomp_ex.csv", 0.035],
            ["decks/meta_teal_mask_ogerpon_ex.csv", 0.03],
        ],
        # 弱点重点（実測の苦手: Teal Ogerpon は greedy 操縦でも 51-59% しか取れない）
        "weak": [
            ["decks/meta_teal_mask_ogerpon_ex.csv", 0.35],
            ["decks/meta_mega_kangaskhan_ex.csv", 0.25],
            ["decks/greattusk_crustle_lo.csv", 0.2],
            ["decks/prop_d5_crustle_stall.csv", 0.2],
        ],
        "uniform": [],  # 空 = decks/ 全体から等確率（supervisor 側で展開）
    }
    fixed = [
        {"id": "bc_frozen@alakazam", "kind": K_NP_FIXED, "path": anchor_export,
         "deck": my_deck, "ema": 0.5, "games": 0},
        {"id": "greedy_lethal2@mix", "kind": K_AGENT, "agent": "greedy_lethal2",
         "deck": "mix", "ema": 0.5, "games": 0},
        {"id": "greedy_first@mix", "kind": K_AGENT, "agent": "greedy_first",
         "deck": "mix", "ema": 0.5, "games": 0},
        {"id": "kangaskhan_playbook2", "kind": K_AGENT, "agent": "kangaskhan_playbook2",
         "deck": "agents/kangaskhan_playbook2/deck.csv", "ema": 0.5, "games": 0},
        {"id": "greattusk_lo_playbook", "kind": K_AGENT, "agent": "greattusk_lo_playbook",
         "deck": "agents/greattusk_lo_playbook/deck.csv", "ema": 0.5, "games": 0},
    ]
    return {
        "policy_version": 0,
        "buckets": {"latest": 0.5, "snapshot": 0.3, "fixed": 0.2},
        "pfsp_floor": 0.05,
        "snapshots": [],
        "fixed": fixed,
        "decks": decks,
        "deck_mix": {"meta": 0.55, "weak": 0.25, "uniform": 0.20},
    }


def load_state(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_state(state: dict, path) -> None:
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _pfsp_pick(cells: list[dict], floor: float, rng):
    # "w" = 静的重み（例: 実メタシェア比例）。省略時 1.0 で従来挙動と一致。
    ws = [float(c.get("w", 1.0)) * max(floor, 1.0 - float(c.get("ema", 0.5))) ** 2 for c in cells]
    s = sum(ws)
    r = rng.random() * s
    acc = 0.0
    for c, w in zip(cells, ws):
        acc += w
        if r <= acc:
            return c
    return cells[-1]


def sample_deck(state: dict, uniform_decks: list[str], rng) -> str:
    mix = state["deck_mix"]
    r = rng.random()
    if r < mix["meta"]:
        pool = state["decks"]["meta"]
    elif r < mix["meta"] + mix["weak"]:
        pool = state["decks"]["weak"]
    else:
        pool = [[d, 1.0] for d in uniform_decks]
    total = sum(w for _, w in pool)
    r2 = rng.random() * total
    acc = 0.0
    for d, w in pool:
        acc += w
        if r2 <= acc:
            return d
    return pool[-1][0]


def sample_opponent(state: dict, uniform_decks: list[str], rng) -> dict:
    """相手スペック {kind, id, path/agent, deck} を抽選する。"""
    b = state["buckets"]
    has_snap = bool(state["snapshots"])
    p_latest = b["latest"] + (0 if has_snap else b["snapshot"])
    r = rng.random()
    if r < p_latest:
        return {"kind": K_MIRROR, "id": "latest_mirror", "deck": None}
    if has_snap and r < p_latest + b["snapshot"]:
        c = _pfsp_pick(state["snapshots"], state["pfsp_floor"], rng)
        return {"kind": K_SNAPSHOT, "id": c["id"], "path": c["path"],
                "deck": c.get("deck") or sample_deck(state, uniform_decks, rng)}
    c = _pfsp_pick(state["fixed"], state["pfsp_floor"], rng)
    deck = c["deck"]
    if deck == "mix":
        deck = sample_deck(state, uniform_decks, rng)
    return {"kind": c["kind"], "id": c["id"], "path": c.get("path"),
            "agent": c.get("agent"), "deck": deck}


def update_ema(state: dict, opp_id: str, my_wins: int, games: int, alpha: float = 0.02) -> None:
    """learner 側: 相手セルの EMA 勝率（自分視点）を更新する。"""
    for pool in (state["snapshots"], state["fixed"]):
        for c in pool:
            if c["id"] == opp_id:
                ema = float(c.get("ema", 0.5))
                for _ in range(games):
                    pass
                if games > 0:
                    wr = my_wins / games
                    k = 1.0 - (1.0 - alpha) ** games
                    c["ema"] = ema * (1 - k) + wr * k
                    c["games"] = int(c.get("games", 0)) + games
                return

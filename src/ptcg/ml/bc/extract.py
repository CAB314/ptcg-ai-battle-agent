"""episode JSON → 決定レコード列（off-by-one 補正とフィルタの正典）。

スキーマは実データで検証済み（docs/experiments-log.md 2026-07-16 / plan）:
- 決定点 = steps[t][i].status == 'ACTIVE' かつ observation.select が非 null
- 教師ペア = (steps[t][i].observation, steps[t+1][i].action)  ← off-by-one
- minCount==0 の空 action は「pass」の正解ラベル
- statuses==['DONE','DONE'] かつ rewards 両方非 None のみ採用（TIMEOUT/ERROR排除）
- t=1 の 60 枚 action はデッキ提出（select=None なので決定点に含まれない）

教師（チーム）フィルタは学習時に適用する方針のため、ここでは全チームの決定を返す。
"""

from __future__ import annotations

from ..schema import as_int, g
from ..vocab import C_CARD_TYPE, C_HP, CARD_ROWS


def episode_ok(ep: dict) -> bool:
    statuses = ep.get("statuses") or []
    rewards = ep.get("rewards") or []
    if list(statuses) != ["DONE", "DONE"]:
        return False
    if len(rewards) != 2 or any(not isinstance(r, (int, float)) for r in rewards):
        return False
    return True


def decks_of(ep: dict):
    """各 agent の 60 枚デッキ（最初の 60 要素 action）。見つからなければ None。"""
    decks = [None, None]
    steps = ep.get("steps") or []
    for st in steps[:6]:
        for i in (0, 1):
            if i < len(st) and decks[i] is None:
                a = st[i].get("action")
                if isinstance(a, list) and len(a) == 60:
                    decks[i] = [as_int(x) for x in a]
        if decks[0] is not None and decks[1] is not None:
            break
    return decks


def ace_of(deck, tables) -> int:
    """「2枚以上採用の最高HPポケモン」のカードID（アーキタイプの代理ラベル）。"""
    if not deck:
        return 0
    counts: dict[int, int] = {}
    for cid in deck:
        counts[cid] = counts.get(cid, 0) + 1
    best_id, best_hp = 0, -1.0
    cards = tables["cards"]
    for cid, k in counts.items():
        if k < 2 or not (0 < cid < CARD_ROWS):
            continue
        row = cards[cid]
        if int(row[C_CARD_TYPE]) != 0:  # POKEMON のみ
            continue
        if row[C_HP] > best_hp:
            best_id, best_hp = cid, float(row[C_HP])
    return best_id


def iter_decisions(ep: dict):
    """(agent_idx, obs_dict, label(list[int]), turn) を返すジェネレータ。

    ラベル健全性（重複なし・範囲内・minCount<=len<=maxCount）を破るものは
    drop し、呼び出し側で計数できるよう ('drop', 理由) を yield する。
    """
    steps = ep.get("steps") or []
    for t in range(len(steps) - 1):
        st = steps[t]
        nxt = steps[t + 1]
        for i in (0, 1):
            if i >= len(st) or i >= len(nxt):
                continue
            cell = st[i]
            if cell.get("status") != "ACTIVE":
                continue
            obs = cell.get("observation") or {}
            sel = obs.get("select")
            if sel is None:
                continue
            options = sel.get("option") or []
            label = nxt[i].get("action")
            if not isinstance(label, list):
                yield ("drop", "label_not_list", i, None, 0)
                continue
            label = [as_int(x, -1) for x in label]
            n = len(options)
            k_min = as_int(sel.get("minCount"), 0)
            k_max = as_int(sel.get("maxCount"), n)
            if any(x < 0 or x >= n for x in label) or len(set(label)) != len(label):
                yield ("drop", "label_out_of_range", i, None, 0)
                continue
            if not (k_min <= len(label) <= k_max):
                yield ("drop", "label_count", i, None, 0)
                continue
            turn = as_int(g(obs.get("current") or {}, "turn"), 0)
            yield ("ok", None, i, (obs, label), turn)

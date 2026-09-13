#!/usr/bin/env python
"""最終2提出のラダーリプレイを解析し、デッキの想定どおりに打てていたかを実測で確認する。

Writeup §7 / Fig. 11 の「このデッキは何をするデッキか」という主張を、
**カードテキストの読みではなく実際の対戦ログ**で裏づけるための一次データ。

入力: data/episodes_ours/<submission>/episode-<id>-replay.json（`fetch_our_episodes.py` が取得）
      窓は `_matchups.json`（1,000試合・対面と勝敗つき）に揃える。

**イベント列の復元**（重要）: 各エージェントの `observation.logs` は「そのエージェントが前回促された時点からの
イベント」で、促されていない間は同じバッチが再掲される。そのため *自エージェントが ACTIVE のステップのみ* を
連結する。この方法で相手視点から復元した列と件数が一致することを確認済み（差は最終ターンの末尾1件以内）。
最後に、自分の最終 ACTIVE 以降に相手側で起きたイベントを重複を除いて連結する（試合終了時の KO を取り込む）。
出力: docs/data/replay_behavior_<agent>.json（集計）と <agent>_games.jsonl（試合ごとの素データ）

主な測定項目:
  * 壁（Crustle / Cornerstone / Dipplin）が実際にバトル場に立った試合の割合と、その試合の勝率
  * 相手のワザ1回ごとに「ダメージが0だったか」を集計し、**攻撃側が ex か / 特性を持つか**で層別する
    → Crustle は ex、Cornerstone は特性持ちを止めるという「二軸の壁」の主張の直接検証
  * 自分のワザ別の使用回数と与ダメージ（どのカードで勝っていたか）
  * Meganium 在場時に基本草1枚が2個分として提供されるか（盤面の energies 配列）と、
    Syrup Storm / Giant Bouquet の実測ダメージが2倍化後のエネルギー数と整合するか
  * 回復（Jumbo Ice Cream）・道具（Hero's Cape）・スタジアム（Forest of Vitality）の使用回数

    poetry run python scripts/analyze_replay_behavior.py --workers 48
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import REPO_ROOT

CACHE = REPO_ROOT / "data/episodes_ours"
OUT = REPO_ROOT / "docs/data"
OUR_TEAM = "cabbage patch"
SUBS = {"kangaskhan_dh": 55541008, "hydrapple_k2": 55537575}

# LogType（cg.api）
SHUFFLE, HAS_BASIC, TURN_START, TURN_END, DRAW, DRAW_REV = 0, 1, 2, 3, 4, 5
MOVE_CARD, MOVE_CARD_REV, SWITCH, CHANGE, PLAY, ATTACH = 6, 7, 8, 9, 10, 11
EVOLVE, DEVOLVE, MOVE_ATTACHED, ATTACK, HP_CHANGE = 12, 13, 14, 15, 16
RESULT = 23

BASIC_G = 1  # Basic {G} Energy の cardId
WALLS = {345: "Crustle", 117: "Cornerstone", 921: "Dipplin"}
KEY_EVO = {"kangaskhan_dh": {345: "Crustle"},
           "hydrapple_k2": {710: "Meganium", 150: "Hydrapple ex", 919: "Mega Meganium ex"}}
COUNT_PLAYS = {1147: "Jumbo Ice Cream", 1261: "Forest of Vitality", 1264: "Battle Cage",
               1182: "Boss's Orders", 1094: "Bug Catching Set", 1227: "Lillie's Determination"}
COUNT_ATTACH = {1159: "Hero's Cape", BASIC_G: "Basic G Energy", 18: "Grow Grass Energy",
                11: "Mist Energy", 14: "Spiky Energy", 20: "Rock Fighting Energy"}
SCALING_ATK = {195: "Syrup Storm", 1325: "Giant Bouquet", 120: "Myriad Leaf Shower"}


def card_facts() -> dict:
    """エンジンのカード表から ex / 特性の有無 / 弱点 / ワザ（名前・基礎打点）を引く。"""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ptcg.engine import ensure_cg_importable, engine_attacks, engine_card_data
    ensure_cg_importable()
    cards, atks = {}, {}
    for c in engine_card_data():
        cards[int(c.cardId)] = {"name": c.name, "ex": bool(c.ex or c.megaEx), "mega": bool(c.megaEx),
                                "ability": bool(c.skills), "weak": int(c.weakness or 0),
                                "hp": int(c.hp or 0)}
    for a in engine_attacks():
        atks[int(a.attackId)] = {"name": a.name, "damage": int(a.damage or 0)}
    return {"cards": cards, "attacks": atks}


FACTS: dict = {}


def _init(facts):
    global FACTS
    FACTS = facts


def _observations(d: dict, us: int) -> list[dict]:
    """自エージェントが ACTIVE のステップの観測を時系列で返す（末尾に相手側の残りを足す）。"""
    steps = d.get("steps") or []
    obs, last = [], -1
    for i, step in enumerate(steps):
        if len(step) <= us or not step[us]:
            continue
        if (step[us] or {}).get("status") != "ACTIVE":
            continue
        obs.append((step[us].get("observation") or {}))
        last = i
    # 自分の最終 ACTIVE より後に相手が動いた分（試合を決めた KO など）を重複なく足す
    tail: list[dict] = []
    for step in steps[last + 1:]:
        for j, ag in enumerate(step):
            if j == us or not ag or ag.get("status") != "ACTIVE":
                continue
            batch = ((ag.get("observation") or {}).get("logs")) or []
            k = min(len(batch), len(tail))
            while k > 0 and tail[-k:] != batch[:k]:
                k -= 1
            tail.extend(batch[k:])
    if tail:
        obs.append({"logs": tail, "current": {}})
    return obs


def _mons(player: dict | None) -> list:
    if not player:
        return []
    return [m for m in (player.get("active") or []) + (player.get("bench") or []) if m]


def one_game(args) -> dict | None:
    path, agent, opp_label, reward = args
    try:
        d = json.loads(Path(path).read_text())
    except Exception as e:  # 壊れたキャッシュは落とさず記録する
        return {"agent": agent, "path": str(path), "error": repr(e)[:120]}
    names = (d.get("info") or {}).get("TeamNames") or []
    if OUR_TEAM not in names:
        return None
    us = names.index(OUR_TEAM)
    cards, atk_tbl = FACTS["cards"], FACTS["attacks"]

    r = {"agent": agent, "episode": Path(path).stem, "opp": opp_label, "win": int(reward > 0),
         "our_turns": 0, "steps": len(d.get("steps") or []),
         "evolved": {}, "wall_active_turns": Counter(), "our_attacks": Counter(),
         "our_damage": Counter(), "plays": Counter(), "attaches": Counter(),
         "heal": 0, "spiky_returns": 0, "opp_attacks": [], "scaling": [],
         "doubling_obs": [], "opp_attack_total": 0}
    prev_active = None          # 相手のワザを受けた時点の自分のバトルポケモン
    prev_active_serial = None
    prev_us_board = None        # 直前の自分の盤面（打点スケーリングの検証用）

    for o in _observations(d, us):
        logs = o.get("logs") or []
        cur = o.get("current") or {}
        players = cur.get("players") or []
        us_p = players[us] if len(players) > us else None
        cur_atk = None
        for lg in logs:
            t = lg.get("type")
            if t == TURN_START:
                if lg.get("playerIndex") == us:
                    r["our_turns"] += 1
                    if prev_active in WALLS:
                        r["wall_active_turns"][WALLS[prev_active]] += 1
            elif t == EVOLVE and lg.get("playerIndex") == us:
                cid = lg.get("cardId")
                if cid in KEY_EVO.get(agent, {}) and str(cid) not in r["evolved"]:
                    r["evolved"][str(cid)] = r["our_turns"]
            elif t == PLAY and lg.get("playerIndex") == us:
                if lg.get("cardId") in COUNT_PLAYS:
                    r["plays"][COUNT_PLAYS[lg["cardId"]]] += 1
            elif t == ATTACH and lg.get("playerIndex") == us:
                if lg.get("cardId") in COUNT_ATTACH:
                    r["attaches"][COUNT_ATTACH[lg["cardId"]]] += 1
            elif t == ATTACK:
                who = lg.get("playerIndex")
                aid, cid = lg.get("attackId"), lg.get("cardId")
                base = (atk_tbl.get(aid) or {}).get("damage", 0)
                cur_atk = {"who": who, "aid": aid, "cid": cid, "base": base,
                           "dmg_to_us": 0, "dmg_to_opp": 0, "dmg_to_active": 0}
                if who == us:
                    r["our_attacks"][(atk_tbl.get(aid) or {}).get("name", str(aid))] += 1
                    if aid in SCALING_ATK and prev_us_board is not None:
                        cur_atk["scaling"] = {"atk": SCALING_ATK[aid], **prev_us_board}
                else:
                    r["opp_attack_total"] += 1
                    f = cards.get(cid) or {}
                    cur_atk["meta"] = {"ex": bool(f.get("ex")), "mega": bool(f.get("mega")),
                                       "ability": bool(f.get("ability")),
                                       "attacker": f.get("name", str(cid)), "target": prev_active}
            elif t == HP_CHANGE:
                v = lg.get("value") or 0
                mine = lg.get("playerIndex") == us
                if v > 0 and mine:
                    r["heal"] += v
                elif v < 0 and cur_atk is not None:
                    if mine:
                        cur_atk["dmg_to_us"] += -v
                        if lg.get("serial") == prev_active_serial:
                            cur_atk["dmg_to_active"] += -v
                    else:
                        cur_atk["dmg_to_opp"] += -v
        if cur_atk is not None:
            _close(r, cur_atk, us)
        # ステップ末で盤面のスナップショットを更新する
        if us_p:
            act = (us_p.get("active") or [None])[0]
            prev_active = act.get("id") if act else None
            prev_active_serial = act.get("serial") if act else None
            mons = _mons(us_p)
            basic_g = sum(1 for m in mons for c in (m.get("energyCards") or []) if c.get("id") == BASIC_G)
            provided_g = sum(1 for m in mons for e in (m.get("energies") or []) if e == BASIC_G)
            self_g = 0
            if act:
                self_g = sum(1 for e in (act.get("energies") or []) if e == BASIC_G)
            ids = {m.get("id") for m in mons}
            prev_us_board = {"basic_g_cards": basic_g, "provided_g": provided_g,
                             "active_provided_g": self_g, "meganium": 710 in ids}
            if 710 in ids and basic_g:
                r["doubling_obs"].append({"basic_g_cards": basic_g, "provided_g": provided_g})
    r["wall_active_turns"] = dict(r["wall_active_turns"])
    r["our_attacks"] = dict(r["our_attacks"])
    r["our_damage"] = dict(r["our_damage"])
    r["plays"] = dict(r["plays"])
    r["attaches"] = dict(r["attaches"])
    r["doubling_obs"] = r["doubling_obs"][:3]
    return r


def _close(r: dict, a: dict, us: int) -> None:
    """1回のワザの結果を確定させる。"""
    if a["who"] == us:
        name = (a.get("scaling") or {}).get("atk")
        r["our_damage"][FACTS["attacks"].get(a["aid"], {}).get("name", str(a["aid"]))] = \
            r["our_damage"].get(FACTS["attacks"].get(a["aid"], {}).get("name", str(a["aid"])), 0) + a["dmg_to_opp"]
        if name and a["dmg_to_opp"]:
            s = dict(a["scaling"]); s["damage"] = a["dmg_to_opp"]; s["base"] = a["base"]
            r["scaling"].append(s)
    else:
        m = a.get("meta") or {}
        r["opp_attacks"].append({"ex": m.get("ex"), "mega": m.get("mega"), "ability": m.get("ability"),
                                 "target": m.get("target"), "base": a["base"],
                                 "dmg": a["dmg_to_active"], "dmg_any": a["dmg_to_us"],
                                 "atk": FACTS["attacks"].get(a["aid"], {}).get("name", str(a["aid"]))})
        if a["dmg_to_opp"]:      # 相手のワザで相手側にダメージ = Spiky Energy の返し
            r["spiky_returns"] += a["dmg_to_opp"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="both", choices=[*SUBS, "both"])
    ap.add_argument("--limit", type=int, default=0, help="先頭N試合だけ（動作確認用）")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    facts = card_facts()
    agents = list(SUBS) if args.agent == "both" else [args.agent]
    for agent in agents:
        sub = SUBS[agent]
        cdir = CACHE / str(sub)
        mt = json.loads((cdir / "_matchups.json").read_text())
        jobs = []
        for m in mt:
            p = cdir / f"episode-{m['id']}-replay.json"
            if p.exists():
                jobs.append((str(p), agent, m["opp"], m["reward"]))
        if args.limit:
            jobs = jobs[: args.limit]
        print(f"[{agent}] {len(jobs)} 試合を解析（workers={args.workers}）", flush=True)
        rows = []
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(facts,)) as ex:
            for i, rec in enumerate(ex.map(one_game, jobs, chunksize=4), 1):
                if rec:
                    rows.append(rec)
                if i % 200 == 0:
                    print(f"   {i}/{len(jobs)}", flush=True)
        summarize(agent, rows)
    return 0


def summarize(agent: str, rows: list[dict]) -> None:
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    n = len(ok)
    wins = sum(r["win"] for r in ok)
    s: dict = {"agent": agent, "games": n, "wins": wins, "wr": round(100 * wins / n, 1) if n else None,
               "errors": len(err), "mean_our_turns": round(sum(r["our_turns"] for r in ok) / n, 1) if n else None}

    # 主要カードが場に出た試合の割合と、その条件つき勝率
    cond = {}
    for cid, label in KEY_EVO.get(agent, {}).items():
        got = [r for r in ok if str(cid) in r["evolved"]]
        no = [r for r in ok if str(cid) not in r["evolved"]]
        cond[label] = {
            "games_landed": len(got), "share": round(100 * len(got) / n, 1) if n else None,
            "wr_landed": round(100 * sum(x["win"] for x in got) / len(got), 1) if got else None,
            "wr_not_landed": round(100 * sum(x["win"] for x in no) / len(no), 1) if no else None,
            "median_turn": sorted(r["evolved"][str(cid)] for r in got)[len(got) // 2] if got else None,
        }
    s["key_cards"] = cond

    # 壁がバトル場に立ったターン数
    s["wall_active_turns_mean"] = {
        k: round(sum(r["wall_active_turns"].get(k, 0) for r in ok) / n, 2)
        for k in ("Crustle", "Cornerstone", "Dipplin") if any(r["wall_active_turns"].get(k) for r in ok)}
    s["games_with_wall_active"] = {
        k: sum(1 for r in ok if r["wall_active_turns"].get(k)) for k in s["wall_active_turns_mean"]}

    # 相手のワザ1回ごとの被ダメージを、攻撃側の ex / 特性で層別（打点0のワザは除く）
    tab = defaultdict(lambda: {"n": 0, "zero": 0, "dmg": 0})
    for r in ok:
        for a in r["opp_attacks"]:
            if not a["base"]:
                continue
            tgt = WALLS.get(a["target"], "other")
            kind = "mega-ex" if a.get("mega") else ("ex" if a["ex"] else "non-ex")
            key = (tgt, kind, "ability" if a["ability"] else "no-ability")
            c = tab[key]
            c["n"] += 1
            c["dmg"] += a["dmg"]
            if a["dmg"] == 0:
                c["zero"] += 1
    # 壁が立っている時に貫通したワザの内訳（防御の穴の説明）
    leak = defaultdict(Counter)
    for r in ok:
        for a in r["opp_attacks"]:
            if not a["base"] or a["target"] not in WALLS:
                continue
            if a["dmg"] > 0:
                kind = "mega-ex" if a.get("mega") else ("ex" if a["ex"] else "non-ex")
                leak[(WALLS[a["target"]], kind)][f'{a.get("atk", "?")} ({a["dmg"]})'] += 1
    s["leak_by_attack"] = {f"{k[0]} / {k[1]}": v.most_common(6) for k, v in leak.items()}
    s["bench_damage_per_game"] = round(sum(a["dmg_any"] - a["dmg"] for r in ok for a in r["opp_attacks"]) / n, 1) if n else None

    s["incoming_by_attacker"] = [
        {"our_active": k[0], "attacker_ex": k[1], "attacker_ability": k[2], "attacks": v["n"],
         "zero_damage_share": round(100 * v["zero"] / v["n"], 1), "mean_damage": round(v["dmg"] / v["n"], 1)}
        for k, v in sorted(tab.items(), key=lambda kv: -kv[1]["n"])]

    # 自分のワザの使用回数と与ダメージ
    use, dmg = Counter(), Counter()
    for r in ok:
        use.update(r["our_attacks"])
        dmg.update(r["our_damage"])
    s["our_attacks"] = [{"attack": k, "uses": v, "uses_per_game": round(v / n, 2),
                         "total_damage": dmg[k], "mean_damage": round(dmg[k] / v, 1) if v else None}
                        for k, v in use.most_common()]
    s["plays_per_game"] = {k: round(sum(r["plays"].get(k, 0) for r in ok) / n, 2)
                           for k in COUNT_PLAYS.values() if any(r["plays"].get(k) for r in ok)}
    s["attaches_per_game"] = {k: round(sum(r["attaches"].get(k, 0) for r in ok) / n, 2)
                              for k in COUNT_ATTACH.values() if any(r["attaches"].get(k) for r in ok)}
    s["heal_per_game"] = round(sum(r["heal"] for r in ok) / n, 1) if n else None
    s["spiky_return_per_game"] = round(sum(r["spiky_returns"] for r in ok) / n, 1) if n else None

    # Meganium 在場時の基本草の提供数（2倍化の直接確認）
    dob = [o for r in ok for o in r["doubling_obs"]]
    if dob:
        exact = sum(1 for o in dob if o["provided_g"] == 2 * o["basic_g_cards"])
        s["wild_growth_doubling"] = {"observations": len(dob),
                                     "provided_equals_twice_cards": exact,
                                     "share": round(100 * exact / len(dob), 1)}
    # 打点スケーリング（実測ダメージ vs 提供エネルギー数）
    sc = [x for r in ok for x in r["scaling"]]
    if sc:
        by = defaultdict(list)
        for x in sc:
            by[x["atk"]].append(x)
        s["damage_scaling"] = {}
        for k, xs in by.items():
            fits_cards = fits_provided = 0
            for x in xs:
                per = 30 if k != "Giant Bouquet" else 50
                cnt = x["active_provided_g"] if k == "Giant Bouquet" else x["provided_g"]
                cards_only = x["basic_g_cards"] if k != "Giant Bouquet" else None
                if x["damage"] == x["base"] + per * cnt:
                    fits_provided += 1
                if cards_only is not None and x["damage"] == x["base"] + per * cards_only:
                    fits_cards += 1
            s["damage_scaling"][k] = {"n": len(xs), "fits_provided_energy": fits_provided,
                                      "fits_card_count": fits_cards,
                                      "mean_damage": round(sum(x["damage"] for x in xs) / len(xs), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"replay_behavior_{agent}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with (OUT / f"replay_behavior_{agent}_games.jsonl").open("w", encoding="utf-8") as fh:
        for r in ok:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "opp_attacks"}, ensure_ascii=False) + "\n")
    print(json.dumps(s, ensure_ascii=False, indent=1)[:2600])
    if err:
        print(f"  読めなかったファイル {len(err)} 件: {err[0]['error']}")


if __name__ == "__main__":
    raise SystemExit(main())

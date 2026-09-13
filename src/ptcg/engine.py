"""cabt エンジン (`cg` ライブラリ) のロードとローカル対戦プリミティブ。

`cg` は配布データ内 (data/sample_submission/sample_submission/cg) にあり、
プラットフォーム別の共有ライブラリ (libcg.so / libcg-arm64.so / libcg.dylib / cg.dll)
を `cg.sim` が自動選択する。このモジュールはそれを import path に載せ、
ローカルで対戦を回すための最小 API を提供する。

環境変数 `PTCG_CG_DIR` で cg ディレクトリを明示指定できる。
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

# agent(obs_dict) -> list[int]
Agent = Callable[[dict], list]

_cg_dir: Optional[Path] = None


def find_cg_dir() -> Path:
    """配布データ内の cg ディレクトリを探す。"""
    candidates: list[Path] = []
    env = os.environ.get("PTCG_CG_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO_ROOT / "data/sample_submission/sample_submission/cg")
    candidates += [Path(p) for p in glob.glob(str(REPO_ROOT / "data/**/cg"), recursive=True)]
    for c in candidates:
        if (c / "api.py").exists():
            return c
    raise FileNotFoundError(
        "cg ライブラリが見つかりません。配布データを data/ に展開するか "
        "PTCG_CG_DIR を設定してください。探索先: "
        + ", ".join(str(c) for c in candidates)
    )


def ensure_cg_importable() -> Path:
    """`import cg` できるよう sys.path を整える。cg ディレクトリを返す。"""
    global _cg_dir
    if _cg_dir is None:
        _cg_dir = find_cg_dir()
        parent = str(_cg_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
    return _cg_dir


# 各エージェントが自分のディレクトリから vendor しているモジュール。
# 1プロセスに2体ロードすると sys.modules のキャッシュで 2体目が 1体目のものを掴む。
_VENDORED = ("features", "np_forward", "vocab", "guards", "schema")


def load_agent(main_path: str | Path, *, add_agent_dir: bool = True) -> Agent:
    """Kaggle と同じ流儀で main.py から agent 関数をロードする。

    Kaggle はエージェントを **素の名前空間で exec** し、`__file__` を渡さない。
    ここでも `__file__` を注入しないので、`os.path.dirname(__file__)` 依存の
    コードはこの時点で落ちる（＝ローカルで本番の読み込み事故を再現できる）。

    **1プロセスに複数エージェントを載せる場合の分離**（2026-08-02 修正）:
    従来は ①`PTCG_AGENT_DIR` が単一のグローバルなので 2体目が 1体目の weights.npz を
    読み ②`sys.modules` のキャッシュで 2体目の `import np_forward` が 1体目の
    vendor コピーを返す、という二重の混線があった。結果、ML 対 ML の評価が
    **同一方策どうしの対戦**になっていた（run_eval の H2H が方策差を測れていなかった）。
    ここで agent ディレクトリを sys.path の先頭に置き、vendor モジュールを
    sys.modules から外してから exec することで、各エージェントが自分の資産を掴む。
    main.py は import 時に _TABLES/_POLICY を自分の名前空間へ捕まえるので、
    exec 後に sys.modules がどうなっても混ざらない。
    """
    main_path = Path(main_path)
    d = str(main_path.parent)
    if add_agent_dir:
        # 「無ければ追加」ではなく必ず先頭へ（2体目が1体目のディレクトリを先に引くのを防ぐ）
        while d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
        os.environ["PTCG_AGENT_DIR"] = d
    ensure_cg_importable()
    saved = {k: sys.modules.pop(k, None) for k in _VENDORED}
    try:
        src = main_path.read_text(encoding="utf-8")
        ns: dict = {}
        exec(compile(src, "<agent>", "exec"), ns)  # noqa: S102 — 本番ハーネスの再現
    finally:
        # 次のロードのために掃除する。既に exec 済みのエージェントは自分の名前空間に
        # モジュール参照を持っているので、ここで消しても動作に影響しない。
        for k in _VENDORED:
            sys.modules.pop(k, None)
        for k, m in saved.items():
            if m is not None and k not in sys.modules:
                sys.modules[k] = m
    if "agent" not in ns or not callable(ns["agent"]):
        raise AttributeError(f"{main_path} に呼び出し可能な agent() がありません")
    return ns["agent"]


def check_deck(deck: Sequence[int]) -> tuple[bool, Optional[int]]:
    """デッキ合法性をエンジンで検証する。

    Returns (ok, error_type)。error_type: 1=不正ID, 2=4枚超過, 3=たね不在, 4=ACE SPEC超過。
    """
    ensure_cg_importable()
    from cg.game import battle_finish, battle_start

    _obs, start = battle_start(list(deck), list(deck))
    if start.errorPlayer < 0:
        battle_finish()
        return True, None
    # 不正デッキ時は battle_ptr が None のため battle_finish は呼ばない
    return False, start.errorType


DECK_ERROR_MESSAGES = {
    1: "存在しないカードIDが含まれています",
    2: "同名カードが4枚を超えています（基本エネルギーは例外）",
    3: "たねポケモンが1枚もありません",
    4: "ACE SPEC が2枚以上あります（1枚まで）",
}


def play_game(
    agent0: Agent,
    agent1: Agent,
    deck0: Sequence[int],
    deck1: Sequence[int],
) -> int:
    """agent0(先攻側 index0) と agent1 を1試合対戦させ結果を返す。

    Returns 勝者 player index (0 または 1)、引き分けは 2。
    注意: エンジンはプロセス内グローバル state を使うため **並列対戦不可**。
    エージェントは fail-closed（例外を投げず合法手を返す）前提。
    """
    ensure_cg_importable()
    from cg.game import battle_finish, battle_select, battle_start

    obs, start = battle_start(list(deck0), list(deck1))
    if start.errorPlayer >= 0:
        raise ValueError(
            f"player {start.errorPlayer} のデッキが不正: "
            f"{DECK_ERROR_MESSAGES.get(start.errorType, start.errorType)}"
        )
    try:
        while obs["current"]["result"] < 0:
            who = obs["current"]["yourIndex"]
            move = (agent0 if who == 0 else agent1)(obs)
            obs = battle_select(move)
        return obs["current"]["result"]
    finally:
        battle_finish()


def engine_card_data() -> list:
    """エンジンが持つ全カードデータ（attacks/skills/ex フラグ等の構造化情報）。"""
    ensure_cg_importable()
    from cg.api import all_card_data

    return all_card_data()


def engine_attacks() -> list:
    """エンジンが持つ全ワザデータ。"""
    ensure_cg_importable()
    from cg.api import all_attack

    return all_attack()

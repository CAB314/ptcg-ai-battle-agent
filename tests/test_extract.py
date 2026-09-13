"""BC 抽出（off-by-one・フィルタ）の合成 fixture テスト。エンジン不要。"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ptcg.ml.bc.extract import episode_ok, iter_decisions  # noqa: E402


def _cell(status="INACTIVE", action=None, select=None, turn=1):
    obs = {"current": {"turn": turn, "yourIndex": 0, "players": [{}, {}]}, "select": select}
    return {"status": status, "action": action if action is not None else [], "observation": obs}


def _sel(n_options, k_min=1, k_max=1):
    return {"type": 0, "context": 0, "minCount": k_min, "maxCount": k_max,
            "option": [{"type": 14} for _ in range(n_options)]}


def test_off_by_one_pairing():
    steps = [
        [_cell(), _cell()],
        [_cell(action=[0] * 60), _cell(action=[0] * 60)],           # デッキ提出（select None）
        [_cell("ACTIVE", select=_sel(3), turn=1), _cell()],          # t=2: agent0 が3択
        [_cell(action=[2]), _cell()],                                # t=3: 応答 [2]
        [_cell(), _cell("ACTIVE", select=_sel(2, 0, 1), turn=2)],    # t=4: agent1 pass可
        [_cell(), _cell(action=[])],                                 # t=5: pass
    ]
    ep = {"statuses": ["DONE", "DONE"], "rewards": [1, -1], "steps": steps}
    assert episode_ok(ep)
    got = [(i, payload[1]) for kind, _, i, payload, _ in iter_decisions(ep) if kind == "ok"]
    assert got == [(0, [2]), (1, [])]  # off-by-one 補正 + pass ラベル保持


def test_label_validity_drops():
    steps = [
        [_cell("ACTIVE", select=_sel(2)), _cell()],
        [_cell(action=[5]), _cell()],           # 範囲外
        [_cell("ACTIVE", select=_sel(3, 1, 1)), _cell()],
        [_cell(action=[0, 1]), _cell()],        # 個数違反
        [_cell("ACTIVE", select=_sel(3)), _cell()],
        [_cell(action=[1, 1]), _cell()],        # 重複
    ]
    ep = {"statuses": ["DONE", "DONE"], "rewards": [1, -1], "steps": steps}
    kinds = [(k, r) for k, r, *_ in iter_decisions(ep)]
    assert ("drop", "label_out_of_range") in kinds
    assert ("drop", "label_count") in kinds
    assert kinds.count(("drop", "label_out_of_range")) == 2  # 重複も範囲チェック側で落ちる


def test_episode_filters():
    assert not episode_ok({"statuses": ["DONE", "TIMEOUT"], "rewards": [1, -1]})
    assert not episode_ok({"statuses": ["DONE", "DONE"], "rewards": [1, None]})
    assert episode_ok({"statuses": ["DONE", "DONE"], "rewards": [1, -1]})

"""RL 基盤の単体テスト: numpy↔torch logp パリティ / ガードマスク / GAE。"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ptcg.ml.rl.sample import masked_full_logits, sample_single, _log_softmax  # noqa: E402


def test_numpy_torch_logp_parity():
    torch = pytest.importorskip("torch")
    from ptcg.ml.rl.sample import logp_entropy_torch

    rng = np.random.default_rng(0)
    B, K = 64, 12
    opt = rng.normal(size=(B, K)).astype(np.float32)
    noopl = rng.normal(size=(B,)).astype(np.float32)
    opt_mask = (rng.random((B, K)) > 0.2).astype(np.float32)
    opt_mask[:, 0] = 1.0  # 少なくとも1手は合法
    extra = (rng.random((B, K)) > 0.1).astype(np.float32)
    extra[:, 0] = 1.0
    noop_allow = (rng.random(B) > 0.5).astype(np.float32)

    acts = []
    np_logps = []
    for b in range(B):
        full = masked_full_logits(opt[b] + (1 - opt_mask[b]) * -1e9, extra[b], noopl[b], bool(noop_allow[b]))
        logp_all = _log_softmax(full)
        # 合法手の中から1つ選ぶ（確率最大）
        act = int(np.argmax(logp_all))
        acts.append(act)
        np_logps.append(logp_all[act])
    t_logp, t_ent = logp_entropy_torch(
        torch.from_numpy(opt), torch.from_numpy(noopl),
        torch.from_numpy(opt_mask), torch.from_numpy(extra),
        torch.from_numpy(noop_allow), torch.tensor(acts),
    )
    assert np.max(np.abs(t_logp.numpy() - np.array(np_logps))) < 1e-5
    assert (t_ent >= 0).all()


def test_sample_respects_masks():
    rng = np.random.default_rng(1)
    K = 8
    opt = np.zeros(K, dtype=np.float32)
    extra = np.zeros(K, dtype=np.float32)
    extra[3] = 1.0  # singleton ガード
    for _ in range(50):
        act, logp, ent = sample_single(opt, extra, noop_logit=5.0, noop_allow=False, rng=rng)
        assert act == 3
        assert abs(logp) < 1e-6  # p=1 → logp=0
    # noop 許可時は K を選び得る
    seen_noop = False
    for _ in range(200):
        act, _, _ = sample_single(opt, np.ones(K, np.float32), noop_logit=3.0, noop_allow=True, rng=rng)
        if act == K:
            seen_noop = True
    assert seen_noop


def test_guard_mask_lethal_singleton():
    from ptcg.ml.features import F_O, OF_EFF_DMG, OF_LETHAL
    from ptcg.ml.rl.masks import guard_mask

    K = 5
    of = np.zeros((K, F_O), dtype=np.float32)
    of[2, OF_LETHAL] = 1.0
    of[2, OF_EFF_DMG] = 0.5
    of[4, OF_LETHAL] = 1.0
    of[4, OF_EFF_DMG] = 0.9  # こちらが最大打点
    feats = {"of": of, "noop_allowed": True, "k_min": 1, "k_max": 1}
    mask, noop = guard_mask(feats, sel_type=0)
    assert mask.tolist() == [0, 0, 0, 0, 1]
    assert noop is False
    # リーサル無しなら全許可
    feats2 = {"of": np.zeros((K, F_O), np.float32), "noop_allowed": True, "k_min": 1, "k_max": 1}
    mask2, noop2 = guard_mask(feats2, sel_type=0)
    assert mask2.sum() == K and noop2 is True


def test_gae_terminal_only_reward():
    from ptcg.ml.rl.traj import compute_gae

    # 1席・3決定、終端 +1、V=0 → adv は λ 割引で伝播、returns=adv
    pack = {
        "act": np.zeros(3, np.int16),
        "ep_seat": np.array([7, 7, 7], np.int32),
        "value_pred": np.zeros(3, np.float32),
        "reward": np.array([0, 0, 1.0], np.float32),
        "done": np.array([0, 0, 1], np.int8),
    }
    adv, ret = compute_gae(pack, gamma=1.0, lam=0.95)
    assert abs(adv[2] - 1.0) < 1e-6
    assert abs(adv[1] - 0.95) < 1e-6
    assert abs(adv[0] - 0.95 ** 2) < 1e-6
    assert np.allclose(ret, adv)

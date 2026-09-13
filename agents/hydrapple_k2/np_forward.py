# AUTO-VENDORED from src/ptcg/ml/np_forward.py — edit the source there and re-run scripts/vendor.py
"""weights.npz の numpy 純推論（vendor対象・torch 非依存・~200行）。

model.py の PTCGNet と 1:1 対応（LayerNorm eps=1e-5 / GELU tanh 近似 / pre-LN）。
1決定=1レコード（バッチなし）で forward する。目標: <10ms/手（CPU）。

使い方:
    pol = NpPolicy.load("weights.npz", "config.json")
    out = pol.score(feats)   # feats = features.featurize(...) の出力
    out = (opt_logits [K], noop_logit float, value float) / 失敗時 None
"""

from __future__ import annotations

import json
import math

import numpy as np

_LN_EPS = 1e-5
_G0 = math.sqrt(2.0 / math.pi)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(_G0 * (x + 0.044715 * x * x * x)))


def _ln(x, w, b):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + _LN_EPS) * w + b


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class NpPolicy:
    def __init__(self, weights: dict, config: dict):
        self.w = weights
        self.cfg = config
        self.d = int(config["d_model"])
        self.h = int(config["n_heads"])
        self.n_state_layers = int(config["n_state_layers"])
        self.n_option_layers = int(config["n_option_layers"])

    # ---- ロード（失敗は None を返す。例外は出さない）----
    # score() が参照する重み。1つでも欠けると全推論が None になり**無言でルール縮退**するので、
    # ロード時に落として MODEL_OK=False にする（2026-08-02: v1 の weights を v2 のコードで
    # 読むと card_static が無く、ロードは成功するのに毎回 None を返す事故があった）。
    REQUIRED = ("card_emb.weight", "atk_emb.weight", "card_static", "atk_static",
                "s_proj.weight", "o_proj.weight", "logit_head.weight",
                "noop_head.weight", "value_head.weight")

    @staticmethod
    def load(weights_path, config_path, feature_version=None):
        try:
            cfg = json.loads(open(config_path, encoding="utf-8").read())
            if feature_version is not None and int(cfg.get("feature_version", -1)) != int(feature_version):
                return None
            z = np.load(weights_path)
            weights = {k: z[k].astype(np.float32) for k in z.files}
            missing = [k for k in NpPolicy.REQUIRED if k not in weights]
            if missing:
                return None
            return NpPolicy(weights, cfg)
        except Exception:
            return None

    def _lin(self, name, x):
        return x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]

    def _attn(self, prefix, q_in, kv_in, kv_mask):
        """q_in:[Lq,D] kv_in:[Lk,D] kv_mask:[Lk] (1=有効)。"""
        w = self.w
        qn = _ln(q_in, w[f"{prefix}.ln_q.weight"], w[f"{prefix}.ln_q.bias"])
        kn = _ln(kv_in, w[f"{prefix}.ln_kv.weight"], w[f"{prefix}.ln_kv.bias"])
        q = self._lin(f"{prefix}.wq", qn)
        k = self._lin(f"{prefix}.wk", kn)
        v = self._lin(f"{prefix}.wv", kn)
        Lq, D = q.shape
        Lk = k.shape[0]
        h, dh = self.h, D // self.h
        q = q.reshape(Lq, h, dh).transpose(1, 0, 2)          # [h,Lq,dh]
        k = k.reshape(Lk, h, dh).transpose(1, 0, 2)
        v = v.reshape(Lk, h, dh).transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(dh)     # [h,Lq,Lk]
        if kv_mask is not None:
            scores = scores + (1.0 - kv_mask)[None, None, :] * -1e9
        att = _softmax(scores, axis=-1) @ v                   # [h,Lq,dh]
        out = att.transpose(1, 0, 2).reshape(Lq, D)
        x = q_in + self._lin(f"{prefix}.wo", out)
        fn = _ln(x, w[f"{prefix}.ln_ff.weight"], w[f"{prefix}.ln_ff.bias"])
        f = self._lin(f"{prefix}.ff2", _gelu_tanh(self._lin(f"{prefix}.ff1", fn)))
        return x + f

    def score(self, feats):
        try:
            w = self.w
            sid = np.clip(feats["sid"].astype(np.int64), 0, w["card_emb.weight"].shape[0] - 1)
            sf = feats["sf"].astype(np.float32)
            g = feats["g"].astype(np.float32)
            s = np.concatenate([w["card_emb.weight"][sid], w["card_static"][sid], sf], axis=-1)
            x = self._lin("s_proj", s)
            x[0] = x[0] + self._lin("g_proj", g)
            state_mask = 1.0 - sf[:, 32]  # SF_PAD 列（=1がpad）
            for i in range(self.n_state_layers):
                x = self._attn(f"state_blocks.{i}", x, x, state_mask)

            oid = feats["oid"].astype(np.int64)
            of = feats["of"].astype(np.float32)
            ci = np.clip(oid[:, 0], 0, w["card_emb.weight"].shape[0] - 1)
            ai = np.clip(oid[:, 1], 0, w["atk_emb.weight"].shape[0] - 1)
            o = self._lin("o_proj", np.concatenate([
                w["card_emb.weight"][ci], w["atk_emb.weight"][ai],
                w["card_static"][ci], w["atk_static"][ai], of,
            ], axis=-1))
            for i in range(self.n_option_layers):
                o = self._attn(f"option_blocks.{i}", o, x, state_mask)
            on = _ln(o, w["ln_out.weight"], w["ln_out.bias"])
            opt_logits = (self._lin("logit_head", on)).reshape(-1)
            g_tok = _ln(x[0], w["ln_state_out.weight"], w["ln_state_out.bias"])
            noop_logit = float(self._lin("noop_head", g_tok).reshape(()))
            value = float(np.tanh(self._lin("value_head", g_tok).reshape(())))
            return opt_logits, noop_logit, value
        except Exception:
            return None

    def choose(self, feats, extra_mask=None):
        """スコア → 合法な選択（index リスト）。extra_mask[K]=0 でガードマスク。

        単一選択(k_max<=1)は argmax（noop 許可なら noop と比較）。
        複数選択はスコア降順に k_min..k_max（noop 許可でスコアが全て noop 未満なら空）。
        失敗時 None。
        """
        out = self.score(feats)
        if out is None:
            return None
        opt_logits, noop_logit, _ = out
        k_min = int(feats["k_min"])
        k_max = int(feats["k_max"])
        noop_ok = bool(feats["noop_allowed"])
        logits = opt_logits.copy()
        if extra_mask is not None:
            logits = logits + (1.0 - np.asarray(extra_mask, dtype=np.float32)) * -1e9
        order = np.argsort(-logits)
        if k_max <= 1:
            best = int(order[0])
            if noop_ok and noop_logit > logits[best]:
                return []
            if logits[best] <= -1e8:  # 全部ガードで潰れた
                return None
            return [best]
        picks = [int(i) for i in order if logits[i] > -1e8]
        if noop_ok:
            above = [i for i in picks if logits[i] >= noop_logit]
            k = max(k_min, min(len(above), k_max))
            if k == 0:
                return []
        else:
            k = max(k_min, min(k_max, len(picks)))
        if len(picks) < max(k_min, 1):
            return None
        return picks[:k]

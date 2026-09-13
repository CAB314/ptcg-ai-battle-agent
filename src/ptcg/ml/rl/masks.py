"""ガードマスク（純 numpy）— 学習と配備で同一の「証明可能ガード」をマスクとして表現。

設計原則: ガードは方策の「上書き」ではなく**行動空間のマスク**。actor はマスク後分布から
サンプルし logp もマスク後分布で計算する → 学習と配備が同一方策になり、PPO の on-policy 性が
壊れない。

現行ガードは1つ: **正確打点リーサルの即取り**（配備側 agents/_ml_template/main.py の
guards.lethal_attack と同条件）。featurize が option 導出特徴に実効打点・リーサルflagを
埋めている（features.OF_LETHAL / OF_EFF_DMG、guards と同一式）ので、ここでは feats を
読むだけでよい＝cg 非依存・発火条件のドリフトなし。
"""

from __future__ import annotations

import numpy as np

try:
    from ..features import OF_EFF_DMG, OF_LETHAL
    from ..schema import SEL_ATTACK, SEL_MAIN
except ImportError:  # vendor 文脈
    from features import OF_EFF_DMG, OF_LETHAL  # type: ignore
    from schema import SEL_ATTACK, SEL_MAIN  # type: ignore


def guard_mask(feats: dict, sel_type: int) -> tuple[np.ndarray, bool]:
    """feats（featurize 出力）→ (extra_mask[K] (1=許可), noop_allow)。

    リーサル可能な ATTACK option があれば、その中で実効打点最大の1手だけを許可する
    singleton マスク（noop も禁止）。それ以外は全許可。
    """
    of = feats["of"]
    K = of.shape[0]
    mask = np.ones(K, dtype=np.float32)
    noop_allow = bool(feats["noop_allowed"])
    if sel_type not in (SEL_MAIN, SEL_ATTACK):
        return mask, noop_allow
    if int(feats["k_min"]) > 1 or int(feats["k_max"]) > 1:
        return mask, noop_allow
    lethal = of[:, OF_LETHAL] > 0.5
    if lethal.any():
        eff = np.where(lethal, of[:, OF_EFF_DMG], -1.0)
        idx = int(np.argmax(eff))
        mask = np.zeros(K, dtype=np.float32)
        mask[idx] = 1.0
        noop_allow = False
    return mask, noop_allow

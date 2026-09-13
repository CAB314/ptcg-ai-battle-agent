"""torch(PTCGNet) と numpy(NpPolicy) の logit パリティ検査。

torch 未導入環境と、シャード未生成環境では skip。
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPO = Path(__file__).resolve().parents[1]
SHARD = REPO / "data" / "bc_shards" / "v1" / "2026-07-15" / "shard_0000.npz"

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not SHARD.exists(), reason="シャード未生成")
def test_random_model_parity(tmp_path):
    from ptcg.ml.export import export_policy

    from ptcg.ml.model import ModelConfig, PTCGNet

    model = PTCGNet(ModelConfig())
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": 0}, ckpt)
    rep = export_policy(ckpt, tmp_path / "export", SHARD)
    assert rep["max_diff"] < 1e-4
    assert rep["argmax_match"] == 1.0

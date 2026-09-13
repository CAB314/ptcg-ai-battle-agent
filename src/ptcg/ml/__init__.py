"""ML（BC→PPO）パイプライン。

パッケージ import は torch 非依存（torch を使うのは model / bc.dataset / rl の中だけ、
かつ遅延 import）。schema / features / np_forward は提出エージェントへ vendor される
ため numpy+stdlib のみで書く（scripts/vendor.py 参照）。
"""

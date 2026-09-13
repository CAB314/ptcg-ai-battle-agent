"""ptcg — Pokémon TCG AI Battle 開発・評価用パッケージ（提出物には含めない）。

提出tarに入るのは agents/<name>/ 配下だけ。ここは cg エンジンのロード、
ローカル自己対戦、評価、カードデータ読込などの開発ツールを提供する。
"""

from . import engine  # noqa: F401

__all__ = ["engine"]

"""ローカル開発用のエージェント基底とビルトインエージェント。

提出物 (agents/<name>/main.py) は self-contained で、ここには依存しない。
ここは評価ハーネスがプログラム的にエージェントを組むための土台。
"""

from .base import Agent, RandomAgent, random_move, validate_move

__all__ = ["Agent", "RandomAgent", "random_move", "validate_move"]

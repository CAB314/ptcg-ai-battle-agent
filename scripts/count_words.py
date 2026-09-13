#!/usr/bin/env python
"""Writeup 本文の語数を数える（上限 2,000語）。

`---\n---` で囲まれた区間だけを本文として数える。見出し記号・強調記号・リンク記法は語として数えない。
2種類を報告する:

  * **kaggle**: 空白区切り（運営 topic 738324 の数え方）。Title/Subtitle 行は別欄なので除く。
    2026-09-07 にユーザーが Writeup 画面で実測した 1,879語と本カウンタが一致することを確認した基準。
  * **strict**: 保守的な上限（ハイフン結合語を分割し、数値も1語として数える）。従来の値。

罰則の判定は Kaggle 側のカウンタで行われるため **kaggle を主**とし、strict は上振れの目安として併記する。

使い方: poetry run python scripts/count_words.py docs/report-draft.md
        poetry run python scripts/count_words.py docs/report-draft.md --export docs/submission/writeup_body.md
        （--export: Title/Subtitle 行を除いた本文だけを、Kaggle の Project Description に貼る形で書き出す）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def body_of(text: str) -> str:
    parts = text.split("---\n---")
    return parts[1] if len(parts) >= 3 else (parts[1] if len(parts) == 2 else text)


def count_kaggle(text: str) -> int:
    """Kaggle の数え方（空白区切り）。Title/Subtitle 行は別入力欄なので除外する。"""
    t = body_of(text)
    t = "\n".join(l for l in t.splitlines()
                  if not l.startswith("**Title:**") and not l.startswith("**Subtitle:**"))
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"[*_#>|]", " ", t)
    return len(t.split())


def count(text: str) -> int:
    t = body_of(text)
    t = re.sub(r"`[^`]*`", " ", t)                 # コード
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)  # リンク
    t = re.sub(r"[*_#>|]", " ", t)                  # 装飾
    t = t.replace("—", " ").replace("–", " ").replace("-", " ")
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9']*", t))


def export_body(text: str) -> str:
    """Kaggle に貼る本文（Title/Subtitle 行とその直後の区切り線を除き、前後の空行を落とす）。"""
    lines = body_of(text).splitlines()
    lines = [l for l in lines if not l.startswith("**Title:**") and not l.startswith("**Subtitle:**")]
    while lines and lines[0].strip() in ("", "---"):
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def main() -> int:
    args = sys.argv[1:]
    export = None
    if "--export" in args:
        i = args.index("--export")
        export = Path(args[i + 1]); del args[i:i + 2]
    if not args:
        print("usage: count_words.py <file.md> [--export <out.md>]", file=sys.stderr)
        return 2
    if export is not None:
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(export_body(Path(args[0]).read_text(encoding="utf-8")), encoding="utf-8")
        print(f"書き出し: {export}  ({len(export.read_text(encoding='utf-8').split()):,} 語・空白区切り)")
    for p in args:
        text = Path(p).read_text(encoding="utf-8")
        k, n = count_kaggle(text), count(text)
        limit, target = 2000, 1950
        mark = "OK" if k <= target else ("上限内だが余裕なし" if k <= limit else "★超過")
        print(f"{p}: kaggle {k:,} 語 / {limit:,}（目標 {target:,} 残り {target - k:+,}）"
              f"  strict {n:,}  {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

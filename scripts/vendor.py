#!/usr/bin/env python
"""src の純モジュールを agent ディレクトリへ vendoring（コピー）する。

提出物は self-contained（src/ptcg に依存できない）。相手推定など共有したい
ロジックはここでコピーして同梱する。build_submission が agent 内 *.py を
tar に含めるので、コピーしておけば提出時に一緒に入る。

使い方:
    poetry run python scripts/vendor.py meta_aware

編集は必ず src 側で行い、その後このスクリプトを再実行して同期する
（agent 内のコピーは AUTO-VENDORED ヘッダ付きで上書きされる）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import AGENTS_DIR, REPO_ROOT

# vendored ファイル: agent 内での名前 -> src の実体
VENDOR = {
    "archetypes.py": REPO_ROOT / "src" / "ptcg" / "meta" / "archetypes.py",
}
# ML エージェント用（numpy+stdlib 純。--ml で追加）
VENDOR_ML = {
    "schema.py": REPO_ROOT / "src" / "ptcg" / "ml" / "schema.py",
    "vocab.py": REPO_ROOT / "src" / "ptcg" / "ml" / "vocab.py",
    "features.py": REPO_ROOT / "src" / "ptcg" / "ml" / "features.py",
    "np_forward.py": REPO_ROOT / "src" / "ptcg" / "ml" / "np_forward.py",
}
HEADER = "# AUTO-VENDORED from {src} — edit the source there and re-run scripts/vendor.py\n"


def vendor(agent: str, table: dict | None = None) -> None:
    dst_dir = AGENTS_DIR / agent
    if not dst_dir.exists():
        raise SystemExit(f"agent ディレクトリがありません: {dst_dir}")
    for name, src in (table or VENDOR).items():
        header = HEADER.format(src=src.relative_to(REPO_ROOT))
        (dst_dir / name).write_text(header + src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"vendored {src.relative_to(REPO_ROOT)} -> {(dst_dir / name).relative_to(REPO_ROOT)}")


def check_sync(agent: str, table: dict) -> list[str]:
    """vendorコピーが src と一致しているか（ドリフト検出）。不一致ファイル名を返す。"""
    dst_dir = AGENTS_DIR / agent
    bad = []
    for name, src in table.items():
        dst = dst_dir / name
        expect = HEADER.format(src=src.relative_to(REPO_ROOT)) + src.read_text(encoding="utf-8")
        if not dst.exists() or dst.read_text(encoding="utf-8") != expect:
            bad.append(name)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="コピー先の agent 名")
    ap.add_argument("--ml", action="store_true", help="MLエージェント用（schema/vocab/features/np_forward）")
    args = ap.parse_args()
    vendor(args.agent, VENDOR_ML if args.ml else VENDOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

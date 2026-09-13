#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export KAGGLE_CONFIG_DIR="$SCRIPT_DIR/.kaggle"

if [[ ! -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]]; then
  echo "エラー: $KAGGLE_CONFIG_DIR/kaggle.json がありません（APIトークンを置いてください）。" >&2
  exit 1
fi

COMP_BASENAME="$(basename "$SCRIPT_DIR")"
echo "Downloading files for: $COMP_BASENAME"
mkdir -p "$SCRIPT_DIR/data"

poetry run kaggle competitions download -c "$COMP_BASENAME" -p "$SCRIPT_DIR/data"

ZIP="$SCRIPT_DIR/data/${COMP_BASENAME}.zip"
if [[ -f "$ZIP" ]]; then
  if command -v unzip >/dev/null 2>&1; then
    unzip -o "$ZIP" -d "$SCRIPT_DIR/data"
  else
    poetry run python -m zipfile -e "$ZIP" "$SCRIPT_DIR/data"
  fi
  echo "Downloaded & extracted to: $SCRIPT_DIR/data"
else
  echo "警告: ZIP が見つかりませんでした（規約同意/権限/ネットワークをご確認ください）。" >&2
fi

"""scripts から src/ptcg を import できるようにする共通ブートストラップ。

各スクリプトの冒頭で `import _bootstrap  # noqa` するだけ。
poetry install 済みなら不要だが、未インストールでも動くようにしておく。
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / "agents"
SUBMISSION_DIR = REPO_ROOT / "submission"

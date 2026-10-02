"""テストから src/ 配下のモジュールを import できるよう sys.path に追加する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

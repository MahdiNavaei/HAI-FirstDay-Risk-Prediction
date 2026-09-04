"""Small path/bootstrap helpers shared by the public CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = REPOSITORY_ROOT / "src" / "modeling"
if str(MODEL_SOURCE) not in sys.path:
    sys.path.insert(0, str(MODEL_SOURCE))


def require_data(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV was not found: {path}")
    return path

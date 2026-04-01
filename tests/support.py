from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

TEMP_ROOT = Path("tests") / ".tmp"


@contextmanager
def workspace_temp_dir():
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

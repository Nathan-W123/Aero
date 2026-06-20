"""Smoke test for the interactive 3D viewport pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tests" / "support" / "viewport_smoke_runner.py"


def test_3d_viewport_smoke():
    if not (REPO / "outputs3d" / "sphere_re20_volume.vti").is_file():
        return
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")

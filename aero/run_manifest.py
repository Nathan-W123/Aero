"""Write/read lightweight JSON manifests so the GUI can plot live results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def write_run_manifest(output_dir: str | Path, payload: Dict[str, Any]) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "last_run.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_run_manifest(output_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(output_dir) / "last_run.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())

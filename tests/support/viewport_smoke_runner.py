#!/usr/bin/env python3
"""Smoke test for animated inlet flow arrows in the 3D viewport."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

_mpl = REPO / ".cache" / "matplotlib"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

VTI = REPO / "outputs3d" / "sphere_re20_volume.vti"


def main() -> int:
    if not VTI.is_file():
        print("SKIP: no volume file at", VTI)
        return 0

    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication([])

    from aero.gui.mpl_viewport import MplFlowViewport

    host = QtWidgets.QMainWindow()
    host.resize(960, 720)
    viewport = MplFlowViewport()
    host.setCentralWidget(viewport)
    host.show()
    app.processEvents()

    done = {"ok": False, "error": ""}

    def finish_ok() -> None:
        if viewport._particle_points is None:
            done["error"] = "no particles"
            app.quit()
            return
        start = viewport._particle_points[:, 0].copy()
        QtCore.QTimer.singleShot(600, lambda: _check_motion(start))

    def _check_motion(start_x) -> None:
        if viewport._particle_points is None:
            done["error"] = "particles cleared"
            app.quit()
            return
        moved = float(np.mean(viewport._particle_points[:, 0] - start_x))
        if moved < 0.05:
            done["error"] = f"arrows did not advance (dx={moved:.4f})"
            app.quit()
            return
        if viewport._quiver is None:
            done["error"] = "quiver missing"
            app.quit()
            return
        if not viewport._flow_timer.isActive():
            done["error"] = "flow timer not running"
            app.quit()
            return
        done["ok"] = True
        print(f"OK: arrows advanced mean dx={moved:.3f}")
        app.quit()

    def on_fail(msg: str) -> None:
        done["error"] = msg
        app.quit()

    viewport.load_succeeded.connect(finish_ok)
    viewport.load_failed.connect(on_fail)
    viewport.load_volume_file(VTI, force=True)

    QtCore.QTimer.singleShot(20000, app.quit)
    app.exec()

    if done["error"]:
        print("FAIL:", done["error"])
        return 1
    if not done["ok"]:
        print("FAIL: timed out")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

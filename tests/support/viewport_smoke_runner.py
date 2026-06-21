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
        if not viewport._streamlets:
            done["error"] = "no streamlets"
            app.quit()
            return
        # snapshot head positions for each alive streamlet
        heads_before = [
            s["history"][-1][0] for s in viewport._streamlets if s["alive"]
        ]
        QtCore.QTimer.singleShot(600, lambda: _check_motion(heads_before))

    def _check_motion(heads_before) -> None:
        if not viewport._streamlets:
            done["error"] = "streamlets cleared"
            app.quit()
            return
        heads_after = [
            s["history"][-1][0] for s in viewport._streamlets if s["alive"]
        ]
        n = min(len(heads_before), len(heads_after))
        if n == 0:
            done["error"] = "no alive streamlets after tick"
            app.quit()
            return
        moved = float(np.mean(
            [heads_after[i] - heads_before[i] for i in range(n)]
        ))
        if moved < 0.0:
            done["error"] = f"streamlets did not advance (dx={moved:.4f})"
            app.quit()
            return
        if not viewport._stream_lines or viewport._tip_quiver is None:
            done["error"] = "stream lines or tip quiver missing"
            app.quit()
            return
        if not viewport._flow_timer.isActive():
            done["error"] = "flow timer not running"
            app.quit()
            return
        done["ok"] = True
        print(f"OK: streamlets advanced mean dx={moved:.3f}")
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

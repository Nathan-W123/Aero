"""Separate top-level window for the interactive 3D flow viewport."""

from __future__ import annotations

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    HAS_QT = True
except ImportError:
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    HAS_QT = False

from .viewport import FlowViewport


if HAS_QT:
    class FlowViewerWindow(QtWidgets.QMainWindow):
        """VTK lives here so the main setup window stays responsive."""

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Aero CFD — 3D Flow View")
            self.resize(1100, 780)
            self.viewport = FlowViewport()
            self.setCentralWidget(self.viewport)

        def load_volume(self, path, *, force: bool = False) -> None:
            self.viewport.load_volume_file(path, force=force)

        def changeEvent(self, event: QtCore.QEvent) -> None:
            super().changeEvent(event)
            if event.type() == QtCore.QEvent.Type.WindowStateChange:
                if self.windowState() & QtCore.Qt.WindowState.WindowMinimized:
                    self.viewport.pause_animation()
                else:
                    QtCore.QTimer.singleShot(0, self._restore_from_minimize)

        def _restore_from_minimize(self) -> None:
            if self.isMinimized():
                self.showNormal()
            self.show()
            self.raise_()
            self.activateWindow()

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            self.viewport.pause_animation()
            self.viewport.shutdown()
            event.accept()

else:
    class FlowViewerWindow:  # type: ignore[no-redef]
        pass

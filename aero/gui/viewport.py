"""Interactive 3D flow viewport for the Aero CFD GUI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    HAS_QT = True
except ImportError:
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    HAS_QT = False

from .styles import VIEWPORT_STYLESHEET

_HAS_PYVISTA: Optional[bool] = None
_HAS_QT_PYVISTA: Optional[bool] = None


def _log(message: str) -> None:
    print(f"  [viewport] {message}", flush=True)


def _ensure_pyvista() -> bool:
    global _HAS_PYVISTA
    if _HAS_PYVISTA is None:
        try:
            import pyvista  # noqa: F401
            _HAS_PYVISTA = True
        except ImportError:
            _HAS_PYVISTA = False
    return _HAS_PYVISTA


def _ensure_qt_pyvista():
    global _HAS_QT_PYVISTA
    if _HAS_QT_PYVISTA is None:
        try:
            from pyvistaqt import QtInteractor as _QtInteractor
            _HAS_QT_PYVISTA = True
            return _QtInteractor
        except ImportError:
            _HAS_QT_PYVISTA = False
            return None
    if _HAS_QT_PYVISTA:
        from pyvistaqt import QtInteractor as _QtInteractor
        return _QtInteractor
    return None


def _visualization3d():
    from aero import visualization3d as viz3d

    return viz3d


def _np_random():
    import numpy as np

    return np.random.default_rng(42)


if HAS_QT:
    class FlowViewport(QtWidgets.QWidget):
        load_failed = QtCore.Signal(str)
        load_succeeded = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.setObjectName("FlowViewportRoot")
            self.setStyleSheet(VIEWPORT_STYLESHEET)
            self.setMinimumSize(480, 360)

            self._grid = None
            self._flow_actor = None
            self._plotter = None
            self._plotter_ready = False
            self._loaded_volume_path: Optional[Path] = None
            self._scalar_bar_shown = False
            self._load_pending = False
            self._pending_load_path: Optional[Path] = None
            self._pending_load_force = False
            self._pending_scene: Optional[Dict[str, Any]] = None
            self._load_stage = ""
            self._particle_points = None
            self._rng = None
            self._layout_waits = 0

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            overlay = QtWidgets.QHBoxLayout()
            overlay.setContentsMargins(8, 8, 8, 0)
            self.status_label = QtWidgets.QLabel("Run a 3D case to load the wind tunnel")
            self.status_label.setObjectName("viewportStatus")
            overlay.addWidget(self.status_label, 1)

            self.play_button = self._make_button("Pause Flow", self._toggle_flow)
            self.play_button.setEnabled(False)
            overlay.addWidget(self.play_button)

            self.reset_button = self._make_button("Reset View", self._reset_camera)
            self.reset_button.setEnabled(False)
            overlay.addWidget(self.reset_button)

            self.zoom_in_button = self._make_button("+", self._zoom_in)
            self.zoom_in_button.setEnabled(False)
            self.zoom_in_button.setFixedWidth(32)
            overlay.addWidget(self.zoom_in_button)

            self.zoom_out_button = self._make_button("−", self._zoom_out)
            self.zoom_out_button.setEnabled(False)
            self.zoom_out_button.setFixedWidth(32)
            overlay.addWidget(self.zoom_out_button)

            density_label = QtWidgets.QLabel("Flow density")
            density_label.setObjectName("viewportStatus")
            overlay.addWidget(density_label)
            self.density_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.density_slider.setRange(3, 10)
            self.density_slider.setValue(5)
            self.density_slider.setFixedWidth(110)
            self.density_slider.setEnabled(False)
            self.density_slider.valueChanged.connect(self._respawn_particles)
            overlay.addWidget(self.density_slider)
            layout.addLayout(overlay)

            self._viewer_container = QtWidgets.QWidget()
            self._viewer_container.setMinimumSize(400, 320)
            self._viewer_layout = QtWidgets.QVBoxLayout(self._viewer_container)
            self._viewer_layout.setContentsMargins(0, 0, 0, 0)
            self._placeholder = QtWidgets.QLabel(
                "Interactive 3D wind tunnel\n\n"
                "Drag to rotate · scroll to zoom · shift-drag to pan\n"
                "Animated velocity vectors start automatically after each run"
            )
            self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setWordWrap(True)
            self._placeholder.setStyleSheet("color: #4de8ff; background-color: #000000;")
            self._viewer_layout.addWidget(self._placeholder, 1)
            layout.addWidget(self._viewer_container, 1)

            self._flow_timer = QtCore.QTimer(self)
            self._flow_timer.setInterval(400)
            self._flow_timer.timeout.connect(self._advance_flow)

        @property
        def plotter(self):
            return self._plotter

        def _make_button(self, text: str, handler) -> QtWidgets.QWidget:
            button = QtWidgets.QPushButton(text)
            button.setObjectName("viewportButton")
            button.clicked.connect(handler)
            return button

        def shutdown(self) -> None:
            self._flow_timer.stop()
            self._load_pending = False
            if self._plotter is not None:
                try:
                    self._plotter.close()
                except Exception:
                    pass
                self._plotter = None
                self._plotter_ready = False

        def _particle_count(self) -> int:
            return 30 + int(self.density_slider.value()) * 10

        def _viewer_ready(self) -> bool:
            if not self.isVisible():
                return False
            return self._viewer_container.width() >= 80 and self._viewer_container.height() >= 80

        def _ensure_plotter(self) -> bool:
            if self._plotter_ready and self._plotter is not None:
                return True
            if os.environ.get("AERO_GUI_NO_3D") == "1":
                self._fail_load("3D disabled (AERO_GUI_NO_3D=1).")
                return False
            if not _ensure_pyvista():
                self._fail_load("Install pyvista and pyvistaqt for the 3D viewport.")
                return False
            qt_interactor = _ensure_qt_pyvista()
            if qt_interactor is None:
                self._fail_load("Install pyvistaqt for the interactive 3D viewport.")
                return False
            try:
                _log("creating QtInteractor...")
                self.status_label.setText("Initializing 3D renderer...")
                QtWidgets.QApplication.processEvents()
                plotter = qt_interactor(self._viewer_container, multi_samples=0)
                plotter.set_background("#000000", top="#0a1020")
                plotter.enable_trackball_style()
                self._plotter = plotter
                self._viewer_layout.removeWidget(self._placeholder)
                self._placeholder.hide()
                self._viewer_layout.addWidget(plotter.interactor, 1)
                self._plotter_ready = True
                _log("QtInteractor ready")
                return True
            except Exception as exc:
                _log(f"QtInteractor failed: {exc}")
                self._fail_load(f"3D renderer failed: {exc}")
                return False

        def load_volume_file(self, path: Path, *, force: bool = False) -> None:
            if self._load_pending:
                return
            path = path.resolve()
            if not force and path == self._loaded_volume_path and self._plotter_ready:
                self.load_succeeded.emit()
                return

            self._flow_timer.stop()
            self._pending_load_path = path
            self._pending_load_force = force
            self._load_pending = True
            self._pending_scene = None
            self._load_stage = "plotter"
            self._layout_waits = 0
            self.status_label.setText(f"Preparing {path.name}...")
            _log(f"load requested: {path.name}")
            self.show()
            self.raise_()
            QtWidgets.QApplication.processEvents()
            QtCore.QTimer.singleShot(50, self._advance_load)

        def _advance_load(self) -> None:
            try:
                if self._load_stage == "plotter":
                    if not self._viewer_ready():
                        self._layout_waits += 1
                        if self._layout_waits > 80:
                            self._fail_load("Viewport layout timeout.")
                            return
                        QtCore.QTimer.singleShot(50, self._advance_load)
                        return
                    if not self._ensure_plotter():
                        return
                    self._load_stage = "read"
                    QtCore.QTimer.singleShot(0, self._advance_load)
                    return

                if self._load_stage == "read":
                    path = self._pending_load_path
                    if path is None:
                        self._finish_load()
                        return
                    _log("reading volume...")
                    viz = _visualization3d()
                    grid = viz.load_volume_grid(path)
                    tunnel, solid = viz.build_tunnel_and_solid(grid)
                    self._pending_scene = {
                        "grid": grid,
                        "path": path,
                        "tunnel": tunnel,
                        "solid": solid,
                    }
                    self._load_stage = "draw"
                    QtCore.QTimer.singleShot(0, self._advance_load)
                    return

                if self._load_stage == "draw":
                    self._draw_scene()
                    self._load_stage = "flow"
                    QtCore.QTimer.singleShot(0, self._advance_load)
                    return

                if self._load_stage == "flow":
                    self._start_flow_animation()
                    self._load_stage = "done"
                    QtCore.QTimer.singleShot(0, self._advance_load)
                    return

                if self._load_stage == "done":
                    self._finish_load()
            except Exception as exc:
                _log(f"load failed: {exc}")
                self._fail_load(str(exc))

        def _draw_scene(self) -> None:
            scene = self._pending_scene
            if self._plotter is None or scene is None:
                return
            path = Path(scene["path"]).resolve()
            self._grid = scene["grid"]
            self._loaded_volume_path = path
            self._flow_actor = None
            self._scalar_bar_shown = False

            self._plotter.clear()
            self._plotter.set_background("#000000", top="#0a1020")

            tunnel = scene["tunnel"]
            if tunnel.n_cells > 0:
                self._plotter.add_mesh(
                    tunnel,
                    color="#8899aa",
                    line_width=1.0,
                    opacity=0.45,
                    lighting=False,
                    name="tunnel",
                    render=False,
                )
            solid = scene["solid"]
            if solid.n_cells > 0:
                self._plotter.add_mesh(
                    solid,
                    color="#5ec8e8",
                    opacity=0.88,
                    smooth_shading=True,
                    name="solid",
                    render=False,
                )

            self._plotter.view_isometric()
            self._plotter.reset_camera()
            self._plotter.camera.zoom(1.05)
            _log("rendering geometry...")
            self._plotter.render()
            self.reset_button.setEnabled(True)
            self.zoom_in_button.setEnabled(True)
            self.zoom_out_button.setEnabled(True)
            self.density_slider.setEnabled(True)
            self.play_button.setEnabled(True)
            self.load_succeeded.emit()
            QtWidgets.QApplication.processEvents()

        def _init_particles(self) -> None:
            if self._grid is None:
                return
            viz = _visualization3d()
            self._rng = _np_random()
            self._particle_points = viz.seed_particles(
                self._grid,
                count=self._particle_count(),
                rng=self._rng,
            )
            self._update_flow_vectors()

        def _respawn_particles(self) -> None:
            if self._grid is None:
                return
            was_running = self._flow_timer.isActive()
            self._flow_timer.stop()
            self._init_particles()
            if was_running:
                self._flow_timer.start()

        def _update_flow_vectors(self) -> None:
            if self._plotter is None or self._grid is None or self._particle_points is None:
                return
            viz = _visualization3d()
            vectors, speed, _ = viz.sample_velocity(self._grid, self._particle_points)
            glyphs = viz.build_flow_lines(self._particle_points, vectors, speed)
            if self._flow_actor is not None:
                self._plotter.remove_actor(self._flow_actor, render=False)
                self._flow_actor = None
            if glyphs.n_cells > 0:
                self._flow_actor = self._plotter.add_mesh(
                    glyphs,
                    scalars="speed",
                    cmap="coolwarm",
                    line_width=2.0,
                    opacity=0.95,
                    lighting=False,
                    show_scalar_bar=not self._scalar_bar_shown,
                    scalar_bar_args={
                        "title": "|u|",
                        "color": "#d7f0ff",
                        "n_labels": 4,
                        "vertical": True,
                    },
                    name="flow_vectors",
                    render=False,
                )
                self._scalar_bar_shown = True
            self._plotter.render()

        def _advance_flow(self) -> None:
            if self._plotter is None or self._grid is None or self._particle_points is None:
                return
            viz = _visualization3d()
            self._particle_points, _, _ = viz.advect_particles(
                self._grid,
                self._particle_points,
                dt=1.0,
                rng=self._rng,
            )
            self._update_flow_vectors()

        def _start_flow_animation(self) -> None:
            if self._grid is None:
                return
            _log("starting flow animation...")
            self._init_particles()
            self.play_button.setText("Pause Flow")
            self._flow_timer.start()
            self.status_label.setText("Drag to rotate · scroll to zoom · flow animating")
            _log(f"scene ready: {self._loaded_volume_path.name if self._loaded_volume_path else '?'}")

        def _fail_load(self, message: str) -> None:
            self.status_label.setText(message)
            self.load_failed.emit(message)
            self._finish_load()

        def _finish_load(self) -> None:
            self._load_pending = False
            self._pending_scene = None
            self._load_stage = ""
            self._layout_waits = 0

        def _toggle_flow(self) -> None:
            if self._flow_timer.isActive():
                self._flow_timer.stop()
                self.play_button.setText("Play Flow")
            else:
                self._flow_timer.start()
                self.play_button.setText("Pause Flow")

        def clear_scene(self, message: str) -> None:
            self._flow_timer.stop()
            self._grid = None
            self._loaded_volume_path = None
            self._flow_actor = None
            self._particle_points = None
            self._scalar_bar_shown = False
            self.status_label.setText(message)
            self.reset_button.setEnabled(False)
            self.zoom_in_button.setEnabled(False)
            self.zoom_out_button.setEnabled(False)
            self.density_slider.setEnabled(False)
            self.play_button.setEnabled(False)
            self.play_button.setText("Play Flow")
            if self._plotter is not None:
                self._plotter.clear()
                self._plotter.set_background("#000000", top="#0a1020")
                self._plotter.render()

        def pause_animation(self) -> None:
            self._flow_timer.stop()
            if self.play_button.isEnabled():
                self.play_button.setText("Play Flow")

        def _zoom_in(self) -> None:
            if self._plotter is None:
                return
            self._plotter.camera.zoom(1.3)
            self._plotter.render()

        def _zoom_out(self) -> None:
            if self._plotter is None:
                return
            self._plotter.camera.zoom(1.0 / 1.3)
            self._plotter.render()

        def _reset_camera(self) -> None:
            if self._plotter is None:
                return
            self._plotter.view_isometric()
            self._plotter.reset_camera()
            self._plotter.camera.zoom(1.05)
            self._plotter.render()

else:
    class FlowViewport:  # type: ignore[no-redef]
        load_failed = None
        load_succeeded = None

        def shutdown(self) -> None:
            pass

        def pause_animation(self) -> None:
            pass

        def load_volume_file(self, path, *, force=False) -> None:
            pass

        def clear_scene(self, message: str) -> None:
            pass

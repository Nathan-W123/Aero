"""Matplotlib-based interactive 3D flow viewport (no embedded OpenGL/VTK)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.cm import coolwarm
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from PySide6 import QtCore, QtWidgets
    HAS_QT = True
except ImportError:
    FigureCanvasQTAgg = None  # type: ignore[assignment,misc]
    Figure = None  # type: ignore[assignment]
    QtCore = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    HAS_QT = False

from .styles import VIEWPORT_STYLESHEET

_N_STREAMLETS = 20
_MAX_BODY_LEN = 14       # positions before tail starts following
_STREAMLET_WIDTH = 1.8   # line width
_FLOW_INTERVAL_MS = 40
_TIP_UPDATE_EVERY = 4    # redraw quiver tip only every N frames


def _log(message: str) -> None:
    print(f"  [viewport] {message}", flush=True)


def _visualization3d():
    from aero import visualization3d as viz3d

    return viz3d


def _np_random():
    return np.random.default_rng(42)


def _plot_line_mesh(ax, mesh, *, color: str, alpha: float, linewidth: float) -> None:
    if mesh is None or mesh.n_cells == 0:
        return
    pts = np.asarray(mesh.points, dtype=np.float64)
    raw = np.asarray(mesh.lines, dtype=np.int64)
    idx = 0
    while idx < len(raw):
        count = int(raw[idx])
        if count >= 2:
            seg = raw[idx + 1 : idx + 1 + count]
            xs = pts[seg, 0]
            ys = pts[seg, 1]
            zs = pts[seg, 2]
            ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth)
        idx += count + 1


def _plot_solid_mesh(ax, mesh, *, color: str, alpha: float) -> None:
    if mesh is None or mesh.n_cells == 0:
        return
    pts = np.asarray(mesh.points, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return
    triangles = faces.reshape(-1, 4)[:, 1:4]
    if len(triangles) > 12000:
        step = max(1, len(triangles) // 12000)
        triangles = triangles[::step]
    ax.plot_trisurf(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        triangles=triangles,
        color=color,
        alpha=alpha,
        shade=True,
        linewidth=0.0,
        antialiased=True,
    )


if HAS_QT:
    class MplFlowViewport(QtWidgets.QWidget):
        """Interactive 3D wind tunnel with animated inlet flow arrows."""

        load_failed = QtCore.Signal(str)
        load_succeeded = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.setObjectName("FlowViewportRoot")
            self.setStyleSheet(VIEWPORT_STYLESHEET)
            self.setMinimumSize(480, 360)

            self._grid = None
            self._loaded_volume_path: Optional[Path] = None
            self._rng = None
            self._speed_norm: Optional[Normalize] = None
            self._advect_dt = 6.0
            self._tip_length = 1.0
            self._tip_frame = 0
            self._streamlets: list = []
            self._stream_lines: list = []
            self._tip_quiver = None
            self._default_view = {"elev": 22.0, "azim": -58.0}
            self._load_pending = False
            self._preview_mode = False
            self._view_bounds: Optional[tuple] = None

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
            layout.addLayout(overlay)

            self.figure = Figure(figsize=(7.0, 5.5), dpi=100)
            self.figure.set_facecolor("#000000")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumSize(400, 320)
            self.canvas.setStyleSheet("background-color: #000000;")
            layout.addWidget(self.canvas, 1)
            self.ax = self.figure.add_subplot(111, projection="3d")
            self._draw_placeholder()

            self._flow_timer = QtCore.QTimer(self)
            self._flow_timer.setInterval(_FLOW_INTERVAL_MS)
            self._flow_timer.timeout.connect(self._advance_flow)

        @property
        def plotter(self):
            return None

        def _make_button(self, text: str, handler) -> QtWidgets.QWidget:
            button = QtWidgets.QPushButton(text)
            button.setObjectName("viewportButton")
            button.clicked.connect(handler)
            return button

        def shutdown(self) -> None:
            self._flow_timer.stop()
            self._load_pending = False

        def _draw_placeholder(self) -> None:
            self.figure.clf()
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.ax.set_facecolor("#000000")
            self.ax.set_axis_off()
            self.ax.text2D(
                0.5,
                0.55,
                "Interactive 3D wind tunnel",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="#4de8ff",
                fontsize=13,
                fontweight="bold",
            )
            self.ax.text2D(
                0.5,
                0.38,
                "Drag to rotate · scroll to zoom · right-drag to pan\n"
                "Streamlets grow from the inlet, sweep around the body, then fade out",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="#94a3b8",
                fontsize=10,
            )
            self.figure.tight_layout(pad=0.2)
            self.canvas.draw_idle()

        def load_volume_file(self, path: Path, *, force: bool = False) -> None:
            if self._load_pending:
                return
            path = path.resolve()
            if not path.is_file():
                self._fail_load(f"Volume file not found: {path.name}")
                return
            if not force and path == self._loaded_volume_path and self._grid is not None:
                self.load_succeeded.emit()
                return

            self._preview_mode = False
            self._load_pending = True
            self._flow_timer.stop()
            self.status_label.setText(f"Loading {path.name}...")
            _log(f"load requested: {path.name}")
            QtCore.QTimer.singleShot(0, lambda: self._load_volume(path))

        def _load_volume(self, path: Path) -> None:
            try:
                viz = _visualization3d()
                _log("reading volume...")
                grid = viz.load_volume_grid(path)
                tunnel, solid = viz.build_tunnel_and_solid(grid)
                self._grid = grid
                self._loaded_volume_path = path
                self._rng = _np_random()
                self._advect_dt = viz.compute_flow_advect_dt(grid) * 0.5
                self._configure_speed_colormap(grid)
                _log("drawing 3D scene...")
                self._draw_scene(tunnel, solid)
                self._spawn_streamlets()
                self.play_button.setText("Pause Flow")
                self.play_button.setEnabled(True)
                self.reset_button.setEnabled(True)
                self._flow_timer.start()
                self.status_label.setText(
                    "Streamlets flowing from inlet — drag to rotate the view"
                )
                _log(f"scene ready: {path.name} ({len(self._streamlets)} streamlets)")
                self.load_succeeded.emit()
            except Exception as exc:
                _log(f"load failed: {exc}")
                self._fail_load(str(exc))
            finally:
                self._load_pending = False

        def load_geometry_preview(self, solid: np.ndarray, *, label: str = "") -> None:
            """Show tunnel + obstacle immediately; flow streamlets load after the run."""
            if self._load_pending:
                return
            self._load_pending = True
            self._flow_timer.stop()
            self._clear_stream_artists()
            self._loaded_volume_path = None
            self._preview_mode = True
            self.status_label.setText("Updating geometry preview…")
            QtCore.QTimer.singleShot(0, lambda: self._draw_geometry_preview(solid, label))

        def _draw_geometry_preview(self, solid: np.ndarray, label: str) -> None:
            try:
                viz = _visualization3d()
                grid = viz.build_preview_grid(solid)
                tunnel, solid_mesh = viz.build_tunnel_and_solid(grid)
                self._grid = grid
                self._draw_scene(tunnel, solid_mesh)
                self.play_button.setEnabled(False)
                self.play_button.setText("Flow after run")
                self.reset_button.setEnabled(True)
                cells = int(np.asarray(solid, dtype=bool).sum())
                default = (
                    f"Geometry preview ({cells:,} solid cells) — "
                    "flow arrows load when the run completes"
                )
                self.status_label.setText(label or default)
                _log(f"geometry preview ready ({cells} solid cells — streamlets load after run)")
            except Exception as exc:
                _log(f"geometry preview failed: {exc}")
                self._fail_load(str(exc))
            finally:
                self._load_pending = False

        def _configure_speed_colormap(self, grid) -> None:
            viz = _visualization3d()
            vmin, vmax = viz.fluid_speed_range(grid)
            self._speed_norm = Normalize(vmin=vmin, vmax=vmax)

        def _style_axes(self) -> None:
            self.ax.set_facecolor("#000000")
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor("#334155")
            self.ax.yaxis.pane.set_edgecolor("#334155")
            self.ax.zaxis.pane.set_edgecolor("#334155")
            self.ax.tick_params(colors="#8899aa", labelsize=8)
            self.ax.set_xlabel("x (flow)", color="#8899aa", fontsize=9)
            self.ax.set_ylabel("y", color="#8899aa", fontsize=9)
            self.ax.set_zlabel("z", color="#8899aa", fontsize=9)

        def _draw_scene(self, tunnel, solid) -> None:
            self._clear_stream_artists()
            self.figure.clf()
            self.ax = self.figure.add_subplot(111, projection="3d")
            _plot_line_mesh(self.ax, tunnel, color="#8899aa", alpha=0.55, linewidth=0.6)
            _plot_solid_mesh(self.ax, solid, color="#5ec8e8", alpha=0.88)
            x0, x1, y0, y1, z0, z1 = self._grid.bounds  # type: ignore[union-attr]
            pad = 0.05 * max(x1 - x0, y1 - y0, z1 - z0, 1.0)
            self._view_bounds = (x0 - pad, x1 + pad, y0 - pad, y1 + pad, z0 - pad, z1 + pad)
            self.ax.set_xlim(x0 - pad, x1 + pad)
            self.ax.set_ylim(y0 - pad, y1 + pad)
            self.ax.set_zlim(z0 - pad, z1 + pad)
            self.ax.view_init(elev=self._default_view["elev"], azim=self._default_view["azim"])
            self._style_axes()
            self.figure.tight_layout(pad=0.2)
            self.canvas.draw()

        def _clear_stream_artists(self) -> None:
            self._streamlets = []
            self._stream_lines = []
            if self._tip_quiver is not None:
                try:
                    self._tip_quiver.remove()
                except Exception:
                    pass
                self._tip_quiver = None

        def _spawn_streamlets(self) -> None:
            if self._preview_mode or self._grid is None:
                return
            viz = _visualization3d()
            births = viz.seed_object_biased(self._grid, count=_N_STREAMLETS, rng=self._rng)
            _, speed, _ = viz.sample_velocity(self._grid, births)
            vmin = self._speed_norm.vmin  # type: ignore[union-attr]
            vmax = self._speed_norm.vmax  # type: ignore[union-attr]

            x0, x1 = self._grid.bounds[:2]
            self._tip_length = 0.065 * max(float(x1 - x0), 1.0)

            # Spread initial stages so streamlets aren't all born simultaneously.
            pre_steps = self._rng.integers(0, 22, size=_N_STREAMLETS)  # type: ignore[union-attr]

            self._streamlets = []
            self._stream_lines = []

            for i in range(len(births)):
                birth = births[i].copy()
                t = float(np.clip(
                    (float(speed[i]) - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0
                ))
                color = coolwarm(t)

                history: list = [birth]
                alive = True
                for _ in range(int(pre_steps[i])):
                    new_head, valid = viz.step_particle(
                        self._grid, np.array(history[-1]), self._advect_dt
                    )
                    if valid:
                        history.append(new_head)
                        # update color to local speed at new head
                        _, spd_arr, _ = viz.sample_velocity(
                            self._grid, new_head.reshape(1, 3)
                        )
                        t2 = float(np.clip(
                            (float(spd_arr[0]) - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0
                        ))
                        color = coolwarm(t2)
                    else:
                        alive = False
                        break
                tail = max(0, len(history) - _MAX_BODY_LEN)

                self._streamlets.append({
                    "history": history,
                    "tail": tail,
                    "alive": alive,
                    "color": color,
                })
                line, = self.ax.plot(
                    [], [], [],
                    "-",
                    color=color[:3],
                    linewidth=_STREAMLET_WIDTH,
                    alpha=0.85,
                    solid_capstyle="round",
                )
                self._stream_lines.append(line)

            self._draw_streamlets()

        def _draw_streamlets(self) -> None:
            if not self._streamlets or not self._stream_lines:
                return

            # Update body lines in-place.
            hx, hy, hz, dx, dy, dz, tip_colors = [], [], [], [], [], [], []
            for s, line in zip(self._streamlets, self._stream_lines):
                body = s["history"][s["tail"]:]
                if len(body) >= 2:
                    pts = np.array(body)
                    line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
                    line.set_color(s["color"][:3])
                    line.set_visible(True)
                    # collect arrowhead data
                    direction = pts[-1] - pts[-2]
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-12:
                        direction /= norm
                    hx.append(pts[-1, 0])
                    hy.append(pts[-1, 1])
                    hz.append(pts[-1, 2])
                    dx.append(direction[0])
                    dy.append(direction[1])
                    dz.append(direction[2])
                    tip_colors.append(s["color"][:3])
                else:
                    line.set_visible(False)

            # Redraw batched tip quiver only every N frames (expensive remove+recreate).
            self._tip_frame += 1
            if self._tip_frame >= _TIP_UPDATE_EVERY:
                self._tip_frame = 0
                if self._tip_quiver is not None:
                    try:
                        self._tip_quiver.remove()
                    except Exception:
                        pass
                    self._tip_quiver = None
                if hx:
                    self._tip_quiver = self.ax.quiver(
                        hx, hy, hz, dx, dy, dz,
                        color=tip_colors,
                        length=self._tip_length,
                        arrow_length_ratio=1.0,
                        linewidth=0.0,
                        alpha=0.96,
                    )

            self.canvas.draw_idle()

        def _respawn_streamlet(self, s: dict) -> None:
            if self._grid is None:
                return
            viz = _visualization3d()
            births = viz.seed_object_biased(self._grid, count=1, rng=self._rng)
            birth = births[0].copy()
            _, speed, _ = viz.sample_velocity(self._grid, births)
            vmin = self._speed_norm.vmin  # type: ignore[union-attr]
            vmax = self._speed_norm.vmax  # type: ignore[union-attr]
            t = float(np.clip(
                (float(speed[0]) - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0
            ))
            s["history"] = [birth]
            s["tail"] = 0
            s["alive"] = True
            s["color"] = coolwarm(t)

        def _advance_flow(self) -> None:
            if self._preview_mode or self._grid is None or not self._streamlets:
                return
            viz = _visualization3d()
            vmin = self._speed_norm.vmin  # type: ignore[union-attr]
            vmax = self._speed_norm.vmax  # type: ignore[union-attr]
            for s in self._streamlets:
                if s["alive"]:
                    new_head, valid = viz.step_particle(
                        self._grid, np.array(s["history"][-1]), self._advect_dt
                    )
                    if valid:
                        s["history"].append(new_head)
                        # Recolor to local speed at the head.
                        _, spd_arr, _ = viz.sample_velocity(
                            self._grid, new_head.reshape(1, 3)
                        )
                        t = float(np.clip(
                            (float(spd_arr[0]) - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0
                        ))
                        s["color"] = coolwarm(t)
                        if len(s["history"]) - s["tail"] > _MAX_BODY_LEN:
                            s["tail"] += 1
                    else:
                        s["alive"] = False
                else:
                    s["tail"] += 1

                if s["tail"] >= len(s["history"]):
                    self._respawn_streamlet(s)

            self._draw_streamlets()

        def _fail_load(self, message: str) -> None:
            self.status_label.setText(message)
            self.load_failed.emit(message)

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
            self._clear_stream_artists()
            self._preview_mode = False
            self._view_bounds = None
            self.status_label.setText(message)
            self.reset_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.play_button.setText("Play Flow")
            self._draw_placeholder()

        def pause_animation(self) -> None:
            self._flow_timer.stop()
            if self.play_button.isEnabled():
                self.play_button.setText("Play Flow")

        def _reset_camera(self) -> None:
            if self._view_bounds is None:
                return
            x0, x1, y0, y1, z0, z1 = self._view_bounds
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y0, y1)
            self.ax.set_zlim(z0, z1)
            self.ax.view_init(elev=self._default_view["elev"], azim=self._default_view["azim"])
            self.canvas.draw_idle()

else:
    class MplFlowViewport:  # type: ignore[no-redef]
        load_failed = None
        load_succeeded = None

        def shutdown(self) -> None:
            pass

        def pause_animation(self) -> None:
            pass

        def load_geometry_preview(self, solid, *, label: str = "") -> None:
            pass

        def load_volume_file(self, path, *, force=False) -> None:
            pass

        def clear_scene(self, message: str) -> None:
            pass

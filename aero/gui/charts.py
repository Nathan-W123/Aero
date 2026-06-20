"""Live matplotlib chart widgets for the Aero CFD GUI."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6 import QtCore, QtWidgets
    HAS_QT = True
except ImportError:
    FigureCanvasQTAgg = None  # type: ignore[assignment,misc]
    Figure = None  # type: ignore[assignment,misc]
    QtWidgets = None  # type: ignore[assignment]
    HAS_QT = False

from .styles import CHART_THEME


if HAS_QT:
    class MatplotlibChart(QtWidgets.QFrame):
        def __init__(self, title: str):
            super().__init__()
            self.setObjectName("chartFrame")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)

            self.title_label = QtWidgets.QLabel(title)
            self.title_label.setObjectName("panelTitle")
            layout.addWidget(self.title_label)

            self.figure = Figure(figsize=(4.0, 2.6), dpi=100)
            self.figure.set_facecolor(CHART_THEME["figure_bg"])
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumHeight(200)
            layout.addWidget(self.canvas, 1)
            self.ax = self.figure.add_subplot(111)
            self._style_axes(self.ax)

        def _style_axes(self, ax) -> None:
            ax.set_facecolor(CHART_THEME["axes_bg"])
            ax.tick_params(colors=CHART_THEME["text"], labelsize=8)
            ax.xaxis.label.set_color(CHART_THEME["text"])
            ax.yaxis.label.set_color(CHART_THEME["text"])
            for spine in ax.spines.values():
                spine.set_color(CHART_THEME["grid"])

        def _draw_empty(self, message: str) -> None:
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self._style_axes(self.ax)
            self.ax.text(
                0.5, 0.5, message,
                ha="center", va="center",
                transform=self.ax.transAxes,
                color=CHART_THEME["empty"], fontsize=10,
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.figure.tight_layout(pad=0.8)
            self.canvas.draw_idle()

        def plot_convergence(
            self,
            cd_history: Sequence[float],
            cly_history: Optional[Sequence[float]] = None,
            *,
            step: Optional[int] = None,
            total_steps: Optional[int] = None,
            running: bool = False,
        ) -> None:
            if running and not cd_history:
                self.title_label.setText("Convergence (running…)")
                self._draw_empty("Simulation in progress…")
                return
            if not cd_history:
                self.title_label.setText("Convergence")
                self._draw_empty("Run a case to see convergence.")
                return
            if step is not None and total_steps is not None:
                self.title_label.setText(f"Convergence ({step:,}/{total_steps:,})")
            else:
                self.title_label.setText("Convergence")
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self._style_axes(self.ax)
            steps = np.arange(1, len(cd_history) + 1)
            self.ax.plot(
                steps, cd_history,
                color=CHART_THEME["cd"], linewidth=1.2, label="Cd",
            )
            if cly_history and len(cly_history) == len(cd_history):
                self.ax.plot(
                    steps, cly_history,
                    color=CHART_THEME["cl"], linewidth=1.0, label="Cl",
                )
            self.ax.axhline(
                float(cd_history[-1]),
                color=CHART_THEME["cd"],
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
            )
            self.ax.set_xlabel("Step", fontsize=9)
            self.ax.set_ylabel("Coefficient", fontsize=9)
            self.ax.grid(True, alpha=0.35, color=CHART_THEME["grid"])
            legend = self.ax.legend(loc="best", fontsize=8, framealpha=0.85)
            if legend:
                legend.get_frame().set_facecolor(CHART_THEME["axes_bg"])
                legend.get_frame().set_edgecolor(CHART_THEME["grid"])
                for text in legend.get_texts():
                    text.set_color(CHART_THEME["text"])
            self.figure.tight_layout(pad=0.8)
            self.canvas.draw_idle()

        def plot_re_sweep(
            self,
            re_values: Sequence[float],
            cd_values: Sequence[float],
        ) -> None:
            if not re_values or not cd_values or len(re_values) != len(cd_values):
                self.title_label.setText("Re sweep")
                self._draw_empty("Run Re Sweep to plot Cd vs Re.")
                return
            self.title_label.setText("Cd vs Re")
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self._style_axes(self.ax)
            re_arr = np.asarray(re_values, dtype=np.float64)
            cd_arr = np.asarray(cd_values, dtype=np.float64)
            order = np.argsort(re_arr)
            re_arr = re_arr[order]
            cd_arr = cd_arr[order]
            self.ax.plot(
                re_arr, cd_arr,
                color=CHART_THEME["cd"],
                marker="o",
                linewidth=1.4,
                markersize=6,
            )
            self.ax.set_xlabel("Reynolds number", fontsize=9)
            self.ax.set_ylabel("Cd (mean)", fontsize=9)
            self.ax.grid(True, alpha=0.35, color=CHART_THEME["grid"])
            self.figure.tight_layout(pad=0.8)
            self.canvas.draw_idle()

        def plot_pressure_field(self, pressure: np.ndarray, solid: Optional[np.ndarray] = None) -> None:
            if pressure.size == 0:
                self._draw_empty("Run a 3D case to see pressure.")
                return
            field = np.asarray(pressure, dtype=np.float64)
            if solid is not None:
                mask = np.asarray(solid, dtype=bool)
                if mask.shape == field.shape:
                    field = field.copy()
                    field[mask] = np.nan
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self._style_axes(self.ax)
            im = self.ax.imshow(
                field,
                origin="lower",
                cmap=CHART_THEME["pressure_cmap"],
                aspect="auto",
            )
            self.ax.set_xlabel("x", fontsize=9)
            self.ax.set_ylabel("y", fontsize=9)
            cbar = self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7, colors=CHART_THEME["text"])
            cbar.set_label("p", fontsize=8, color=CHART_THEME["text"])
            self.figure.tight_layout(pad=0.8)
            self.canvas.draw_idle()

        def plot_mesh_voxel_preview(
            self,
            slice_field: np.ndarray,
            *,
            blockage: float,
            solid_cells: int,
        ) -> None:
            """Y-midplane voxel preview before running a mesh case."""
            if slice_field.size == 0:
                self._draw_empty("Select an STL to preview voxelization.")
                return
            self.title_label.setText("Mesh voxel preview")
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self._style_axes(self.ax)
            im = self.ax.imshow(
                slice_field,
                origin="lower",
                cmap="gray_r",
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
            )
            self.ax.set_xlabel("streamwise x", fontsize=9)
            self.ax.set_ylabel("spanwise z", fontsize=9)
            self.ax.set_title(
                f"Blockage {blockage*100:.1f}% · {solid_cells:,} solid cells",
                fontsize=9,
                color=CHART_THEME["text"],
            )
            self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            self.figure.tight_layout(pad=0.8)
            self.canvas.draw_idle()

    class CenterSliceView(QtWidgets.QFrame):
        """Midplane velocity preview in the center panel (no PNG files)."""

        def __init__(self):
            super().__init__()
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self.status = QtWidgets.QLabel("Run a 3D case to preview the flow field")
            self.status.setObjectName("viewportStatus")
            self.status.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(self.status)

            self.figure = Figure(figsize=(6.0, 4.5), dpi=100)
            self.figure.set_facecolor(CHART_THEME["figure_bg"])
            self.canvas = FigureCanvasQTAgg(self.figure)
            layout.addWidget(self.canvas, 1)
            self.ax = self.figure.add_subplot(111)

        def show_velocity_midplane(self, velocity: np.ndarray, solid: Optional[np.ndarray] = None) -> None:
            if velocity.size == 0:
                self.status.setText("No flow data available")
                return
            field = np.asarray(velocity, dtype=np.float64)
            if solid is not None:
                mask = np.asarray(solid, dtype=bool)
                if mask.shape == field.shape:
                    field = field.copy()
                    field[mask] = np.nan
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor(CHART_THEME["axes_bg"])
            im = self.ax.imshow(field, origin="lower", cmap="viridis", aspect="auto")
            cbar = self.figure.colorbar(im, ax=self.ax, fraction=0.035, pad=0.02)
            cbar.set_label("|u|", color=CHART_THEME["cd"])
            cbar.ax.tick_params(colors=CHART_THEME["text"], labelsize=8)
            self.ax.set_title("Midplane velocity", color=CHART_THEME["cd"], fontsize=11)
            self.ax.tick_params(colors=CHART_THEME["text"], labelsize=8)
            self.status.setText("Midplane preview")
            self.figure.tight_layout(pad=0.6)
            self.canvas.draw_idle()

else:
    class MatplotlibChart:  # type: ignore[no-redef]
        pass

    class CenterSliceView:  # type: ignore[no-redef]
        pass

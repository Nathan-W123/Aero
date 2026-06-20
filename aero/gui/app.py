"""PySide6 desktop app for configuring and running Aero CFD cases."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    HAS_QT = True
except ImportError:
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    HAS_QT = False

from .state import (
    GuiConfig,
    GUI_COMBO_FIELDS,
    build_command,
    build_grid_study_commands,
    build_re_sweep_commands,
    parse_progress_line,
    parse_run_results,
    read_checkpoint_step,
    remaining_steps_for_resume,
    search_resumable_checkpoint,
)
from aero.benchmarks import ValidationReport, build_validation_report
from .run_data import chart_data_from_run, latest_volume_file, resolve_output_dir
from .cases import (
    CaseEntry,
    apply_case_config,
    list_saved_cases,
    load_case_results,
    set_case_output_dir,
)
from .charts import MatplotlibChart
from .styles import APP_STYLESHEET
from .mpl_viewport import MplFlowViewport
from .flow_window import FlowViewerWindow
from .geometry_preview import solid_mask_for_3d_config


def _ribbon_group(title: str, widgets: List[QtWidgets.QWidget]) -> QtWidgets.QFrame:
    frame = QtWidgets.QFrame()
    frame.setObjectName("ribbonGroup")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(10, 6, 10, 6)
    layout.setSpacing(4)
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(6)
    for widget in widgets:
        row.addWidget(widget)
    layout.addLayout(row)
    caption = QtWidgets.QLabel(title)
    caption.setObjectName("ribbonGroupTitle")
    caption.setAlignment(QtCore.Qt.AlignHCenter)
    layout.addWidget(caption)
    return frame


if HAS_QT:
    class SimulationProcess(QtCore.QObject):
        output = QtCore.Signal(str)
        finished = QtCore.Signal(int)

        def __init__(self, parent: Optional[QtCore.QObject] = None):
            super().__init__(parent)
            self._process = QtCore.QProcess(self)
            self._process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self._process.readyReadStandardOutput.connect(self._read_output)
            self._process.finished.connect(self._on_finished)
            self._process.errorOccurred.connect(self._on_error)
            self._stdout_buffer = ""

        def start(self, command: list[str], workdir: str) -> None:
            if self._process.state() != QtCore.QProcess.NotRunning:
                return
            self._stdout_buffer = ""
            env = QtCore.QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONUNBUFFERED", "1")
            self._process.setProcessEnvironment(env)
            self._process.setWorkingDirectory(workdir)
            self._process.setProgram(command[0])
            self._process.setArguments(command[1:])
            self._process.start()

        def stop(self) -> None:
            if self._process.state() != QtCore.QProcess.NotRunning:
                self._process.kill()

        def is_running(self) -> bool:
            return self._process.state() != QtCore.QProcess.NotRunning

        def _emit_lines(self, text: str) -> None:
            self._stdout_buffer += text
            while "\n" in self._stdout_buffer:
                line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line:
                    self.output.emit(line)

        def _read_output(self) -> None:
            data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
            if data:
                self._emit_lines(data)

        def _on_error(self, _error: QtCore.QProcess.ProcessError) -> None:
            if self._process.state() == QtCore.QProcess.NotRunning:
                message = self._process.errorString() or "process failed to start"
                self.output.emit(f"[simulation error: {message}]")
                self.finished.emit(-1)

        def _on_finished(self, exit_code: int, _status: QtCore.QProcess.ExitStatus) -> None:
            self._read_output()
            if self._stdout_buffer.strip():
                self.output.emit(self._stdout_buffer.strip())
                self._stdout_buffer = ""
            self.finished.emit(int(exit_code))

    class ResultsSummaryPanel(QtWidgets.QFrame):
        def __init__(self):
            super().__init__()
            self.setObjectName("resultsSummary")
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)

            self.headline = QtWidgets.QLabel("No results yet")
            self.headline.setObjectName("resultsHeadline")
            self.headline.setWordWrap(True)
            layout.addWidget(self.headline)

            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(10)
            self._value_labels: Dict[str, QtWidgets.QLabel] = {}
            rows = [
                ("Drag coefficient", "cd"),
                ("Pressure drag", "cd_p"),
                ("Viscous drag", "cd_v"),
                ("Lift coefficient (Y)", "cl_y"),
                ("Lift coefficient (Z)", "cl_z"),
                ("Runtime", "elapsed"),
                ("Throughput", "throughput"),
            ]
            for row, (label, key) in enumerate(rows):
                name = QtWidgets.QLabel(label)
                name.setObjectName("resultsMetricName")
                value = QtWidgets.QLabel("—")
                value.setObjectName("resultsMetricValue")
                value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                grid.addWidget(name, row, 0)
                grid.addWidget(value, row, 1)
                self._value_labels[key] = value
            layout.addLayout(grid)

            self.validation_title = QtWidgets.QLabel("Physics validation")
            self.validation_title.setObjectName("resultsMetricName")
            layout.addWidget(self.validation_title)

            self.validation_benchmark = QtWidgets.QLabel("Literature: —")
            self.validation_benchmark.setObjectName("validationLine")
            self.validation_benchmark.setWordWrap(True)
            layout.addWidget(self.validation_benchmark)

            self.validation_grid = QtWidgets.QLabel("Grid: —")
            self.validation_grid.setObjectName("validationLine")
            self.validation_grid.setWordWrap(True)
            layout.addWidget(self.validation_grid)

            self.validation_collision = QtWidgets.QLabel("Collision: —")
            self.validation_collision.setObjectName("validationLine")
            self.validation_collision.setWordWrap(True)
            layout.addWidget(self.validation_collision)

            self.validation_bc = QtWidgets.QLabel("BC notes: —")
            self.validation_bc.setObjectName("validationLine")
            self.validation_bc.setWordWrap(True)
            layout.addWidget(self.validation_bc)

        def _style_validation(self, label: QtWidgets.QLabel, status: str, text: str) -> None:
            label.setText(text)
            label.setProperty("validationStatus", status)
            label.style().unpolish(label)
            label.style().polish(label)

        def set_validation(self, report: ValidationReport) -> None:
            prefix = {"pass": "✓", "warn": "!", "fail": "✗", "n/a": "—"}
            self._style_validation(
                self.validation_benchmark,
                report.benchmark_status,
                f"{prefix.get(report.benchmark_status, '—')} Literature: {report.benchmark_message}",
            )
            self._style_validation(
                self.validation_grid,
                report.grid_status,
                f"{prefix.get(report.grid_status, '—')} Grid: {report.grid_message}",
            )
            self._style_validation(
                self.validation_collision,
                report.collision_status,
                f"{prefix.get(report.collision_status, '—')} Collision: {report.collision_message}",
            )
            if report.bc_warnings:
                bc_text = " · ".join(report.bc_warnings)
                self._style_validation(self.validation_bc, "warn", f"! BC: {bc_text}")
            else:
                self._style_validation(self.validation_bc, "pass", "✓ BC: configuration OK")

        def set_results(
            self,
            *,
            mode: str,
            shape: str,
            re: str,
            steps: str,
            metrics: Dict[str, str],
        ) -> None:
            self.headline.setText(f"{mode.upper()} · {shape.title()} · Re = {re} · {steps} steps")
            cd = metrics.get("cd_mean", "—")
            cd_std = metrics.get("cd_std", "")
            cl_y = metrics.get("cl_y_mean", "—")
            cl_y_std = metrics.get("cl_y_std", "")
            cl_z = metrics.get("cl_z_mean", "—")
            cl_z_std = metrics.get("cl_z_std", "")
            elapsed = metrics.get("elapsed", "—")
            steps_per_sec = metrics.get("steps_per_sec", "")

            self._value_labels["cd"].setText(f"{cd} ± {cd_std}" if cd_std else cd)
            self._value_labels["cd_p"].setText(metrics.get("cd_p_mean", "—"))
            self._value_labels["cd_v"].setText(metrics.get("cd_v_mean", "—"))
            self._value_labels["cl_y"].setText(f"{cl_y} ± {cl_y_std}" if cl_y_std else cl_y)
            self._value_labels["cl_z"].setText(f"{cl_z} ± {cl_z_std}" if cl_z_std else cl_z)
            if elapsed != "—":
                self._value_labels["elapsed"].setText(f"{elapsed} s")
            else:
                self._value_labels["elapsed"].setText("—")
            self._value_labels["throughput"].setText(
                f"{steps_per_sec} steps/s" if steps_per_sec else "—"
            )

        def clear_results(self) -> None:
            self.headline.setText("No results yet")
            for label in self._value_labels.values():
                label.setText("—")
            empty = ValidationReport()
            self.set_validation(empty)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, repo_root: str):
            super().__init__()
            self.repo_root = Path(repo_root)
            self.config = GuiConfig()
            self._simulation = SimulationProcess(self)
            self._run_metrics: Dict[str, str] = {}
            self._flow_window: Optional[FlowViewerWindow] = None
            self._live_cd_history: List[float] = []
            self._live_cl_history: List[float] = []
            self._grid_study_active = False
            self._grid_study_commands: List[List[str]] = []
            self._grid_study_index = 0
            self._grid_study_cds: List[float] = []
            self._saved_params_before_grid: Optional[Dict[str, str]] = None
            self._re_sweep_active = False
            self._re_sweep_commands: List[List[str]] = []
            self._re_sweep_res: List[float] = []
            self._re_sweep_index = 0
            self._re_sweep_points: List[tuple[float, float]] = []
            self._resume_checkpoint: Optional[Path] = None
            self._run_total_steps: int = 0
            self._run_started_at: Optional[float] = None

            self.setWindowTitle("Aero CFD Studio")
            self.resize(1680, 980)
            self.setStyleSheet(APP_STYLESHEET)

            self._build_menu()
            self._build_status_bar()
            self._build_ui()

            self._shape_options = {
                "2d": ["cylinder", "rectangle"],
                "3d": ["sphere", "box", "cylinder", "mesh"],
            }
            self._on_mode_changed(self.config.mode, refresh=False)
            self._simulation.output.connect(self._on_sim_output)
            self._simulation.finished.connect(self._on_run_finished)
            QtCore.QTimer.singleShot(400, self._autoload_last_volume)

        def _autoload_last_volume(self) -> None:
            if self.config.mode != "3d":
                return
            volume_path = latest_volume_file(self.config, self.repo_root)
            if volume_path is not None:
                self.viewport.load_volume_file(volume_path)

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            self._simulation.stop()
            self.viewport.shutdown()
            if self._flow_window is not None:
                self._flow_window.close()
            event.accept()

        def changeEvent(self, event: QtCore.QEvent) -> None:
            super().changeEvent(event)
            if event.type() == QtCore.QEvent.Type.WindowStateChange:
                if not (self.windowState() & QtCore.Qt.WindowState.WindowMinimized):
                    self.raise_()
                    self.activateWindow()

        def _restore_window(self) -> None:
            if self.isMinimized():
                self.showNormal()
            self.show()
            self.raise_()
            self.activateWindow()

        def _build_menu(self) -> None:
            file_menu = self.menuBar().addMenu("File")
            file_menu.addAction("Load 3D View", self._load_3d_if_available)
            file_menu.addAction("Open VTK Window", self._open_vtk_window)
            file_menu.addAction("Refresh Results", self._refresh_live_charts)
            file_menu.addAction("Refresh Cases", self._refresh_case_list)
            file_menu.addSeparator()
            file_menu.addAction("Quit", self.close)

        def _build_status_bar(self) -> None:
            self.status_bar = self.statusBar()
            self.summary_label = QtWidgets.QLabel("Ready")
            self.summary_label.setObjectName("statusLabel")
            self.status_bar.addWidget(self.summary_label, 1)

            self.run_progress = QtWidgets.QProgressBar()
            self.run_progress.setObjectName("runProgress")
            self.run_progress.setRange(0, 100)
            self.run_progress.setValue(0)
            self.run_progress.setTextVisible(True)
            self.run_progress.setFormat("Ready")
            self.run_progress.setMinimumWidth(300)
            self.run_progress.setMaximumWidth(480)
            self.run_progress.setFixedHeight(22)
            self.status_bar.addPermanentWidget(self.run_progress)

        def _build_ui(self) -> None:
            root = QtWidgets.QWidget()
            root.setObjectName("centralRoot")
            self.setCentralWidget(root)
            layout = QtWidgets.QVBoxLayout(root)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            layout.addWidget(self._build_ribbon())

            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter.setChildrenCollapsible(False)
            layout.addWidget(splitter, 1)

            splitter.addWidget(self._build_left_panel())
            splitter.addWidget(self._build_center_panel())
            splitter.addWidget(self._build_right_panel())
            splitter.setSizes([320, 900, 360])

        def _build_ribbon(self) -> QtWidgets.QFrame:
            ribbon = QtWidgets.QFrame()
            ribbon.setObjectName("ribbonBar")
            row = QtWidgets.QHBoxLayout(ribbon)
            row.setContentsMargins(8, 4, 8, 4)
            row.setSpacing(0)

            self.run_button = QtWidgets.QPushButton("Run")
            self.run_button.setObjectName("runButton")
            self.run_button.clicked.connect(self._run_case)
            self.stop_button = QtWidgets.QPushButton("Stop")
            self.stop_button.clicked.connect(self._stop_case)
            self.stop_button.setEnabled(False)
            refresh_button = QtWidgets.QPushButton("Refresh")
            refresh_button.clicked.connect(self._refresh_live_charts)
            load3d_button = QtWidgets.QPushButton("Load 3D")
            load3d_button.clicked.connect(self._load_3d_if_available)
            grid_study_button = QtWidgets.QPushButton("Grid Study")
            grid_study_button.clicked.connect(self._run_grid_study)
            re_sweep_button = QtWidgets.QPushButton("Re Sweep")
            re_sweep_button.clicked.connect(self._run_re_sweep)
            self.resume_button = QtWidgets.QPushButton("Resume")
            self.resume_button.clicked.connect(self._resume_case)
            self.resume_button.setEnabled(False)
            row.addWidget(_ribbon_group("Simulation", [
                self.run_button, self.resume_button, self.stop_button,
                refresh_button, load3d_button, grid_study_button, re_sweep_button,
            ]))

            self.mode_box = QtWidgets.QComboBox()
            self.mode_box.addItems(["2d", "3d"])
            self.mode_box.setCurrentText(self.config.mode)
            self.mode_box.currentTextChanged.connect(self._on_mode_changed)
            self.shape_box = QtWidgets.QComboBox()
            self.shape_box.currentTextChanged.connect(self._on_shape_changed)
            row.addWidget(_ribbon_group("Case", [
                self._labeled_widget("Mode", self.mode_box),
                self._labeled_widget("Shape", self.shape_box),
            ]))

            row.addStretch(1)
            return ribbon

        def _labeled_widget(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
            box = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(box)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            caption = QtWidgets.QLabel(label)
            caption.setObjectName("ribbonGroupTitle")
            layout.addWidget(caption)
            layout.addWidget(widget)
            return box

        def _build_left_panel(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QFrame()
            panel.setObjectName("panelFrame")
            layout = QtWidgets.QVBoxLayout(panel)
            layout.setContentsMargins(10, 10, 10, 10)

            title = QtWidgets.QLabel("Simulation Setup")
            title.setObjectName("panelTitle")
            layout.addWidget(title)

            tabs = QtWidgets.QTabWidget()
            self.left_tabs = tabs

            setup_tab = QtWidgets.QWidget()
            setup_tab.setObjectName("paramTab")
            setup_layout = QtWidgets.QVBoxLayout(setup_tab)
            setup_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QtWidgets.QScrollArea()
            scroll.setObjectName("paramScroll")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            form_host = QtWidgets.QWidget()
            form_host.setObjectName("paramFormHost")
            self.form_layout = QtWidgets.QFormLayout(form_host)
            self.form_layout.setSpacing(10)
            self.form_layout.setContentsMargins(12, 12, 12, 12)
            self.form_layout.setLabelAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
            )
            scroll.setWidget(form_host)
            setup_layout.addWidget(scroll)
            tabs.addTab(setup_tab, "Parameters")

            results_tab = QtWidgets.QWidget()
            results_layout = QtWidgets.QVBoxLayout(results_tab)
            self.results_summary = ResultsSummaryPanel()
            results_layout.addWidget(self.results_summary)
            results_layout.addStretch(1)
            tabs.addTab(results_tab, "Results")

            log_tab = QtWidgets.QWidget()
            log_layout = QtWidgets.QVBoxLayout(log_tab)
            self.log_output = QtWidgets.QPlainTextEdit()
            self.log_output.setReadOnly(True)
            log_layout.addWidget(self.log_output)
            tabs.addTab(log_tab, "Run Log")

            cases_tab = QtWidgets.QWidget()
            cases_tab.setObjectName("casesTab")
            cases_layout = QtWidgets.QVBoxLayout(cases_tab)
            cases_layout.setContentsMargins(0, 0, 0, 0)
            cases_hint = QtWidgets.QLabel(
                "Runs are saved to ./cases automatically. Double-click a case to open its results."
            )
            cases_hint.setObjectName("panelSubtitle")
            cases_hint.setWordWrap(True)
            cases_layout.addWidget(cases_hint)

            self.case_list = QtWidgets.QListWidget()
            self.case_list.setObjectName("caseList")
            self.case_list.itemDoubleClicked.connect(lambda _item: self._open_selected_case())
            cases_layout.addWidget(self.case_list, 1)

            case_buttons = QtWidgets.QHBoxLayout()
            refresh_cases_btn = QtWidgets.QPushButton("Refresh")
            refresh_cases_btn.clicked.connect(self._refresh_case_list)
            load_setup_btn = QtWidgets.QPushButton("Load Setup")
            load_setup_btn.clicked.connect(self._load_selected_case_setup)
            open_case_btn = QtWidgets.QPushButton("Open Results")
            open_case_btn.clicked.connect(self._open_selected_case)
            case_buttons.addWidget(refresh_cases_btn)
            case_buttons.addWidget(load_setup_btn)
            case_buttons.addWidget(open_case_btn)
            cases_layout.addLayout(case_buttons)
            tabs.addTab(cases_tab, "Cases")

            self._case_entries: List[CaseEntry] = []
            self._refresh_case_list()

            layout.addWidget(tabs, 1)
            return panel

        def _build_center_panel(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(panel)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(6)

            header = QtWidgets.QHBoxLayout()
            title = QtWidgets.QLabel("Wind Tunnel")
            title.setObjectName("panelTitle")
            subtitle = QtWidgets.QLabel("Interactive 3D wind tunnel with animated flow vectors")
            subtitle.setObjectName("panelSubtitle")
            header.addWidget(title)
            header.addStretch(1)
            header.addWidget(subtitle)
            layout.addLayout(header)

            frame = QtWidgets.QFrame()
            frame.setObjectName("viewportFrame")
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)

            self.viewport = MplFlowViewport()
            self.viewport.load_failed.connect(self._on_3d_load_failed)
            self.viewport.load_succeeded.connect(self._on_3d_load_succeeded)
            frame_layout.addWidget(self.viewport)

            run_status = QtWidgets.QFrame()
            run_status.setObjectName("runStatusFrame")
            run_status_layout = QtWidgets.QVBoxLayout(run_status)
            run_status_layout.setContentsMargins(0, 4, 0, 0)
            run_status_layout.setSpacing(4)

            progress_header = QtWidgets.QLabel("Simulation Progress")
            progress_header.setObjectName("panelTitle")
            run_status_layout.addWidget(progress_header)

            self.run_progress_detail = QtWidgets.QLabel(
                "Run a simulation to see live progress in the status bar below."
            )
            self.run_progress_detail.setObjectName("panelSubtitle")
            self.run_progress_detail.setWordWrap(True)
            run_status_layout.addWidget(self.run_progress_detail)

            live_log_label = QtWidgets.QLabel("Live Run Log")
            live_log_label.setObjectName("panelSubtitle")
            run_status_layout.addWidget(live_log_label)

            self.live_log_output = QtWidgets.QPlainTextEdit()
            self.live_log_output.setObjectName("liveRunLog")
            self.live_log_output.setReadOnly(True)
            self.live_log_output.setMaximumBlockCount(500)
            self.live_log_output.setMinimumHeight(120)
            run_status_layout.addWidget(self.live_log_output, 1)

            self._center_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self._center_splitter.setObjectName("centerSplitter")
            self._center_splitter.setChildrenCollapsible(False)
            self._center_splitter.addWidget(frame)
            self._center_splitter.addWidget(run_status)
            self._center_splitter.setStretchFactor(0, 3)
            self._center_splitter.setStretchFactor(1, 1)
            run_status.setMinimumHeight(180)
            self._center_splitter.setSizes([520, 200])

            layout.addWidget(self._center_splitter, 1)
            return panel

        def _append_log_line(self, line: str) -> None:
            self.log_output.appendPlainText(line)
            self.live_log_output.appendPlainText(line)
            for editor in (self.log_output, self.live_log_output):
                scrollbar = editor.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())

        def _reset_run_progress(self, *, total_steps: int, label: str) -> None:
            self._run_total_steps = max(int(total_steps), 1)
            self._run_started_at = time.perf_counter()
            self.run_progress.setRange(0, 0)
            self.run_progress.setFormat("Starting…")
            self.run_progress_detail.setText(
                f"{label}  ·  progress bar is in the bottom status bar"
            )
            self.live_log_output.clear()
            if hasattr(self, "_center_splitter"):
                sizes = self._center_splitter.sizes()
                total = max(sum(sizes), 1)
                self._center_splitter.setSizes([int(total * 0.62), int(total * 0.38)])

        def _update_run_progress(self, progress) -> None:
            pct = int(100 * progress.step / max(progress.total_steps, 1))
            self.run_progress.setRange(0, 100)
            self.run_progress.setValue(min(max(pct, 0), 100))
            self.run_progress.setFormat(
                f"Step {progress.step:,} / {progress.total_steps:,} ({pct}%)"
            )
            eta_text = ""
            if progress.steps_per_sec > 0:
                remaining = max(progress.total_steps - progress.step, 0)
                eta_sec = remaining / progress.steps_per_sec
                if eta_sec >= 3600:
                    eta_text = f" · ETA ~{eta_sec / 3600:.1f} h"
                elif eta_sec >= 60:
                    eta_text = f" · ETA ~{eta_sec / 60:.0f} min"
                else:
                    eta_text = f" · ETA ~{eta_sec:.0f} s"
            self.run_progress_detail.setText(
                f"{progress.steps_per_sec:,} steps/s · "
                f"Cd={progress.cd:+.4f} · Cl={progress.cl:+.4f}{eta_text}"
            )

        def _finish_run_progress(self, *, success: bool, message: str) -> None:
            if success:
                self.run_progress.setRange(0, 100)
                self.run_progress.setValue(100)
                self.run_progress.setFormat("Complete")
            else:
                self.run_progress.setRange(0, 100)
                self.run_progress.setValue(0)
                self.run_progress.setFormat("Failed")
            self.run_progress_detail.setText(message)

        def _on_3d_load_succeeded(self) -> None:
            self.summary_label.setText("Interactive 3D wind tunnel — drag to rotate")

        def _on_3d_load_failed(self, message: str) -> None:
            self.summary_label.setText(f"3D load failed — {message}")

        def _open_vtk_window(self) -> None:
            if self.config.mode != "3d":
                self.summary_label.setText("Switch to 3D mode for the VTK flow window.")
                return
            volume_path = latest_volume_file(self.config, self.repo_root)
            if volume_path is None:
                self.summary_label.setText("No 3D volume yet — run a 3D case first.")
                return
            if self._flow_window is None:
                self._flow_window = FlowViewerWindow()
            self._flow_window.show()
            self._flow_window.raise_()
            self._flow_window.load_volume(volume_path, force=True)
            self.summary_label.setText("VTK flow window opened (may require Terminal.app).")

        def _build_right_panel(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QFrame()
            panel.setObjectName("panelFrame")
            layout = QtWidgets.QVBoxLayout(panel)
            layout.setContentsMargins(10, 10, 10, 10)

            title = QtWidgets.QLabel("Diagnostics")
            title.setObjectName("panelTitle")
            layout.addWidget(title)

            self.convergence_chart = MatplotlibChart("Convergence")
            self.re_sweep_chart = MatplotlibChart("Cd vs Re")
            self.pressure_chart = MatplotlibChart("Pressure Field")
            layout.addWidget(self.convergence_chart, 1)
            layout.addWidget(self.re_sweep_chart, 1)
            layout.addWidget(self.pressure_chart, 1)
            return panel

        def _current_params(self) -> Dict[str, str]:
            return self.config.params_2d if self.config.mode == "2d" else self.config.params_3d

        def _on_mode_changed(self, mode: str, *, refresh: bool = True) -> None:
            self.config.mode = mode
            self.shape_box.blockSignals(True)
            self.shape_box.clear()
            self.shape_box.addItems(self._shape_options[mode])
            current_shape = self.config.shape_2d if mode == "2d" else self.config.shape_3d
            self.shape_box.setCurrentText(current_shape)
            self.shape_box.blockSignals(False)
            self._rebuild_form(refresh=refresh)

        def _on_shape_changed(self, shape: str) -> None:
            if self.config.mode == "2d":
                self.config.shape_2d = shape
            else:
                self.config.shape_3d = shape
            self._rebuild_form(refresh=True, load_3d=False)

        def _rebuild_form(self, *, refresh: bool = True, load_3d: bool = False) -> None:
            while self.form_layout.rowCount():
                self.form_layout.removeRow(0)
            self._field_widgets: Dict[str, QtWidgets.QWidget] = {}

            params = self._current_params()
            common = ["re", "steps", "u0", "backend", "collision", "trt_lambda", "sponge_cells", "sponge_strength", "les", "les_cs"]
            if self.config.mode == "2d":
                fields = common + ["nx", "ny", "wall_bc", "inlet_bc", "outlet_bc", "inlet_perturbation"]
                shape = self.config.shape_2d
                if shape == "cylinder":
                    fields += ["radius"]
                elif shape == "rectangle":
                    fields += ["width", "height"]
            else:
                fields = common + ["nx", "ny", "nz", "wall_bc", "outlet_bc", "viz3d", "inlet_perturbation"]
                shape = self.config.shape_3d
                if shape == "sphere":
                    fields += ["radius"]
                elif shape == "box":
                    fields += ["width", "height", "depth"]
                elif shape == "cylinder":
                    fields += ["radius", "length"]
                elif shape == "mesh":
                    fields += ["stl_path", "stl_fit", "mesh_bc", "mesh_orient", "mesh_rot_x", "mesh_rot_y", "mesh_rot_z"]

            for field in fields:
                label = QtWidgets.QLabel(field)
                label.setObjectName("paramLabel")
                if field in GUI_COMBO_FIELDS:
                    widget: QtWidgets.QWidget = QtWidgets.QComboBox()
                    widget.setObjectName("paramCombo")
                    widget.addItems(GUI_COMBO_FIELDS[field])
                    value = params.get(field, GUI_COMBO_FIELDS[field][0])
                    idx = widget.findText(value)
                    widget.setCurrentIndex(idx if idx >= 0 else 0)
                    widget.currentTextChanged.connect(self._update_state_from_form)
                else:
                    widget = QtWidgets.QLineEdit(params.get(field, ""))
                    widget.setObjectName("paramField")
                    widget.textChanged.connect(self._update_state_from_form)
                self.form_layout.addRow(label, widget)
                self._field_widgets[field] = widget

            if self.config.mode == "3d" and self.config.shape_3d == "mesh":
                browse = QtWidgets.QPushButton("Browse STL…")
                browse.clicked.connect(self._browse_stl_file)
                self.form_layout.addRow("", browse)
                preview_btn = QtWidgets.QPushButton("Preview mesh voxels")
                preview_btn.clicked.connect(self._refresh_mesh_preview)
                self.form_layout.addRow("", preview_btn)
                self._mesh_preview_label = QtWidgets.QLabel("")
                self._mesh_preview_label.setObjectName("panelSubtitle")
                self._mesh_preview_label.setWordWrap(True)
                self.form_layout.addRow("", self._mesh_preview_label)
                rot_hint = QtWidgets.QLabel(
                    "Rotation (deg): mesh_rot_x = roll (flow), mesh_rot_y = yaw, mesh_rot_z = pitch. "
                    "Applied after mesh_orient."
                )
                rot_hint.setObjectName("panelSubtitle")
                rot_hint.setWordWrap(True)
                self.form_layout.addRow("", rot_hint)
            else:
                self._mesh_preview_label = None

            self._update_state_from_form()
            if refresh:
                self._refresh_live_charts()

        def _field_value(self, field: str) -> str:
            widget = self._field_widgets.get(field)
            if widget is None:
                params = self._current_params()
                return params.get(field, "")
            if isinstance(widget, QtWidgets.QComboBox):
                return widget.currentText()
            return widget.text().strip()

        def _browse_stl_file(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select STL mesh",
                str(self.repo_root),
                "STL files (*.stl *.STL);;All files (*)",
            )
            if not path:
                return
            params = self._current_params()
            params["stl_path"] = path
            widget = self._field_widgets.get("stl_path")
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(path)
            self._apply_mesh_presets(params)
            self._update_state_from_form()
            self._refresh_mesh_preview()

        def _apply_mesh_presets(self, params: dict) -> None:
            """Sensible defaults when loading an STL for the first time."""
            if params.get("sponge_cells", "0") in ("0", ""):
                params["sponge_cells"] = "8"
            if float(params.get("re", "100") or "100") > 150 and params.get("collision") == "bgk":
                params["collision"] = "trt"
            if float(params.get("re", "100") or "100") > 300 and params.get("les") == "0":
                params["les"] = "1"

        def _refresh_mesh_preview(self) -> None:
            if self.config.mode != "3d" or self.config.shape_3d != "mesh":
                return
            params = self._current_params()
            stl = params.get("stl_path", "")
            if not stl:
                if self._mesh_preview_label is not None:
                    self._mesh_preview_label.setText("")
                self.pressure_chart.plot_mesh_voxel_preview(
                    np.array([]), blockage=0.0, solid_cells=0,
                )
                return
            try:
                from aero.geometry3d.mesh_preview import build_mesh_preview

                preview = build_mesh_preview(
                    stl,
                    nz=int(params.get("nz", "64")),
                    ny=int(params.get("ny", "64")),
                    nx=int(params.get("nx", "128")),
                    fit_frac=float(params.get("stl_fit", "0.35") or "0.35"),
                    mesh_orient=params.get("mesh_orient", "auto"),
                    mesh_rot_x=float(params.get("mesh_rot_x", "0") or "0"),
                    mesh_rot_y=float(params.get("mesh_rot_y", "0") or "0"),
                    mesh_rot_z=float(params.get("mesh_rot_z", "0") or "0"),
                )
            except (OSError, ValueError) as exc:
                if self._mesh_preview_label is not None:
                    self._mesh_preview_label.setText(f"Preview failed: {exc}")
                return
            if preview is None:
                if self._mesh_preview_label is not None:
                    self._mesh_preview_label.setText("STL file not found.")
                return
            msg = (
                f"Voxel preview: {preview.solid_cells:,} cells · "
                f"blockage {preview.blockage*100:.1f}% "
                f"(y={preview.blockage_y*100:.1f}%, z={preview.blockage_z*100:.1f}%) · "
                f"D≈{preview.reference_length:.1f}"
            )
            if self._mesh_preview_label is not None:
                self._mesh_preview_label.setText(msg)
            self.pressure_chart.plot_mesh_voxel_preview(
                preview.midplane,
                blockage=preview.blockage,
                solid_cells=preview.solid_cells,
            )
            self._update_validation_panel()

        def _refresh_checkpoint_status(self) -> None:
            checkpoint = search_resumable_checkpoint(self.config, self.repo_root)
            self._resume_checkpoint = checkpoint
            if checkpoint is None:
                self.resume_button.setEnabled(False)
                self.resume_button.setToolTip("")
                return
            try:
                step = read_checkpoint_step(checkpoint)
                target = int(self._current_params().get("steps", "0"))
                self.resume_button.setEnabled(True)
                self.resume_button.setToolTip(
                    f"Resume from step {step:,} ({checkpoint.name}) — {max(target - step, 0):,} steps remaining"
                )
            except (OSError, KeyError, ValueError):
                self.resume_button.setEnabled(True)
                self.resume_button.setToolTip(str(checkpoint))

        def _refresh_live_charts(self) -> None:
            manifest, fields = chart_data_from_run(self.config, self.repo_root)
            if manifest:
                cly = manifest.get("Cly_history") or manifest.get("Cl_history")
                self.convergence_chart.plot_convergence(
                    manifest.get("Cd_history", []),
                    cly,
                )
            else:
                self.convergence_chart.plot_convergence([])

            if self._re_sweep_points:
                re_vals, cd_vals = zip(*self._re_sweep_points)
                self.re_sweep_chart.plot_re_sweep(re_vals, cd_vals)
            else:
                self.re_sweep_chart.plot_re_sweep([], [])

            if fields is not None:
                self.pressure_chart.plot_pressure_field(fields["pressure"], fields["solid"])
            else:
                self.pressure_chart.plot_pressure_field(np.array([]))

        def _refresh_case_list(self) -> None:
            self._case_entries = list_saved_cases(self.repo_root, self.config.cases_root)
            self.case_list.clear()
            for entry in self._case_entries:
                item = QtWidgets.QListWidgetItem(entry.label)
                item.setToolTip(str(entry.path))
                self.case_list.addItem(item)
            if self._case_entries:
                self.case_list.setCurrentRow(0)

        def _selected_case(self) -> Optional[CaseEntry]:
            row = self.case_list.currentRow()
            if row < 0 or row >= len(self._case_entries):
                return None
            return self._case_entries[row]

        def _load_case_config(self, entry: CaseEntry) -> None:
            import json

            config_path = entry.path / "config.json"
            case_config = json.loads(config_path.read_text())
            apply_case_config(self.config, case_config)

        def _load_selected_case_setup(self) -> None:
            entry = self._selected_case()
            if entry is None:
                self.summary_label.setText("Select a saved case first.")
                return
            self._load_case_config(entry)
            self._on_mode_changed(self.config.mode, refresh=False)
            self._rebuild_form(refresh=False)
            self.left_tabs.setCurrentIndex(0)
            self.summary_label.setText(f"Loaded setup from {entry.name}")

        def _open_selected_case(self) -> None:
            entry = self._selected_case()
            if entry is None:
                self.summary_label.setText("Select a saved case first.")
                return
            self._load_case_config(entry)
            set_case_output_dir(self.config, entry.path, self.repo_root)
            self._on_mode_changed(self.config.mode, refresh=False)
            self._rebuild_form(refresh=False)
            self._show_case_results(entry)
            self._refresh_live_charts()
            if entry.mode == "3d":
                volume_path = latest_volume_file(self.config, self.repo_root)
                if volume_path is not None:
                    self.viewport.load_volume_file(volume_path, force=True)
            else:
                self.viewport.clear_scene("Switch to 3D mode or run a 3D case for the wind tunnel.")
            self.left_tabs.setCurrentIndex(1)
            self.summary_label.setText(f"Opened case {entry.name}")

        def _show_case_results(self, entry: CaseEntry) -> None:
            metrics = load_case_results(entry.path)
            params = self._current_params()
            self.results_summary.set_results(
                mode=entry.mode,
                shape=entry.shape,
                re=params.get("re", "?"),
                steps=str(entry.steps),
                metrics=metrics,
            )
            cd_val = None
            try:
                cd_val = float(metrics.get("cd_mean", ""))
            except (TypeError, ValueError):
                cd_val = None
            self._update_validation_panel(cd=cd_val)

        def _adopt_newest_case(self) -> None:
            self._refresh_case_list()
            if not self._case_entries:
                return
            newest = self._case_entries[0]
            set_case_output_dir(self.config, newest.path, self.repo_root)
            self.case_list.setCurrentRow(0)
            self.summary_label.setText(f"Saved case → {newest.name}")

        def _begin_3d_after_run(self) -> None:
            """Load interactive 3D viewport after a successful 3D simulation."""
            volume_path = latest_volume_file(self.config, self.repo_root)
            if volume_path is None:
                self.viewport.clear_scene("No 3D volume file found after run.")
                return
            self.summary_label.setText(f"Loading 3D wind tunnel from {volume_path.name}...")
            self.viewport.load_volume_file(volume_path, force=True)

        def _load_3d_if_available(self) -> None:
            if self.config.mode != "3d":
                self.summary_label.setText("Switch to 3D mode for the wind tunnel.")
                return
            volume_path = latest_volume_file(self.config, self.repo_root)
            if volume_path is None:
                self.summary_label.setText("No 3D volume yet — run a 3D case first.")
                return
            self.viewport.load_volume_file(volume_path, force=True)

        def _update_state_from_form(self) -> None:
            params = self._current_params()
            for field in self._field_widgets:
                params[field] = self._field_value(field)
            self.summary_label.setText(
                f"{self.config.mode.upper()} · {self.shape_box.currentText()} · "
                f"Re={params.get('re', '?')} · steps={params.get('steps', '?')}"
            )
            self._refresh_checkpoint_status()

        def _update_validation_panel(
            self,
            *,
            cd: Optional[float] = None,
            grid_cd_values: Optional[List[float]] = None,
        ) -> None:
            params = self._current_params()
            shape = self.shape_box.currentText()
            report = build_validation_report(
                mode=self.config.mode,
                shape=shape,
                params=params,
                cd=cd,
                grid_cd_values=grid_cd_values,
            )
            self.results_summary.set_validation(report)

        def _start_command(self, command: List[str], *, label: str, total_steps: Optional[int] = None) -> None:
            self.log_output.clear()
            self._live_cd_history = []
            self._live_cl_history = []
            self.convergence_chart.plot_convergence([], running=True)
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.summary_label.setText(label)
            if total_steps is None:
                params = self._current_params()
                total_steps = int(params.get("steps", "500"))
            self._reset_run_progress(total_steps=total_steps, label=label)
            if self.config.mode == "3d":
                self._show_run_geometry_preview()
            else:
                self.viewport.pause_animation()
            self._simulation.start(command, str(self.repo_root))

        def _show_run_geometry_preview(self) -> None:
            solid = solid_mask_for_3d_config(self.config)
            if solid is None:
                self.viewport.pause_animation()
                return
            shape = self.config.shape_3d
            cells = int(np.asarray(solid, dtype=bool).sum())
            self.viewport.load_geometry_preview(
                solid,
                label=(
                    f"Running {shape} ({cells:,} solid cells) — "
                    "flow arrows load when the run completes"
                ),
            )

        def _run_case(self) -> None:
            if self._simulation.is_running():
                return
            self._update_state_from_form()
            if self.config.mode == "3d" and self.config.shape_3d == "mesh":
                stl = Path(self._current_params().get("stl_path", ""))
                if not stl.is_file():
                    self.summary_label.setText("Select a valid STL file for mesh geometry.")
                    return
            self._grid_study_active = False
            self._re_sweep_active = False
            self._run_metrics = {}
            self.results_summary.clear_results()
            self._update_validation_panel()
            self._start_command(build_command(self.config), label="Simulation running...")

        def _resume_case(self) -> None:
            if self._simulation.is_running():
                return
            self._update_state_from_form()
            checkpoint = self._resume_checkpoint or search_resumable_checkpoint(
                self.config, self.repo_root
            )
            if checkpoint is None:
                self.summary_label.setText("No checkpoint found to resume.")
                return
            params = self._current_params()
            target_steps = int(params.get("steps", "5000"))
            remaining = remaining_steps_for_resume(checkpoint, target_steps)
            step = read_checkpoint_step(checkpoint)
            self._grid_study_active = False
            self._re_sweep_active = False
            command = build_command(
                self.config,
                resume_from=str(checkpoint),
                steps_override=remaining,
                enable_checkpoints=True,
            )
            self._start_command(
                command,
                label=f"Resuming from step {step:,} ({remaining:,} steps remaining)…",
                total_steps=remaining,
            )

        def _run_grid_study(self) -> None:
            if self._simulation.is_running():
                return
            self._update_state_from_form()
            self._saved_params_before_grid = dict(self._current_params())
            self._grid_study_commands = build_grid_study_commands(self.config)
            self._grid_study_index = 0
            self._grid_study_cds = []
            self._grid_study_active = True
            self._re_sweep_active = False
            self.results_summary.clear_results()
            self._start_command(
                self._grid_study_commands[0],
                label="Grid study: running level 1/3…",
            )

        def _run_re_sweep(self) -> None:
            if self._simulation.is_running():
                return
            self._update_state_from_form()
            self._re_sweep_commands, self._re_sweep_res = build_re_sweep_commands(self.config)
            self._re_sweep_index = 0
            self._re_sweep_points = []
            self._re_sweep_active = True
            self._grid_study_active = False
            self.results_summary.clear_results()
            self.re_sweep_chart.plot_re_sweep([], [])
            total = len(self._re_sweep_commands)
            self._start_command(
                self._re_sweep_commands[0],
                label=f"Re sweep: run 1/{total}…",
            )

        def _on_sim_output(self, line: str) -> None:
            self._append_log_line(line)
            if "Running 3D simulation" in line or "Running simulation" in line:
                self.run_progress.setRange(0, 0)
                self.run_progress.setFormat("Solver running…")
            if line.startswith("[X]") or line.startswith("[simulation error"):
                self._finish_run_progress(
                    success=False,
                    message=line.removeprefix("[X] ").strip() or "Simulation failed.",
                )
            progress = parse_progress_line(line)
            if progress is None:
                return
            self._live_cd_history.append(progress.cd)
            self._live_cl_history.append(progress.cl)
            self.convergence_chart.plot_convergence(
                self._live_cd_history,
                self._live_cl_history,
                step=progress.step,
                total_steps=progress.total_steps,
            )
            self._update_run_progress(progress)
            pct = int(100 * progress.step / max(progress.total_steps, 1))
            self.summary_label.setText(
                f"Running step {progress.step:,}/{progress.total_steps:,} "
                f"({pct}%) · {progress.steps_per_sec:,} steps/s · "
                f"Cd={progress.cd:+.4f}"
            )

        def _stop_case(self) -> None:
            self._simulation.stop()
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.summary_label.setText("Simulation stopped.")
            self._finish_run_progress(success=False, message="Simulation stopped by user.")
            self._append_log_line("\n[simulation stopped by user]")

        def _on_run_finished(self, returncode: int) -> None:
            self._append_log_line(f"\n[process exited with code {returncode}]")
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            self._run_metrics = parse_run_results(self.log_output.toPlainText())
            params = self._current_params()
            cd_val = None
            try:
                cd_val = float(self._run_metrics.get("cd_mean", ""))
            except (TypeError, ValueError):
                cd_val = None

            if self._re_sweep_active and returncode == 0 and cd_val is not None:
                re_val = self._re_sweep_res[self._re_sweep_index]
                self._re_sweep_points.append((re_val, cd_val))
                self._re_sweep_index += 1
                if self._re_sweep_index < len(self._re_sweep_commands):
                    level = self._re_sweep_index + 1
                    total = len(self._re_sweep_commands)
                    self._start_command(
                        self._re_sweep_commands[self._re_sweep_index],
                        label=f"Re sweep: run {level}/{total}…",
                    )
                    return
                self._re_sweep_active = False
                if self._re_sweep_points:
                    re_vals, cd_vals = zip(*self._re_sweep_points)
                    self.re_sweep_chart.plot_re_sweep(re_vals, cd_vals)
                self.summary_label.setText("Re sweep complete")
            elif self._re_sweep_active:
                self._re_sweep_active = False

            if self._grid_study_active and returncode == 0 and cd_val is not None:
                self._grid_study_cds.append(cd_val)
                self._grid_study_index += 1
                if self._grid_study_index < len(self._grid_study_commands):
                    level = self._grid_study_index + 1
                    self._start_command(
                        self._grid_study_commands[self._grid_study_index],
                        label=f"Grid study: running level {level}/3…",
                    )
                    return
                self._grid_study_active = False
                if self._saved_params_before_grid is not None:
                    target = self._current_params()
                    target.clear()
                    target.update(self._saved_params_before_grid)
                    self._rebuild_form(refresh=False)
                self._update_validation_panel(cd=cd_val, grid_cd_values=self._grid_study_cds)
                self.summary_label.setText("Grid study complete")
            elif not self._re_sweep_active:
                self._grid_study_active = False
                self._update_validation_panel(
                    cd=cd_val,
                    grid_cd_values=self._grid_study_cds if self._grid_study_cds else None,
                )
                if not self._grid_study_active:
                    self.summary_label.setText(
                        "Run complete" if returncode == 0 else f"Run failed ({returncode})"
                    )
            else:
                self._grid_study_active = False

            self.results_summary.set_results(
                mode=self.config.mode,
                shape=self.shape_box.currentText(),
                re=params.get("re", self._run_metrics.get("re", "?")),
                steps=params.get("steps", "?"),
                metrics=self._run_metrics,
            )
            self.left_tabs.setCurrentIndex(1)
            if returncode == 0 and not self._grid_study_active and not self._re_sweep_active:
                self._adopt_newest_case()
            self._refresh_live_charts()
            self._refresh_checkpoint_status()
            if returncode == 0 and self.config.mode == "3d" and not self._grid_study_active and not self._re_sweep_active:
                QtCore.QTimer.singleShot(200, self._begin_3d_after_run)
            elif returncode == 0 and self.config.mode != "3d":
                self.viewport.clear_scene("Run a 3D case to load the interactive wind tunnel.")
            if not self._grid_study_active and not self._re_sweep_active:
                if returncode == 0:
                    detail = "Run complete"
                    if cd_val is not None:
                        detail += f" · Cd={cd_val:.4f}"
                    elapsed = self._run_metrics.get("elapsed")
                    sps = self._run_metrics.get("steps_per_sec")
                    if elapsed:
                        detail += f" · {elapsed}s"
                    if sps:
                        detail += f" · {sps} steps/s"
                    self._finish_run_progress(success=True, message=detail)
                else:
                    self._finish_run_progress(
                        success=False,
                        message=f"Run failed (exit code {returncode}). See log for details.",
                    )
            self._restore_window()


def launch_gui(repo_root: str = ".") -> int:
    import sys

    if not HAS_QT:
        raise ImportError("PySide6 is not installed. Install it with `python3 -m pip install PySide6`.")

    print("  Initializing Qt...", flush=True)
    fmt = QtGui.QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(0)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    def _bring_app_forward(state: QtCore.Qt.ApplicationState) -> None:
        if state != QtCore.Qt.ApplicationState.ApplicationActive:
            return
        for widget in app.topLevelWidgets():
            if isinstance(widget, QtWidgets.QMainWindow) and widget.isMinimized():
                widget.showNormal()
                widget.raise_()
                widget.activateWindow()

    app.applicationStateChanged.connect(_bring_app_forward)

    if os.environ.get("TERM_PROGRAM") in {"vscode", "cursor"}:
        print(
            "  Note: if no window appears, run `python3 gui.py` from Terminal.app "
            "or double-click `open_gui.command`.",
            flush=True,
        )

    print("  Building main window...", flush=True)
    window = MainWindow(repo_root)

    print("  Opening window...", flush=True)
    window.show()
    window.raise_()
    window.activateWindow()
    app.processEvents()

    print("  Ready.", flush=True)
    return app.exec()

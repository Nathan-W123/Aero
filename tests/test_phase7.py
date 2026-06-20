"""
Phase 7 tests — GUI integration.

Focus on the testable non-Qt logic plus launcher behavior when Qt is missing.
"""

import sys
from pathlib import Path

import pytest

from aero.gui import app as gui_app
from aero.gui.run_data import chart_data_from_run
from aero.gui.state import (
    GuiConfig,
    build_command,
    latest_volume_file,
    list_preview_images,
    list_volume_files,
    parse_run_results,
    pick_preview_image,
)
from aero.run_manifest import write_run_manifest


class TestGuiStateCommands:
    def test_build_command_2d_cylinder(self):
        config = GuiConfig(mode="2d")
        config.shape_2d = "cylinder"
        cmd = build_command(config)
        assert cmd[0] == sys.executable
        assert cmd[1:4] == ["cli.py", "--shape", "cylinder"]
        assert "--collision" in cmd
        assert "--radius" in cmd
        assert "--save-case" in cmd
        assert "--check-every" in cmd

    def test_build_command_3d_cylinder(self):
        config = GuiConfig(mode="3d")
        config.shape_3d = "cylinder"
        cmd = build_command(config)
        assert cmd[0] == sys.executable
        assert cmd[1:4] == ["cli3d.py", "--shape", "cylinder"]
        assert "--length" in cmd
        assert "--viz3d" in cmd
        assert "--export-vtk" in cmd
        assert "--check-every" in cmd

    def test_progress_check_every_scales_with_run_length(self):
        from aero.gui.state import progress_check_every

        assert progress_check_every(500) <= 63
        assert progress_check_every(5000) >= 50
        assert progress_check_every(80000) <= 1000

    def test_parse_progress_line_2d(self):
        from aero.gui.state import parse_progress_line

        snap = parse_progress_line("  step   1000/5000  Cd=+1.2345  Cl=+0.0123  [420 steps/s]")
        assert snap is not None
        assert snap.step == 1000
        assert snap.total_steps == 5000
        assert snap.cd == 1.2345
        assert snap.cl == 0.0123
        assert snap.steps_per_sec == 420

    def test_parse_progress_line_3d(self):
        from aero.gui.state import parse_progress_line

        snap = parse_progress_line("  step    250/500  Cd=+2.6100  Cl_y=+0.0010  [38 steps/s]")
        assert snap is not None
        assert snap.step == 250
        assert snap.cl == 0.0010


class TestGuiCases:
    def test_list_saved_cases_reads_existing_case(self):
        from aero.gui.cases import list_saved_cases

        repo = Path(__file__).resolve().parents[1]
        entries = list_saved_cases(repo)
        assert isinstance(entries, list)
        if entries:
            assert entries[0].name
            assert entries[0].path.is_dir()

    def test_apply_case_config_2d(self):
        from aero.gui.cases import apply_case_config
        from aero.gui.state import GuiConfig

        config = GuiConfig(mode="3d")
        apply_case_config(config, {
            "mode": "2d",
            "shape": "cylinder",
            "re": 100,
            "steps": 300,
            "nx": 200,
            "ny": 100,
            "u0": 0.05,
            "radius": 20,
            "backend": "auto",
            "collision": "bgk",
            "wall_bc": "slip",
            "inlet_bc": "velocity",
            "outlet_bc": "convective",
        })
        assert config.mode == "2d"
        assert config.shape_2d == "cylinder"
        assert config.params_2d["nx"] == "200"
        assert config.params_2d["radius"] == "20"

    def test_chart_manifest_for_case_from_history(self, tmp_path):
        from aero.gui.cases import chart_manifest_for_case

        case_dir = tmp_path / "sphere_re20_test"
        case_dir.mkdir()
        (case_dir / "config.json").write_text(
            '{"mode":"3d","shape":"sphere","re":20,"steps":100,"nz":48}'
        )
        (case_dir / "history.json").write_text(
            '{"Cd_history":[1.0,1.1,1.2],"Cly_history":[0.0,0.01,0.02]}'
        )
        manifest = chart_manifest_for_case(case_dir)
        assert manifest is not None
        assert manifest["Cd_history"] == [1.0, 1.1, 1.2]
        assert manifest["Cly_history"] == [0.0, 0.01, 0.02]

    def test_list_preview_images_filters_pngs(self, tmp_path):
        (tmp_path / "a.png").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        (tmp_path / "c.PNG").write_text("x")
        config = GuiConfig(mode="3d", output_dir_3d=str(tmp_path))
        paths = list_preview_images(config)
        assert [path.name for path in paths] == ["a.png", "c.PNG"]

    def test_list_preview_images_handles_missing_dir(self):
        config = GuiConfig(mode="2d", output_dir_2d="./does-not-exist")
        assert list_preview_images(config) == []

    def test_list_volume_files_filters_vti(self, tmp_path):
        (tmp_path / "a.vti").write_text("x")
        (tmp_path / "b.png").write_text("x")
        (tmp_path / "c.VTI").write_text("x")
        config = GuiConfig(mode="3d", output_dir_3d=str(tmp_path))
        paths = list_volume_files(config)
        assert [path.name for path in paths] == ["a.vti", "c.VTI"]

    def test_latest_volume_file_uses_newest_timestamp(self, tmp_path):
        old = tmp_path / "old.vti"
        new = tmp_path / "new.vti"
        old.write_text("a")
        new.write_text("b")
        old.touch()
        new.touch()
        config = GuiConfig(mode="3d", output_dir_3d=str(tmp_path))
        assert latest_volume_file(config) == new

    def test_pick_preview_image_prefers_specific_keywords(self, tmp_path):
        (tmp_path / "sphere_re20_slice_pressure.png").write_text("x")
        (tmp_path / "sphere_re20_cd_history.png").write_text("x")
        paths = sorted(tmp_path.glob("*.png"))
        assert pick_preview_image(paths, ("cd", "history")) == tmp_path / "sphere_re20_cd_history.png"
        assert pick_preview_image(paths, ("pressure",)) == tmp_path / "sphere_re20_slice_pressure.png"

    def test_parse_run_results_extracts_coefficients(self):
        log = """
        Re        : 20.00
          Cd    (mean) = 2.6196  ±  0.0123
          Cl_y  (mean) = 0.0012  ±  0.0004
          Cl_z  (mean) = -0.0003  ±  0.0002
          Elapsed      = 12.4 s  (40 steps/s)
        """
        metrics = parse_run_results(log)
        assert metrics["cd_mean"] == "2.6196"
        assert metrics["cd_std"] == "0.0123"
        assert metrics["cl_y_mean"] == "0.0012"
        assert metrics["steps_per_sec"] == "40"

    def test_run_manifest_roundtrip(self, tmp_path):
        write_run_manifest(
            tmp_path,
            {"mode": "3d", "Cd_history": [1.0, 1.5, 2.0], "Cly_history": [0.0, 0.1, 0.2]},
        )
        manifest, _ = chart_data_from_run(
            GuiConfig(mode="3d", output_dir_3d=str(tmp_path)),
            tmp_path,
        )
        assert manifest is not None
        assert manifest["Cd_history"] == [1.0, 1.5, 2.0]


class TestGuiLauncher:
    def test_launch_gui_raises_without_qt(self, monkeypatch):
        monkeypatch.setattr(gui_app, "HAS_QT", False)
        with pytest.raises(ImportError):
            gui_app.launch_gui(".")

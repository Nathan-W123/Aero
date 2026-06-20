"""
Phase 6 tests — 3D visualization.

These tests validate the expanded visualization API while remaining robust when
PyVista is not installed in the environment.
"""

import numpy as np
import pytest

from cli3d import build_parser as build_parser_3d
from aero.visualization3d import (
    HAS_PYVISTA,
    advect_particles,
    build_particle_glyphs,
    build_velocity_glyphs,
    build_image_data,
    extract_solid_surface,
    sample_velocity,
    seed_particles,
    save_all_3d,
    save_pyvista_artifacts,
    save_slices,
)


def _fake_result(Nz=6, Ny=8, Nx=12):
    rho = np.ones((Nz, Ny, Nx), dtype=np.float64)
    ux = np.full((Nz, Ny, Nx), 0.05, dtype=np.float64)
    uy = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    uz = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    return {
        "rho": rho,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "Cd_mean": 1.23,
        "Cly_mean": 0.01,
        "Clz_mean": 0.0,
        "Cd_history": [1.0, 1.1, 1.2],
        "Cly_history": [0.0, 0.01, 0.02],
    }


class TestCli3DVisualizationFlags:
    def test_cli_accepts_viz3d_and_export_vtk_flags(self):
        args = build_parser_3d().parse_args(["--viz3d", "all", "--export-vtk"])
        assert args.viz3d == "all"
        assert args.export_vtk is True


class TestVisualization3DSlices:
    def test_save_slices_writes_expected_pngs(self, tmp_path):
        result = _fake_result()
        solid = np.zeros((6, 8, 12), dtype=bool)
        written = save_slices(result, solid, u0=0.05, Re=20.0, shape_name="sphere", steps=3, output_dir=str(tmp_path))
        names = {path.name for path in written}
        assert "sphere_re20_slice_velocity.png" in names
        assert "sphere_re20_slice_pressure.png" in names
        assert "sphere_re20_slice_vorticity.png" in names
        assert "sphere_re20_cd_history.png" in names
        for path in written:
            assert path.exists()

    def test_save_all_3d_slices_mode_writes_only_slice_outputs(self, tmp_path):
        result = _fake_result()
        solid = np.zeros((6, 8, 12), dtype=bool)
        written = save_all_3d(
            result, solid, u0=0.05, Re=20.0, shape_name="sphere", steps=3,
            output_dir=str(tmp_path), viz_mode="slices", export_vtk=True,
        )
        assert written
        assert all(path.suffix == ".png" for path in written)
        assert not any(path.suffix == ".vti" for path in written)

    def test_save_all_3d_rejects_unknown_mode(self, tmp_path):
        result = _fake_result()
        solid = np.zeros((6, 8, 12), dtype=bool)
        with pytest.raises(ValueError):
            save_all_3d(result, solid, 0.05, 20.0, "sphere", 3, str(tmp_path), viz_mode="weird")


class TestVisualization3DPyVista:
    def test_build_image_data_requires_pyvista(self):
        result = _fake_result()
        solid = np.zeros((6, 8, 12), dtype=bool)
        if not HAS_PYVISTA:
            with pytest.raises(ImportError):
                build_image_data(result, solid)
        else:
            grid = build_image_data(result, solid)
            assert grid.n_cells == solid.size
            assert "umag" in grid.cell_data
            assert "solid" in grid.cell_data

    def test_save_pyvista_artifacts_noop_without_pyvista(self, tmp_path):
        result = _fake_result()
        solid = np.zeros((6, 8, 12), dtype=bool)
        written = save_pyvista_artifacts(
            result,
            solid,
            u0=0.05,
            Re=20.0,
            shape_name="sphere",
            output_dir=str(tmp_path),
            export_vtk=True,
            render_images=False,
        )
        if not HAS_PYVISTA:
            assert written == []
        else:
            assert written
            assert any(path.suffix == ".vti" for path in written)

    def test_interactive_scene_helpers_build_geometry(self):
        if not HAS_PYVISTA:
            pytest.skip("PyVista not installed")
        result = _fake_result()
        solid = np.zeros((6, 8, 12), dtype=bool)
        solid[:, 3:5, 5:7] = True
        grid = build_image_data(result, solid)
        surface = extract_solid_surface(grid)
        glyphs = build_velocity_glyphs(grid, stride=2, max_points=200)
        particles = seed_particles(grid, count=24)
        vectors, speed, sampled_solid = sample_velocity(grid, particles)
        advanced, advanced_vectors, advanced_speed = advect_particles(grid, particles, dt=0.5)
        particle_glyphs = build_particle_glyphs(advanced, advanced_vectors, advanced_speed)
        assert surface.n_cells > 0
        assert glyphs.n_cells > 0
        assert particles.shape == (24, 3)
        assert vectors.shape == (24, 3)
        assert speed.shape == (24,)
        assert sampled_solid.shape == (24,)
        assert advanced.shape == (24, 3)
        assert particle_glyphs.n_cells > 0

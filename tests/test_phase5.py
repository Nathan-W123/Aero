"""
Phase 5 tests.

Coverage:
  - 3D cylinder geometry
  - 2D/3D CLI collision flag parsing
  - 2D/3D solver MRT collision path
"""

import numpy as np
import pytest

from cli import build_parser as build_parser_2d
from cli3d import build_parser as build_parser_3d
from aero.geometry3d.cylinder3d import Cylinder3D
from aero.geometry.cylinder import Cylinder
from aero.geometry3d.sphere import Sphere
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D


class TestCylinder3DGeometry:
    def test_mark_solid_shape_and_dtype(self):
        geom = Cylinder3D(radius=4.0, length=10.0)
        solid = geom.mark_solid(24, 24, 48)
        assert solid.shape == (24, 24, 48)
        assert solid.dtype == np.bool_

    def test_reference_length_is_diameter(self):
        assert Cylinder3D(radius=6.0, length=12.0).reference_length() == pytest.approx(12.0)

    def test_center_matches_fractional_coordinates(self):
        cx, cy, cz = Cylinder3D(radius=5.0, length=8.0, cx_frac=0.4, cy_frac=0.25, cz_frac=0.75).center(40, 60, 100)
        assert cx == pytest.approx(40.0)
        assert cy == pytest.approx(15.0)
        assert cz == pytest.approx(30.0)

    def test_solid_count_scales_like_cylinder_volume(self):
        radius = 5.0
        length = 14.0
        geom = Cylinder3D(radius=radius, length=length)
        solid = geom.mark_solid(40, 40, 80)
        expected = np.pi * radius**2 * length
        assert 0.7 * expected <= solid.sum() <= 1.3 * expected


class TestCliCollisionFlags:
    def test_cli2d_accepts_collision_flag(self):
        args = build_parser_2d().parse_args(["--collision", "mrt", "--shape", "cylinder"])
        assert args.collision == "mrt"

    def test_cli3d_accepts_collision_and_cylinder_shape(self):
        args = build_parser_3d().parse_args(
            ["--shape", "cylinder", "--collision", "mrt", "--radius", "6", "--length", "18"]
        )
        assert args.shape == "cylinder"
        assert args.collision == "mrt"
        assert args.length == pytest.approx(18.0)


class TestSolverCollisionModes:
    def test_solver2d_rejects_unknown_collision(self):
        solid = np.zeros((20, 40), dtype=bool)
        with pytest.raises(ValueError):
            Solver(Ny=20, Nx=40, solid=solid, omega=1.2, u0=0.05, D=10.0, collision="foo")

    def test_solver3d_rejects_unknown_collision(self):
        solid = np.zeros((8, 8, 16), dtype=bool)
        with pytest.raises(ValueError):
            Solver3D(Nz=8, Ny=8, Nx=16, solid=solid, omega=1.2, u0=0.05, D=10.0, collision="foo")

    def test_solver2d_mrt_runs_without_nan(self):
        geom = Cylinder(radius=6.0, cx_frac=0.35, cy_frac=0.5)
        solid = geom.mark_solid(40, 80)
        solver = Solver(
            Ny=40,
            Nx=80,
            solid=solid,
            omega=1.3,
            u0=0.05,
            D=12.0,
            backend="numpy",
            collision="mrt",
        )
        result = solver.run(steps=30, check_every=100, verbose=False)
        assert solver.collision == "mrt"
        assert len(result["Cd_history"]) == 30
        assert not np.isnan(solver.f).any()

    def test_solver3d_mrt_runs_without_nan(self):
        geom = Sphere(radius=3.0, cx_frac=0.35, cy_frac=0.5, cz_frac=0.5)
        solid = geom.mark_solid(12, 12, 24)
        solver = Solver3D(
            Nz=12,
            Ny=12,
            Nx=24,
            solid=solid,
            omega=1.2,
            u0=0.05,
            D=6.0,
            backend="numpy",
            collision="mrt",
        )
        result = solver.run(steps=10, check_every=100, verbose=False)
        assert solver.collision == "mrt"
        assert len(result["Cd_history"]) == 10
        assert not np.isnan(solver.f).any()

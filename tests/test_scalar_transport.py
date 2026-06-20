"""Tests for passive scalar transport plumbing."""

import numpy as np

from cli import build_parser as build_parser_2d
from cli3d import build_parser as build_parser_3d
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D
from aero.visualization3d import HAS_PYVISTA, build_image_data, save_slices


def test_cli2d_accepts_scalar_flags():
    args = build_parser_2d().parse_args(
        ["--scalar", "--scalar-hot", "1.2", "--scalar-cold", "0.2", "--scalar-diffusivity", "0.02"]
    )
    assert args.thermal is True
    assert args.T_hot == 1.2
    assert args.T_cold == 0.2
    assert args.alpha_T == 0.02


def test_cli3d_accepts_scalar_flags():
    args = build_parser_3d().parse_args(
        ["--scalar", "--scalar-hot", "1.1", "--scalar-cold", "0.1", "--gravity", "1e-4", "--buoyancy"]
    )
    assert args.thermal is True
    assert args.T_hot == 1.1
    assert args.T_cold == 0.1
    assert args.g_gravity == 1e-4
    assert args.buoyancy is True


def test_solver2d_returns_scalar_outputs_and_checkpoint_roundtrip(tmp_path):
    solid = np.zeros((16, 24), dtype=bool)
    solver = Solver(
        Ny=16,
        Nx=24,
        solid=solid,
        omega=1.5,
        u0=0.01,
        D=8.0,
        backend="numpy",
        thermal=True,
        T_hot=1.0,
        T_cold=0.0,
        alpha_T=0.05,
    )
    result = solver.run(steps=8, check_every=100, verbose=False)
    assert result["scalar"] is not None
    assert result["scalar_stats"]["enabled"] is True
    assert result["scalar"].shape == (16, 24)

    ckpt = tmp_path / "scalar2d.npz"
    solver.save_checkpoint(str(ckpt))
    loaded = Solver.from_checkpoint(str(ckpt))
    assert loaded.thermal is True
    assert loaded.g is not None
    assert loaded.alpha_T == 0.05


def test_solver3d_returns_scalar_outputs_and_checkpoint_roundtrip(tmp_path):
    solid = np.zeros((6, 8, 12), dtype=bool)
    solver = Solver3D(
        Nz=6,
        Ny=8,
        Nx=12,
        solid=solid,
        omega=1.4,
        u0=0.01,
        D=6.0,
        backend="numpy",
        thermal=True,
        T_hot=1.0,
        T_cold=0.0,
        alpha_T=0.05,
    )
    result = solver.run(steps=4, check_every=100, verbose=False)
    assert result["scalar"] is not None
    assert result["scalar_stats"]["enabled"] is True
    assert result["scalar"].shape == (6, 8, 12)

    ckpt = tmp_path / "scalar3d.npz"
    solver.save_checkpoint(str(ckpt))
    loaded = Solver3D.from_checkpoint(str(ckpt))
    assert loaded.thermal is True
    assert loaded.g is not None
    assert loaded.alpha_T == 0.05


def test_visualization3d_scalar_slice_and_volume_metadata(tmp_path):
    solid = np.zeros((6, 8, 12), dtype=bool)
    result = {
        "rho": np.ones((6, 8, 12), dtype=np.float64),
        "ux": np.full((6, 8, 12), 0.05, dtype=np.float64),
        "uy": np.zeros((6, 8, 12), dtype=np.float64),
        "uz": np.zeros((6, 8, 12), dtype=np.float64),
        "scalar": np.linspace(0.0, 1.0, 6 * 8 * 12, dtype=np.float64).reshape(6, 8, 12),
        "Cd_mean": 1.0,
        "Cly_mean": 0.0,
        "Cd_history": [1.0],
        "Cly_history": [0.0],
    }
    written = save_slices(result, solid, u0=0.05, Re=20.0, shape_name="sphere", steps=1, output_dir=str(tmp_path))
    assert any(path.name.endswith("_slice_scalar.png") for path in written)
    if HAS_PYVISTA:
        grid = build_image_data(result, solid)
        assert "scalar" in grid.cell_data

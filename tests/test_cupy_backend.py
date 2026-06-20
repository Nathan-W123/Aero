"""CuPy GPU backend tests — all skip automatically when cupy is not installed."""

import pytest
import numpy as np

cupy = pytest.importorskip("cupy", reason="cupy not installed — GPU tests skipped")


from aero.geometry3d.cylinder3d import Cylinder3D
from aero.lbm.solver3d import Solver3D


def _make_solver(backend: str) -> Solver3D:
    cyl = Cylinder3D(radius=5.0, length=16.0, cx_frac=1/3, cy_frac=0.5, cz_frac=0.5)
    solid = cyl.mark_solid(16, 32, 64)
    return Solver3D(
        Nz=16, Ny=32, Nx=64,
        solid=solid,
        omega=1.2,
        u0=0.05,
        D=10.0,
        backend=backend,
        collision="bgk",
    )


def test_cupy_matches_numpy():
    """CuPy and numpy backends must agree on Cd to within floating-point noise."""
    sol_np = _make_solver("numpy")
    sol_cp = _make_solver("cupy")
    result_np = sol_np.run(steps=200, check_every=200, verbose=False)
    result_cp = sol_cp.run(steps=200, check_every=200, verbose=False)
    assert abs(result_np["Cd_mean"] - result_cp["Cd_mean"]) < 1e-10


def test_cupy_not_available_raises():
    """When cupy is absent, solver must raise ImportError with helpful message."""
    import sys
    import importlib

    # Temporarily hide cupy by replacing it in sys.modules with a broken import
    orig = sys.modules.get("cupy")
    sys.modules["cupy"] = None  # type: ignore
    try:
        with pytest.raises(ImportError, match="cupy"):
            _make_solver("cupy")
    finally:
        if orig is None:
            del sys.modules["cupy"]
        else:
            sys.modules["cupy"] = orig

"""Tests for surface observables: Cp, wall shear stress, y+."""

import numpy as np
import pytest

from aero.lbm.d2q9 import E, W, compute_feq
from aero.lbm.d3q19 import E3, W3
from aero.lbm.boundary import build_surface_links
from aero.lbm.boundary3d import build_surface_links_3d
from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D
from aero.surface import (
    surface_fields,
    surface_nodes,
    surface_profile,
    surface_summary,
)


def _channel_solid(ny, nx):
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------

def test_flat_wall_normal_is_axial():
    """
    A flat wall gives three links per node in D2Q9 and only one of them points
    along the normal; the node normal must come out axial regardless.
    """
    ny, nx = 6, 5
    coords, normals, counts = surface_nodes(
        build_surface_links(_channel_solid(ny, nx))[0], E, W
    )
    assert np.all(counts == 3)
    for c, n in zip(coords, normals):
        expected = np.array([0.0, 1.0]) if c[0] == 1 else np.array([0.0, -1.0])
        np.testing.assert_allclose(n, expected, atol=1e-12)


def test_node_normals_are_unit_length():
    ny, nx = 24, 24
    yy, xx = np.mgrid[0:ny, 0:nx]
    solid = ((yy - 12) ** 2 + (xx - 12) ** 2) <= 36
    _, normals, _ = surface_nodes(build_surface_links(solid)[0], E, W)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)


def test_cylinder_normals_point_outward():
    """On a circle the node normal must point away from the centre."""
    ny, nx = 32, 32
    cy = cx = 16.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    solid = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 49
    coords, normals, _ = surface_nodes(build_surface_links(solid)[0], E, W)
    # normals are lattice vectors (x, y); coords are grid indices (y, x)
    radial = np.stack([coords[:, 1] - cx, coords[:, 0] - cy], axis=1)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    assert np.mean(np.einsum("nd,nd->n", normals, radial)) > 0.9


def test_surface_nodes_handles_an_empty_table():
    coords, normals, counts = surface_nodes(np.empty((0, 3), dtype=np.int32), E, W)
    assert coords.shape == (0, 2) and normals.shape == (0, 2) and counts.shape == (0,)


# ---------------------------------------------------------------------------
# Field values
# ---------------------------------------------------------------------------

def test_quiescent_flow_has_no_wall_shear():
    ny, nx = 8, 8
    solid = _channel_solid(ny, nx)
    rho = np.ones((ny, nx))
    f = np.ascontiguousarray(compute_feq(rho, np.zeros((ny, nx)), np.zeros((ny, nx))))
    out = surface_fields(
        f, build_surface_links(solid)[0], e=E, w=W, omega=1.5, u_ref=0.05
    )
    np.testing.assert_allclose(out["tau_wall"], 0.0, atol=1e-18)
    np.testing.assert_allclose(out["cp"], 0.0, atol=1e-14)
    np.testing.assert_allclose(out["y_plus"], 0.0, atol=1e-18)


def test_cp_tracks_the_local_density():
    ny, nx = 8, 8
    solid = _channel_solid(ny, nx)
    rho = np.full((ny, nx), 1.02)
    f = np.ascontiguousarray(compute_feq(rho, np.zeros((ny, nx)), np.zeros((ny, nx))))
    out = surface_fields(
        f, build_surface_links(solid)[0], e=E, w=W, omega=1.5,
        rho_ref=1.0, u_ref=0.05,
    )
    expect = (1.0 / 3.0) * (1.02 - 1.0) / (0.5 * 1.0 * 0.05 ** 2)
    np.testing.assert_allclose(out["cp"], expect, rtol=1e-12)


def test_empty_geometry_gives_empty_fields():
    ny, nx = 6, 6
    f = np.ascontiguousarray(
        compute_feq(np.ones((ny, nx)), np.zeros((ny, nx)), np.zeros((ny, nx)))
    )
    links = np.empty((0, 3), dtype=np.int32)
    out = surface_fields(f, links, e=E, w=W, omega=1.5)
    assert out["tau_wall"].size == 0
    assert surface_summary(out) == {"links": 0}


@pytest.mark.parametrize("omega", [1.2, 1.4, 1.7])
def test_wall_shear_matches_the_exact_poiseuille_value(omega):
    """
    Force-driven channel: the viscous stress at the first fluid node is
    ``rho nu du/dy`` there, and that is an exact number.

    This is the test that catches using a link direction as the normal — that
    mistake computes the shear on three different planes and comes out about a
    third of the right value.
    """
    pytest.importorskip("numba")
    ny, nx = 41, 6
    accel = 2e-6
    nu = (1.0 / omega - 0.5) / 3.0
    solid = _channel_solid(ny, nx)

    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=omega, u0=0.0, D=float(ny),
        backend="numba", wall_bc="noslip", inlet_bc="none", outlet_bc="none",
        body_force_x=accel,
    )
    solver.run(steps=40000, check_every=10 ** 9, verbose=False)
    _, ux, _ = solver.macroscopic()

    # fit the interior profile to locate the effective walls
    y = np.arange(1, ny - 1, dtype=float)
    c2, c1, c0 = np.polyfit(y, ux[1:-1, nx // 2], 2)
    y0, y1 = sorted(np.roots([c2, c1, c0]))
    # exact shear at the first fluid node, one cell in from the solid row
    dudy = abs((accel / (2.0 * nu)) * (y0 + y1 - 2.0 * 1.0))
    expect = 1.0 * nu * dudy

    fields = solver.surface_fields()
    assert fields["tau_wall"].size > 0
    assert float(np.mean(fields["tau_wall"])) == pytest.approx(expect, rel=2e-3)


def test_friction_velocity_and_y_plus_are_consistent():
    pytest.importorskip("numba")
    ny, nx = 41, 6
    omega = 1.4
    nu = (1.0 / omega - 0.5) / 3.0
    solver = Solver(
        Ny=ny, Nx=nx, solid=_channel_solid(ny, nx), omega=omega, u0=0.0,
        D=float(ny), backend="numba", wall_bc="noslip", inlet_bc="none",
        outlet_bc="none", body_force_x=2e-6,
    )
    solver.run(steps=20000, check_every=10 ** 9, verbose=False)
    f = solver.surface_fields()
    np.testing.assert_allclose(
        f["u_tau"], np.sqrt(f["tau_wall"] / f["rho"]), rtol=1e-12
    )
    np.testing.assert_allclose(f["y_plus"], 0.5 * f["u_tau"] / nu, rtol=1e-12)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def test_summary_is_json_safe_and_reports_wall_resolution():
    import json
    ny, nx = 24, 48
    yy, xx = np.mgrid[0:ny, 0:nx]
    solid = ((yy - 12) ** 2 + (xx - 12) ** 2) <= 16
    solver = Solver(
        Ny=ny, Nx=nx, solid=solid, omega=1.6, u0=0.06, D=8.0, backend="numpy",
    )
    result = solver.run(steps=40, check_every=1000, verbose=False)
    summary = result["surface"]
    json.dumps(summary)
    assert summary["links"] > 0
    assert "wall_resolution" in summary
    assert summary["y_plus_min"] <= summary["y_plus_mean"] <= summary["y_plus_max"]
    assert "cp_profile_y" in summary


@pytest.mark.parametrize(
    "y_plus, fragment",
    [(0.5, "wall-resolved"), (3.0, "viscous sublayer"),
     (12.0, "buffer layer"), (80.0, "log layer")],
)
def test_wall_resolution_verdicts(y_plus, fragment):
    from aero.surface import _wall_resolution_verdict
    assert fragment in _wall_resolution_verdict(y_plus)


def test_surface_profile_bins_by_axis():
    fields = {
        "coords": np.array([[1, 0], [1, 5], [3, 2]]),
        "cp": np.array([2.0, 4.0, 10.0]),
    }
    prof = surface_profile(fields, axis=0, length=5, quantity="cp")
    assert prof["cp"][1] == pytest.approx(3.0)      # mean of 2 and 4
    assert prof["cp"][3] == pytest.approx(10.0)
    assert prof["cp"][0] == 0.0                     # no links there
    assert prof["count"] == [0, 2, 0, 1, 0]


def test_surface_profile_on_empty_geometry():
    fields = {"coords": np.empty((0, 2), dtype=np.intp), "cp": np.empty(0)}
    prof = surface_profile(fields, axis=0, length=3, quantity="cp")
    assert prof["cp"] == [0.0, 0.0, 0.0]


def test_solver_3d_reports_surface_observables():
    nz, ny, nx = 10, 12, 16
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    solid = ((zz - 5) ** 2 + (yy - 6) ** 2 + (xx - 5) ** 2) <= 9
    solver = Solver3D(
        Nz=nz, Ny=ny, Nx=nx, solid=solid, omega=1.5, u0=0.05, D=6.0,
        backend="numpy",
    )
    result = solver.run(steps=20, check_every=1000, verbose=False)
    fields = result["surface_fields"]
    assert fields["coords"].shape[1] == 3
    assert fields["normal"].shape[1] == 3
    np.testing.assert_allclose(np.linalg.norm(fields["normal"], axis=1), 1.0, atol=1e-12)
    assert result["surface"]["links"] == fields["rho"].size
    assert "cp_profile_z" in result["surface"]


def test_3d_flat_wall_normal_is_axial():
    nz, ny, nx = 6, 6, 6
    solid = np.zeros((nz, ny, nx), dtype=bool)
    solid[:, 0, :] = True
    coords, normals, _ = surface_nodes(build_surface_links_3d(solid)[0], E3, W3)
    # lattice order is (x, y, z); the wall is at low y so the normal is +y
    expected = np.tile(np.array([0.0, 1.0, 0.0]), (normals.shape[0], 1))
    np.testing.assert_allclose(normals, expected, atol=1e-12)

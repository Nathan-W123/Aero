"""
Tests for the D3Q27 lattice and the lattice-generic 3D machinery.

The shape of this file follows the argument for D3Q27 existing at all:

* the velocity set has to be a valid quadrature (moments),
* it has to reproduce the same hydrodynamics as D3Q19 (viscosity, walls),
* and it has to actually fix the thing it costs ~23% more to fix (Galilean
  invariance).  If the last one does not hold there is no reason to ship it.
"""

import numpy as np
import pytest

from aero.lbm.d3q19 import E3, OPP3, W3, Y_MIR3
from aero.lbm.lattice3d import (
    CS2,
    D3Q19,
    D3Q27,
    Lattice3D,
    compute_feq,
    compute_macroscopic,
    get_lattice3d,
)
from aero.lbm.boundary3d import (
    apply_inlet_sem_3d,
    apply_inlet_velocity_field_3d,
    apply_inlet_zou_he_3d,
    apply_noslip_walls_3d,
    apply_slip_walls_3d,
    build_surface_links_3d,
)
from aero.lbm.solver3d import Solver3D

LATTICES = [D3Q19, D3Q27]


# ---------------------------------------------------------------------------
# The velocity set
# ---------------------------------------------------------------------------

def test_d3q27_has_every_cube_direction_exactly_once():
    seen = {tuple(v) for v in D3Q27.E.tolist()}
    assert len(seen) == 27
    assert seen == {
        (x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
    }


def test_d3q27_extends_d3q19_in_place():
    """
    The first 19 velocity vectors must be D3Q19's, in order.

    Checkpoints on disk store f indexed by direction, so reordering would
    silently reinterpret every saved run.  The *weights* are necessarily
    different -- they come from the 27-point quadrature -- so only the
    velocities carry over.
    """
    np.testing.assert_array_equal(D3Q27.E[:19], E3)
    assert not np.allclose(D3Q27.W[:19], W3)


def test_the_descriptor_reproduces_the_hand_written_d3q19_tables():
    """The derived OPP/Y_MIR must equal the tables they replace."""
    np.testing.assert_array_equal(D3Q19.E, E3)
    np.testing.assert_allclose(D3Q19.W, W3)
    np.testing.assert_array_equal(D3Q19.OPP, OPP3)
    np.testing.assert_array_equal(D3Q19.Y_MIR, Y_MIR3)


@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
def test_opposite_and_mirror_are_involutions(lat):
    idx = np.arange(lat.Q)
    np.testing.assert_array_equal(lat.OPP[lat.OPP[idx]], idx)
    np.testing.assert_array_equal(lat.Y_MIR[lat.Y_MIR[idx]], idx)
    np.testing.assert_array_equal(lat.E[lat.OPP], -lat.E)
    # the mirror flips ey and leaves ex, ez alone
    np.testing.assert_array_equal(lat.E[lat.Y_MIR][:, 0], lat.E[:, 0])
    np.testing.assert_array_equal(lat.E[lat.Y_MIR][:, 1], -lat.E[:, 1])
    np.testing.assert_array_equal(lat.E[lat.Y_MIR][:, 2], lat.E[:, 2])


@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
def test_moments_through_fourth_order_are_exact(lat):
    """Everything Chapman-Enskog needs, on both lattices."""
    err = lat.moment_errors()
    for key in ("mass", "m1", "m2", "m3", "m4"):
        assert err[key] < 1e-14, f"{lat.name} {key} = {err[key]:.3g}"


def test_only_d3q27_carries_the_fully_three_dimensional_moment():
    """
    ``sum_i w_i c_x^2 c_y^2 c_z^2`` is the one representable moment the two
    lattices disagree on, and the origin of the Galilean defect.
    """
    assert D3Q27.moment_errors()["m222"] < 1e-14
    # D3Q19 has no direction with all three components non-zero, so it scores
    # the moment as 0 instead of cs^6
    got19 = float((D3Q19.W * D3Q19.ex ** 2 * D3Q19.ey ** 2 * D3Q19.ez ** 2).sum())
    assert got19 == 0.0
    assert D3Q19.moment_errors()["m222"] == pytest.approx(CS2 ** 3)


@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
def test_transverse_norm_is_the_same_for_y_and_z(lat):
    """The Zou-He closure spreads y and z momentum with one constant."""
    u = lat.x_plus
    sz = float((lat.W[u] * lat.ez[u] ** 2).sum())
    assert lat.transverse_norm == pytest.approx(sz, rel=1e-14)
    assert lat.transverse_norm == pytest.approx(1.0 / 18.0, rel=1e-14)
    # and the cross terms that would couple the two must vanish
    assert float((lat.W[u] * lat.ey[u] * lat.ez[u]).sum()) == pytest.approx(0.0, abs=1e-18)
    assert float((lat.W[u] * lat.ex[u] * lat.ey[u]).sum()) == pytest.approx(0.0, abs=1e-18)


def test_lattice_lookup():
    assert get_lattice3d("d3q27") is D3Q27
    assert get_lattice3d("D3Q19") is D3Q19
    assert get_lattice3d("d3-q27") is D3Q27
    assert get_lattice3d(D3Q27) is D3Q27
    with pytest.raises(ValueError, match="d3q27"):
        get_lattice3d("d3q15")


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
def test_equilibrium_carries_the_right_mass_and_momentum(lat):
    """True whether or not the third-order term is present."""
    rho = np.array([[[1.05]]])
    u = (np.array([[[0.06]]]), np.array([[[-0.03]]]), np.array([[[0.02]]]))
    feq = compute_feq(rho, *u, lat)
    r, ux, uy, uz = compute_macroscopic(feq, lat)
    assert r[0, 0, 0] == pytest.approx(1.05, rel=1e-14)
    for got, want in zip((ux, uy, uz), u):
        assert got[0, 0, 0] == pytest.approx(want[0, 0, 0], rel=1e-12)


def test_only_d3q27_enables_the_third_order_term():
    assert D3Q27.h3 and D3Q27.h3_factor == 1.0
    assert not D3Q19.h3 and D3Q19.h3_factor == 0.0


def _xyz_third_moment(lat, rho, ux, uy, uz):
    """``sum_i c_ix c_iy c_iz f_i^eq``, which should be ``rho ux uy uz``."""
    shape = (1, 1, 1)
    feq = compute_feq(
        np.full(shape, rho), np.full(shape, ux), np.full(shape, uy),
        np.full(shape, uz), lat,
    )[:, 0, 0, 0]
    return float((lat.ex * lat.ey * lat.ez * feq).sum())


def test_d3q19_cannot_represent_the_xyz_third_moment_at_all():
    """
    The defect in its sharpest form.

    A correct equilibrium has ``sum_i c_ix c_iy c_iz f_i^eq = rho ux uy uz``.
    D3Q19 has no direction with all three components non-zero, so that sum is
    *identically zero* whatever the equilibrium -- adding the third-order term
    to D3Q19 cannot help, which is why ``h3`` belongs to the lattice and is not
    a user option.
    """
    args = (1.0, 0.1, 0.07, -0.05)
    exact = args[0] * args[1] * args[2] * args[3]

    assert _xyz_third_moment(D3Q19, *args) == 0.0
    forced = Lattice3D("d3q19+h3", D3Q19.E, D3Q19.W, h3=True)
    assert _xyz_third_moment(forced, *args) == 0.0, "H3 cannot rescue D3Q19"
    assert _xyz_third_moment(D3Q27, *args) == pytest.approx(exact, rel=1e-12)


def test_the_third_order_term_is_what_makes_the_two_lattices_differ():
    """
    Without it, D3Q27 is D3Q19 with eight extra directions that change nothing
    -- ~23% more work for an identical answer.  Pin that, because it is the
    reason ``h3`` is on by default for D3Q27.
    """
    plain27 = Lattice3D("plain27", D3Q27.E, D3Q27.W, h3=False)
    args = (1.0, 0.1, 0.07, -0.05)
    exact = args[0] * args[1] * args[2] * args[3]

    # the extra directions alone do not deliver the moment
    assert _xyz_third_moment(plain27, *args) == pytest.approx(0.0, abs=1e-18)
    # the term does
    assert _xyz_third_moment(D3Q27, *args) == pytest.approx(exact, rel=1e-12)

    # and it leaves the lower moments exactly alone, so hydrodynamics is safe
    shape = (1, 1, 1)
    fields = [np.full(shape, v) for v in args]
    a = compute_feq(*fields, plain27)
    b = compute_feq(*fields, D3Q27)
    np.testing.assert_allclose(a.sum(axis=0), b.sum(axis=0), rtol=1e-14)
    for comp in (D3Q27.ex, D3Q27.ey, D3Q27.ez):
        np.testing.assert_allclose(
            np.einsum("i,izyx->zyx", comp, a),
            np.einsum("i,izyx->zyx", comp, b), rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _noisy_f(lat, shape, seed=0):
    rng = np.random.default_rng(seed)
    f = np.empty((lat.Q, *shape))
    for i in range(lat.Q):
        f[i] = lat.W[i] * (1.0 + 0.02 * rng.standard_normal(shape))
    return f


@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
@pytest.mark.parametrize(
    "target", [(0.05, 0.0, 0.0), (0.05, 0.01, -0.004), (0.02, -0.01, 0.01)]
)
def test_zou_he_inlet_imposes_exactly_the_requested_velocity(lat, target):
    """
    All three components, to machine precision, on both lattices.

    The transverse components are the ones that were wrong: the old
    hand-written closure spread momentum with the sign flipped and never
    cancelled the ex=0 plane's own transverse momentum, so asking for
    uy = +0.01 produced uy = -0.00987.
    """
    nz, ny = 4, 6
    f = _noisy_f(lat, (nz, ny, 5))
    tx, ty, tz = target
    apply_inlet_velocity_field_3d(
        f, np.full((nz, ny), tx), np.full((nz, ny), ty), np.full((nz, ny), tz),
        lattice=lat,
    )
    _, ux, uy, uz = compute_macroscopic(f, lat)
    assert np.abs(ux[:, :, 0] - tx).max() < 1e-14
    assert np.abs(uy[:, :, 0] - ty).max() < 1e-14
    assert np.abs(uz[:, :, 0] - tz).max() < 1e-14


@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
def test_sem_inlet_imposes_its_fluctuations_with_the_right_sign(lat):
    nz, ny = 4, 6
    f = _noisy_f(lat, (nz, ny, 5), seed=3)
    apply_inlet_sem_3d(
        f, 0.05, np.zeros((nz, ny)), np.full((nz, ny), 0.01),
        np.full((nz, ny), -0.004), lattice=lat,
    )
    _, ux, uy, uz = compute_macroscopic(f, lat)
    assert np.abs(ux[:, :, 0] - 0.05).max() < 1e-14
    assert np.abs(uy[:, :, 0] - 0.01).max() < 1e-14
    assert np.abs(uz[:, :, 0] + 0.004).max() < 1e-14


@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
def test_uniform_inlet_imposes_u0(lat):
    nz, ny = 4, 6
    f = _noisy_f(lat, (nz, ny, 5), seed=7)
    apply_inlet_zou_he_3d(f, 0.06, lattice=lat)
    _, ux, uy, uz = compute_macroscopic(f, lat)
    assert np.abs(ux[:, :, 0] - 0.06).max() < 1e-14
    assert np.abs(uy[:, :, 0]).max() < 1e-14
    assert np.abs(uz[:, :, 0]).max() < 1e-14


@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
def test_noslip_walls_kill_the_wall_normal_momentum(lat):
    f = _noisy_f(lat, (4, 6, 5), seed=11)
    apply_noslip_walls_3d(f, lattice=lat)
    # after bounce-back the unknown set mirrors the known one exactly
    up, dn = lat.y_plus, lat.y_minus
    np.testing.assert_allclose(f[up, :, 0, :], f[lat.OPP[up], :, 0, :])
    np.testing.assert_allclose(f[dn, :, -1, :], f[lat.OPP[dn], :, -1, :])


@pytest.mark.parametrize("lat", LATTICES, ids=lambda l: l.name)
def test_slip_walls_reflect_rather_than_reverse(lat):
    f = _noisy_f(lat, (4, 6, 5), seed=13)
    apply_slip_walls_3d(f, lattice=lat)
    up = lat.y_plus
    np.testing.assert_allclose(f[up, :, 0, :], f[lat.Y_MIR[up], :, 0, :])


def test_link_builder_finds_the_corner_links_only_on_d3q27():
    """A sphere has diagonal surface links; D3Q27 has eight more per node."""
    n = 12
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    solid = ((z - 6) ** 2 + (y - 6) ** 2 + (x - 6) ** 2) <= 9
    links19, _ = build_surface_links_3d(solid, lattice=D3Q19)
    links27, _ = build_surface_links_3d(solid, lattice=D3Q27)
    assert links27.shape[0] > links19.shape[0]
    assert links19[:, 0].max() < 19
    assert links27[:, 0].max() >= 19


# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------

def _sphere(nz, ny, nx, r=2.0):
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    return ((z - nz / 2) ** 2 + (y - ny / 2) ** 2 + (x - nx / 4) ** 2) <= r * r


@pytest.mark.parametrize("lattice", ["d3q19", "d3q27"])
@pytest.mark.parametrize("collision", ["bgk", "trt", "regularized"])
def test_poiseuille_viscosity_is_exact(lattice, collision):
    """
    The third-order term must not disturb the hydrodynamics.

    Force-driven channel has the exact solution d2u/dy2 = -a/nu, so this pins
    the recovered viscosity directly rather than by comparison.
    """
    pytest.importorskip("numba")
    nz, ny, nx = 4, 33, 6
    accel, omega = 1e-6, 1.5
    nu = (1.0 / omega - 0.5) / 3.0
    solver = Solver3D(
        Nz=nz, Ny=ny, Nx=nx, solid=np.zeros((nz, ny, nx), dtype=bool),
        omega=omega, u0=0.0, D=float(ny), backend="numba", lattice=lattice,
        collision=collision, wall_bc="noslip", streamwise_bc="periodic",
        body_force_x=accel,
    )
    solver.run(steps=20000, check_every=10 ** 9, verbose=False)
    _, ux, _, _ = solver.macroscopic()
    y = np.arange(ny, dtype=float)
    c2, _, _ = np.polyfit(y[2:-2], ux[nz // 2, 2:-2, nx // 2], 2)
    assert 2.0 * c2 == pytest.approx(-accel / nu, rel=2e-3)


@pytest.mark.parametrize("collision", ["bgk", "trt", "regularized"])
def test_d3q27_numpy_and_numba_agree(collision):
    pytest.importorskip("numba")
    nz, ny, nx = 8, 10, 16
    common = dict(
        Nz=nz, Ny=ny, Nx=nx, solid=_sphere(nz, ny, nx), omega=1.6, u0=0.05,
        D=4.0, lattice="d3q27", collision=collision,
    )
    a = Solver3D(backend="numpy", **common)
    b = Solver3D(backend="numba", **common)
    a.run(steps=25, check_every=10 ** 9, verbose=False)
    b.run(steps=25, check_every=10 ** 9, verbose=False)
    np.testing.assert_allclose(a.macroscopic()[1], b.macroscopic()[1], rtol=1e-10)


def test_mrt_is_refused_on_d3q27():
    """The 19x19 moment matrix has no meaning for 27 directions."""
    with pytest.raises(ValueError, match="only implemented for D3Q19"):
        Solver3D(
            Nz=4, Ny=6, Nx=8, solid=np.zeros((4, 6, 8), dtype=bool), omega=1.5,
            u0=0.05, D=4.0, lattice="d3q27", collision="mrt",
        )


def test_solver_rejects_an_unknown_lattice():
    with pytest.raises(ValueError, match="unknown 3D lattice"):
        Solver3D(
            Nz=4, Ny=6, Nx=8, solid=np.zeros((4, 6, 8), dtype=bool), omega=1.5,
            u0=0.05, D=4.0, lattice="d3q15",
        )


def test_lattice_survives_a_checkpoint_round_trip(tmp_path):
    nz, ny, nx = 8, 10, 16
    s = Solver3D(
        Nz=nz, Ny=ny, Nx=nx, solid=_sphere(nz, ny, nx), omega=1.6, u0=0.05,
        D=4.0, backend="numpy", lattice="d3q27",
    )
    s.run(steps=5, check_every=10 ** 9, verbose=False)
    path = tmp_path / "ckpt.npz"
    s.save_checkpoint(str(path))
    restored = Solver3D.from_checkpoint(str(path))
    assert restored.lattice.name == "d3q27"
    assert restored.f.shape[0] == 27
    np.testing.assert_allclose(restored.f, s.f)


# ---------------------------------------------------------------------------
# The reason D3Q27 exists
# ---------------------------------------------------------------------------

def _shear_wave_viscosity(lattice, u_adv, omega=1.2, n=24, steps=300, amp=0.01):
    """
    Decay rate of a transverse shear wave riding on a uniform (1,1,1) flow.

    An exactly Galilean-invariant scheme decays it at ``nu k^2`` whatever the
    frame; the departure when ``u_adv`` is switched on is the defect.  The
    advection is along the body diagonal because the moment D3Q19 is missing
    is the fully three-dimensional one -- an axis-aligned mean flow does not
    excite it.
    """
    lat = get_lattice3d(lattice)
    nu = (1.0 / omega - 0.5) / 3.0
    k = 2.0 * np.pi / n
    solver = Solver3D(
        Nz=n, Ny=n, Nx=n, solid=np.zeros((n, n, n), dtype=bool), omega=omega,
        u0=0.0, D=float(n), backend="numpy", lattice=lattice,
        wall_bc="periodic", streamwise_bc="periodic",
    )
    _, yy, _ = np.mgrid[0:n, 0:n, 0:n]
    d = u_adv / np.sqrt(3.0)
    solver.f = np.ascontiguousarray(compute_feq(
        np.ones((n, n, n)), d + amp * np.sin(k * yy),
        np.full((n, n, n), d), np.full((n, n, n), d), lat,
    ))

    def amplitude():
        _, ux, _, _ = solver.macroscopic()
        return 2.0 * np.abs(np.fft.rfft(ux.mean(axis=(0, 2)))[1]) / n

    a0 = amplitude()
    solver.run(steps=steps, check_every=10 ** 9, verbose=False)
    return nu, -np.log(amplitude() / a0) / (k * k * steps)


@pytest.mark.parametrize("lattice", ["d3q19", "d3q27"])
def test_shear_wave_decays_at_the_right_rate_at_rest(lattice):
    """With no mean flow both lattices are fine -- that is the control."""
    nu, measured = _shear_wave_viscosity(lattice, 0.0)
    assert measured == pytest.approx(nu, rel=0.02)


def test_d3q27_removes_the_galilean_invariance_defect():
    """
    The whole justification for the lattice, stated as a number.

    Advecting the wave must not change its decay rate.  On D3Q19 it does, by
    about 4% at u = 0.2; on D3Q27, with the third-order equilibrium, by less
    than 0.02%.  Measured ratio is ~490x, asserted loosely at 50x so the test
    is about the mechanism and not the third digit.
    """
    def defect(lattice):
        nu, at_rest = _shear_wave_viscosity(lattice, 0.0)
        _, advected = _shear_wave_viscosity(lattice, 0.2)
        return abs((advected - at_rest) / nu)

    d19 = defect("d3q19")
    d27 = defect("d3q27")
    assert d19 > 1e-2, d19            # D3Q19 really is that bad
    assert d27 < 1e-3, d27            # D3Q27 really does fix it
    assert d19 / d27 > 50.0, (d19, d27)

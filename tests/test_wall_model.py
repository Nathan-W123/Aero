"""Tests for the equilibrium (law-of-the-wall) near-wall model."""

import numpy as np
import pytest

from aero.lbm.solver import Solver
from aero.lbm.solver3d import Solver3D
from aero.lbm.wall_model import (
    KAPPA,
    apply_wall_model,
    effective_wall_viscosity,
    friction_velocity,
    reichardt_u_plus,
    wall_adjacent_mask,
    wall_shear_stress,
)


# ---------------------------------------------------------------------------
# The law of the wall
# ---------------------------------------------------------------------------

def test_reichardt_reduces_to_the_viscous_sublayer():
    """u+ -> y+ as y+ -> 0."""
    y_plus = np.array([0.01, 0.05, 0.1, 0.5])
    np.testing.assert_allclose(reichardt_u_plus(y_plus), y_plus, rtol=0.02)


def test_reichardt_approaches_the_log_law():
    """u+ -> ln(y+)/kappa + B at large y+."""
    y_plus = np.array([300.0, 1000.0, 3000.0])
    log_law = np.log(y_plus) / KAPPA + 5.2
    np.testing.assert_allclose(reichardt_u_plus(y_plus), log_law, rtol=0.03)


def test_reichardt_is_monotonic_and_smooth():
    y_plus = np.logspace(-2, 4, 400)
    u_plus = reichardt_u_plus(y_plus)
    assert np.all(np.diff(u_plus) > 0.0)
    # no kink through the buffer layer: second difference stays bounded
    assert np.all(np.isfinite(np.diff(u_plus, 2)))


# ---------------------------------------------------------------------------
# Inverting for the friction velocity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("u_tau", [1e-4, 1e-3, 5e-3, 2e-2, 1e-1])
@pytest.mark.parametrize("nu", [1e-4, 1e-2])
def test_friction_velocity_round_trip(u_tau, nu):
    """Build u_p from a known u_tau and recover it."""
    y = 0.5
    y_plus = y * u_tau / nu
    u_p = u_tau * reichardt_u_plus(np.array([y_plus]))[0]
    got = friction_velocity(np.array([u_p]), np.array([y]), nu)[0]
    assert got == pytest.approx(u_tau, rel=1e-8)


def test_zero_velocity_gives_zero_stress():
    assert friction_velocity(np.array([0.0]), np.array([0.5]), 0.01)[0] == 0.0
    assert wall_shear_stress(np.array([0.0]), np.array([0.5]), 0.01)[0] == 0.0


def test_stress_is_positive_and_grows_with_velocity():
    y = np.full(4, 0.5)
    u_p = np.array([0.01, 0.05, 0.1, 0.2])
    tau = wall_shear_stress(u_p, y, 1e-4)
    assert np.all(tau > 0.0)
    assert np.all(np.diff(tau) > 0.0)


def test_solver_handles_a_whole_field_of_wall_cells():
    u_p = np.linspace(0.0, 0.2, 50)
    out = friction_velocity(u_p, np.full(50, 0.5), 1e-4)
    assert out.shape == (50,)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0


# ---------------------------------------------------------------------------
# Effective viscosity
# ---------------------------------------------------------------------------

def test_model_is_inert_when_the_wall_is_resolved():
    """
    At y+ << 1 the law is u+ = y+, which is just the molecular stress, so the
    model must return the molecular viscosity and change nothing.
    """
    nu = 5e-2
    nu_w = effective_wall_viscosity(np.array([1e-6]), np.array([0.5]), nu)[0]
    assert nu_w == pytest.approx(nu, rel=1e-3)


def _requested_ratio(y_plus, nu=1e-2, y=0.5):
    """nu_w / nu the model asks for at a given y+, via an exact round trip."""
    y_plus = np.atleast_1d(np.asarray(y_plus, dtype=float))
    u_tau = y_plus * nu / y
    u_p = u_tau * reichardt_u_plus(y_plus)
    return effective_wall_viscosity(u_p, np.full(y_plus.size, y), nu) / nu


def test_model_stays_within_a_whisker_of_molecular_across_the_sublayer():
    """
    Through the whole resolved range the model must ask for essentially the
    molecular viscosity -- that is what makes it safe to leave switched on.

    The departure is not monotone in y+ and it is worth knowing why, because
    it looks like a bug otherwise.  Reichardt crosses the viscous law u+ = y+
    from below at y+ ~ 0.2: under it the model wants a hair *more* than
    molecular, peaking at +1.5e-4 near y+ = 0.1 and falling back to zero as
    y+ -> 0; above it the model wants slightly *less*, and the clamp in
    effective_wall_viscosity pins it to exactly molecular.  So the bound below
    is one-sided in practice and never exceeds a couple of parts in 10^4.
    """
    y_plus = np.logspace(-2, np.log10(2.0), 60)
    ratio = _requested_ratio(y_plus)
    assert np.all(np.abs(ratio - 1.0) < 2e-4), np.abs(ratio - 1.0).max()
    assert np.all(ratio >= 1.0)                  # the clamp holds everywhere


def test_the_model_does_nothing_at_all_through_the_buffer_layer():
    """Where Reichardt sits above u+ = y+, the clamp makes it an exact no-op."""
    np.testing.assert_array_equal(
        _requested_ratio(np.array([0.3, 0.5, 1.0, 2.0])), np.ones(4)
    )


def test_the_model_takes_over_once_the_first_cell_leaves_the_sublayer():
    """
    The flip side of being inert when resolved: it has to actually do
    something when it is needed.  By y+ = 30, the usual target for a wall-
    modelled LES, it is asking for viscosities a factor of ~2 up and rising.
    """
    ratio = _requested_ratio(np.array([5.0, 30.0, 200.0]))
    assert np.all(np.diff(ratio) > 0.0)
    assert ratio[1] > 2.0
    assert ratio[2] > 10.0


def test_model_raises_the_viscosity_once_out_of_the_sublayer():
    nu = 1.67e-4                      # omega ~ 1.998
    nu_w = effective_wall_viscosity(np.array([0.15]), np.array([0.5]), nu)[0]
    assert nu_w > 2.0 * nu


def test_effective_viscosity_never_drops_below_molecular():
    nu = 1e-3
    u_p = np.logspace(-6, -0.5, 40)
    nu_w = effective_wall_viscosity(u_p, np.full(u_p.size, 0.5), nu)
    assert np.all(nu_w >= nu - 1e-15)


# ---------------------------------------------------------------------------
# The wall mask
# ---------------------------------------------------------------------------

def test_wall_mask_finds_face_neighbours_only():
    solid = np.zeros((5, 5), dtype=bool)
    solid[2, 2] = True
    mask = wall_adjacent_mask(solid)
    assert mask[1, 2] and mask[3, 2] and mask[2, 1] and mask[2, 3]
    assert not mask[1, 1]              # diagonal is not a face neighbour
    assert not mask[2, 2]              # the solid cell itself is not fluid


def test_wall_mask_is_empty_without_geometry():
    assert not wall_adjacent_mask(np.zeros((4, 4), dtype=bool)).any()


def test_wall_mask_3d():
    solid = np.zeros((5, 5, 5), dtype=bool)
    solid[:, 0, :] = True
    mask = wall_adjacent_mask(solid)
    assert mask[:, 1, :].all()
    assert not mask[:, 2, :].any()


# ---------------------------------------------------------------------------
# Composition with the omega field
# ---------------------------------------------------------------------------

def test_apply_wall_model_only_touches_wall_cells():
    ny, nx = 6, 6
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    omega = np.full((ny, nx), 1.7)
    ux = np.full((ny, nx), 0.15)
    uy = np.zeros((ny, nx))
    out = apply_wall_model(omega, (ux, uy), solid, nu=1.67e-4)
    mask = wall_adjacent_mask(solid)
    assert not np.allclose(out[mask], 1.7)          # wall row was modelled
    np.testing.assert_allclose(out[~mask], omega[~mask])   # rest untouched
    assert omega[1, 0] == 1.7                        # input not modified


def test_apply_wall_model_without_geometry_is_a_no_op():
    omega = np.full((4, 4), 1.5)
    out = apply_wall_model(
        omega, (np.zeros((4, 4)), np.zeros((4, 4))),
        np.zeros((4, 4), dtype=bool), nu=1e-3,
    )
    np.testing.assert_array_equal(out, omega)


# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------

def _cylinder(ny, nx, r):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return ((yy - ny / 2) ** 2 + (xx - nx / 4) ** 2) <= r * r


def test_solver_runs_with_the_wall_model():
    ny, nx = 32, 64
    result = Solver(
        Ny=ny, Nx=nx, solid=_cylinder(ny, nx, 5.0), omega=1.9, u0=0.08,
        D=10.0, backend="numpy", les=True, wall_model=True,
    ).run(steps=40, check_every=10 ** 9, verbose=False)
    assert np.isfinite(result["Cd_mean"])


@pytest.mark.parametrize("omega", [0.6, 0.8, 1.2, 1.5])
def test_wall_model_barely_touches_a_well_resolved_run(omega):
    """
    A wall model that perturbs an already-resolved run is not usable.

    Every case here sits at y+ < 1 at the wall-adjacent cells, where the model
    asks for within 2e-4 of the molecular viscosity (see the unit tests
    above), so switching it on has to be close to a no-op.  Measured Cd shifts
    are 1.5e-5 (omega=0.6), 1.1e-5 (0.8), 1.1e-6 (1.2) and 1.7e-7 (1.5); one
    bound covers the sweep.

    A per-case tolerance *ladder* would be tempting and would not be measuring
    the model.  Those shifts span 500x while the perturbation the model
    applies spans only 7x (1.1e-4 down to 1.5e-5 in dOmega/Omega).  The rest
    is how many wall cells sit in the y+ ~ 0.1 crossover bump at all -- 94% at
    omega=0.6 against 25% at 1.5, the others being clamped to an exact no-op
    -- and how far their perturbation diffuses in a fixed 60-step window,
    sqrt(nu t) being 4.8 cells against 1.8.  Both are properties of the test
    setup, so pinning them as if they were the model's fidelity would be
    writing the transient into the suite.
    """
    ny, nx = 32, 64
    common = dict(
        Ny=ny, Nx=nx, solid=_cylinder(ny, nx, 5.0), omega=omega, u0=0.05,
        D=10.0, backend="numpy",
    )
    off = Solver(**common).run(steps=60, check_every=10 ** 9, verbose=False)
    on = Solver(wall_model=True, **common).run(steps=60, check_every=10 ** 9, verbose=False)
    assert on["Cd_mean"] == pytest.approx(off["Cd_mean"], rel=1e-4)


def test_wall_model_composes_with_les():
    """The model writes into the same omega field the subgrid model uses."""
    ny, nx = 32, 64
    common = dict(
        Ny=ny, Nx=nx, solid=_cylinder(ny, nx, 5.0), omega=1.99, u0=0.1,
        D=10.0, backend="numpy", les=True, les_cs=0.16,
    )
    plain = Solver(**common).run(steps=40, check_every=10 ** 9, verbose=False)
    modelled = Solver(wall_model=True, **common).run(
        steps=40, check_every=10 ** 9, verbose=False
    )
    assert np.isfinite(modelled["Cd_mean"])
    assert modelled["Cd_mean"] != plain["Cd_mean"]


def test_solver_3d_runs_with_the_wall_model():
    nz, ny, nx = 8, 12, 16
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    solid = ((zz - 4) ** 2 + (yy - 6) ** 2 + (xx - 4) ** 2) <= 4
    result = Solver3D(
        Nz=nz, Ny=ny, Nx=nx, solid=solid, omega=1.9, u0=0.08, D=4.0,
        backend="numpy", wall_model=True,
    ).run(steps=20, check_every=10 ** 9, verbose=False)
    assert np.isfinite(result["Cd_mean"])

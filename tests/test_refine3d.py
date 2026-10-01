"""Tests for local 2:1 refinement around the body (aero.lbm.refine3d)."""

import numpy as np
import pytest

from aero.geometry3d.box import Box
from aero.geometry3d.cylinder3d import Cylinder3D
from aero.geometry3d.sphere import Sphere
from aero.lbm import refine3d
from aero.lbm.lattice3d import D3Q19, compute_feq, compute_macroscopic
from aero.lbm.multiblock import _omega_fine
from aero.lbm.refine3d import (
    RESTRICT_WEIGHTS, RefinedSolver3D, _axis_stencil, fine_body_mask, refinement_box, restrict_block,
)


def _small(body=True, **kw):
    """A 24 x 16 x 16 tunnel with a 10 x 8 x 8 block; a small cube in it, or nothing."""
    nz, ny, nx = 16, 16, 24
    box = (4, 12, 4, 12, 6, 16)
    solid = np.zeros((nz, ny, nx), bool)
    fine = np.zeros((16, 16, 20), bool)
    if body:
        solid[7:9, 7:9, 9:11] = True
        fine[6:10, 6:10, 6:10] = True
    args = dict(omega=1.5, u0=0.03, D=2.0, backend="numpy", collision="bgk")
    args.update(kw)
    return RefinedSolver3D(nz, ny, nx, solid, box, fine, **args)


def test_fine_cells_sit_a_quarter_cell_either_side_of_each_coarse_centre():
    idx, w = _axis_stencil(3)
    loc = 1.75 + 0.5 * np.arange(-1, 7)            # fine j = -1 .. 6, from the coarse cell a - 2
    assert idx.shape == w.shape == (8, 4)
    # cubic stencils: exact on every polynomial up to degree three
    for deg in range(4):
        np.testing.assert_allclose((w * idx.astype(float) ** deg).sum(axis=1), loc ** deg, atol=1e-12)
    assert idx.min() == 0 and idx.max() == 3 + 3    # inside the block and its two-cell halo


def test_restriction_weights_are_fourth_order():
    """Exact on cubics about the coarse centre: the plain mean is off by h^2 f''/32."""
    x = np.array([-0.75, -0.25, 0.25, 0.75])
    for deg in range(4):
        assert (RESTRICT_WEIGHTS * x ** deg).sum() == pytest.approx(0.0 ** deg, abs=1e-15)


@pytest.mark.parametrize("numba", [False, True])
def test_ghost_layer_is_exact_on_a_cubic_field(numba):
    """Tricubic interpolation reproduces any cubic field exactly, faces, edges and corners."""
    if numba and not refine3d._HAS_NUMBA:
        pytest.skip("numba not installed")
    s = _small()
    s._use_numba = numba
    nz, ny, nx = 16, 16, 24
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    field = lambda Z, Y, X: 1.0 + 1e-3 * (X - 10.0) ** 3 - 2e-3 * (Y - 7.0) ** 2 * (Z - 6.0) + 0.005 * Z
    s._coarse.f[:] = field(z, y, x)[None] * (1.0 + 0.1 * np.arange(19))[:, None, None, None]
    faces = s._shell()
    z0, z1, y0, y1, x0, x1 = s.box
    fz = z0 - 0.25 + 0.5 * np.arange(-1, 2 * (z1 - z0) + 1)
    fy = y0 - 0.25 + 0.5 * np.arange(-1, 2 * (y1 - y0) + 1)
    fx = x0 - 0.25 + 0.5 * np.arange(-1, 2 * (x1 - x0) + 1)
    expect = [field(fz[:, None], fy[None, :], fx[0]), field(fz[:, None], fy[None, :], fx[-1]),
              field(fz[:, None], fy[0], fx[None, :]), field(fz[:, None], fy[-1], fx[None, :]),
              field(fz[0], fy[:, None], fx[None, :]), field(fz[-1], fy[:, None], fx[None, :])]
    for face, e in zip(faces, expect):
        np.testing.assert_allclose(face[0], e, atol=1e-12)
        np.testing.assert_allclose(face[5], 1.5 * e, atol=1e-12)


def test_restriction_kernel_matches_the_reference():
    if not refine3d._HAS_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(1)
    s = _small()
    shape = s._fine.f.shape[1:]
    rho = 1.0 + 0.01 * rng.standard_normal(shape)
    u = [0.03 * rng.standard_normal(shape) for _ in range(3)]
    s._fine.f[:] = compute_feq(rho, *u, D3Q19) + 1e-4 * rng.standard_normal(s._fine.f.shape)
    expect = restrict_block(s._fine.f, s._f2c, D3Q19)
    s._use_numba = True
    s._restrict()
    z0, z1, y0, y1, x0, x1 = s.box
    np.testing.assert_allclose(s._coarse.f[:, z0:z1, y0:y1, x0:x1], expect, atol=1e-14)


def test_restriction_keeps_the_moments_and_rescales_the_rest():
    rng = np.random.default_rng(2)
    f = compute_feq(np.ones((6, 6, 6)), np.full((6, 6, 6), 0.04), np.zeros((6, 6, 6)),
                    np.zeros((6, 6, 6)), D3Q19) + 1e-4 * rng.standard_normal((19, 6, 6, 6))
    coarse = restrict_block(f, 2.0, D3Q19)
    w = RESTRICT_WEIGHTS
    filt = np.einsum("a,b,c,qabc->q", w, w, w, f[:, 0:4, 0:4, 0:4])  # coarse cell (0, 0, 0)
    for a, b in zip(compute_macroscopic(coarse[:, :1, :1, :1], D3Q19),
                    compute_macroscopic(filt[:, None, None, None], D3Q19)):
        np.testing.assert_allclose(a, b, atol=1e-14)
    feq = compute_feq(*compute_macroscopic(filt[:, None, None, None], D3Q19), D3Q19)[:, 0, 0, 0]
    np.testing.assert_allclose(coarse[:, 0, 0, 0] - feq, 2.0 * (filt - feq), atol=1e-14)


def test_a_uniform_stream_stays_uniform_through_the_interface():
    """With nothing in the block, the interface itself must not disturb the free stream."""
    s = _small(body=False)
    s.run(30, verbose=False)
    rho, ux, uy, uz = s.macroscopic()
    np.testing.assert_allclose(ux, 0.03, atol=1e-12)
    np.testing.assert_allclose(uy, 0.0, atol=1e-12)
    np.testing.assert_allclose(rho, 1.0, atol=1e-12)
    frho, fux, _, _ = s.fine_macroscopic()
    np.testing.assert_allclose(fux, 0.03, atol=1e-12)


def test_the_fine_block_runs_at_the_same_reynolds_number():
    s = _small()
    assert s.omega_fine == pytest.approx(_omega_fine(1.5))
    nu_c = (1.0 / s.omega - 0.5) / 3.0
    nu_f = (1.0 / s.omega_fine - 0.5) / 3.0
    assert s.u0 * (2 * s.D) / nu_f == pytest.approx(s.u0 * s.D / nu_c)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_shear_wave_decays_at_the_right_rate_across_the_block(backend):
    """
    A decaying shear wave across a fully periodic box, half of it inside
    the refined block, with the block's faces on the wave's crests: what
    reaches the coarse grid must decay as the plain solver's does (0.4% off
    the analytic rate here), not be damped by the interface.  Trilinear
    ghosts and a plain mean took 3.4% off it.
    """
    nz, ny, nx = 12, 32, 12
    amp, omega, steps = 0.02, 1.4, 200
    k = 2.0 * np.pi / ny
    if backend == "numba" and not refine3d._HAS_NUMBA:
        pytest.skip("numba not installed")
    kw = dict(omega=omega, u0=0.0, D=4.0, backend=backend, collision="bgk",
              wall_bc="none", streamwise_bc="periodic")
    s = RefinedSolver3D(nz, ny, nx, np.zeros((nz, ny, nx), bool), (3, 9, 8, 24, 3, 9),
                        np.zeros((12, 32, 12), bool), **kw)
    y = np.arange(ny, dtype=float)
    ux = np.broadcast_to((amp * np.sin(k * y))[None, :, None], (nz, ny, nx)).copy()
    zero = np.zeros_like(ux)
    s._coarse.f[:] = compute_feq(np.ones_like(ux), ux, zero, zero, D3Q19)
    s.prolong()
    for _ in range(steps):
        s.step()
    nu = (1.0 / omega - 0.5) / 3.0
    analytic = amp * np.exp(-nu * k * k * steps)
    measured = np.abs(np.fft.rfft(s.macroscopic()[1].mean(axis=(0, 2)))[1]) * 2.0 / ny
    assert measured == pytest.approx(analytic, rel=0.008)


def test_the_box_grows_around_the_body_and_stays_off_the_walls():
    solid = Sphere(radius=4).mark_solid(40, 40, 80)
    z0, z1, y0, y1, x0, x1 = refinement_box(solid, 8.0)
    zs, ys, xs = np.nonzero(solid)
    assert x0 <= xs.min() - 4 and x1 >= xs.max() + 1 + 12          # half a body ahead, 1.5 behind
    assert y0 <= ys.min() - 4 and y1 >= ys.max() + 1 + 4
    assert min(z0, y0, x0) >= refine3d.BOX_GAP and z1 <= 40 - refine3d.BOX_GAP


def test_a_body_against_the_wall_cannot_be_refined():
    solid = np.zeros((20, 20, 40), bool)
    solid[8:12, 0:4, 10:14] = True
    with pytest.raises(ValueError, match="too close to the tunnel"):
        refinement_box(solid, 4.0)


@pytest.mark.parametrize("geom,volume", [(Sphere(radius=5), 4.0 / 3.0 * np.pi * 125),
                                         (Box(width=6, height=8, depth=5), 240.0),
                                         (Cylinder3D(radius=4, length=10), np.pi * 16 * 10)])
def test_the_fine_body_is_the_same_body_at_twice_the_resolution(geom, volume):
    shape = (32, 32, 60)
    coarse = geom.mark_solid(*shape)
    box = refinement_box(coarse, 8.0)
    fine = fine_body_mask(geom, shape, box)
    # the body's own volume (in coarse cells) -- closer than the coarse
    # staircase gets, which counts a box face on a row of cell centres in
    # full -- and the same place
    assert fine.sum() / 8.0 == pytest.approx(volume, rel=0.05)
    assert abs(fine.sum() / 8.0 - volume) <= abs(coarse.sum() - volume) + 1.0
    z0, _, y0, _, x0, _ = box
    fz, fy, fx = np.nonzero(fine)
    cz, cy, cx = np.nonzero(coarse)
    centre_fine = np.array([z0 - 0.25 + 0.5 * fz.mean(), y0 - 0.25 + 0.5 * fy.mean(), x0 - 0.25 + 0.5 * fx.mean()])
    np.testing.assert_allclose(centre_fine, [cz.mean(), cy.mean(), cx.mean()], atol=0.3)


def test_a_mesh_is_voxelized_again_on_the_fine_grid():
    from aero.geometry3d.mesh_mask import MeshMask
    # an octahedron: thin tips that a finer grid resolves better
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    geom = MeshMask(triangles=v[np.array(faces)], fit_frac=0.4, mesh_orient="none")
    shape = (30, 30, 60)
    coarse = geom.mark_solid(*shape)
    box = refinement_box(coarse, 12.0)
    fine = fine_body_mask(geom, shape, box)
    assert fine.sum() / 8.0 == pytest.approx(coarse.sum(), rel=0.15)


def test_unsupported_options_are_refused():
    with pytest.raises(ValueError, match="does not support"):
        _small(ibm_enabled=True)


def test_part_of_the_body_outside_the_block_is_refused():
    nz, ny, nx = 16, 16, 24
    solid = np.zeros((nz, ny, nx), bool)
    solid[7:9, 7:9, 2:4] = True                               # upstream of the block
    with pytest.raises(ValueError, match="outside the refined block"):
        RefinedSolver3D(nz, ny, nx, solid, (4, 12, 4, 12, 6, 16), np.zeros((16, 16, 20), bool),
                        omega=1.5, u0=0.03, D=2.0, backend="numpy")


def test_the_whole_block_is_restricted_whenever_the_field_is_read():
    if not refine3d._HAS_NUMBA:
        pytest.skip("numba not installed")
    s = _small(body=False, backend="numba")
    for _ in range(3):                              # not a multiple of FULL_RESTRICT_EVERY
        s.step()
    z0, z1, y0, y1, x0, x1 = s.box
    s._coarse.f[:, z0 + 3, y0 + 3, x0 + 3] = -1.0   # a deep cell only a full restriction rewrites
    rho = s.macroscopic()[0]
    assert rho[z0 + 3, y0 + 3, x0 + 3] == pytest.approx(1.0)


def test_forces_come_from_the_fine_grid_and_are_steady():
    """A small cube in the block: a finite, positive, settling drag."""
    s = _small()
    s.run(120, verbose=False)
    cd = np.asarray(s.Cd_history)
    assert np.all(np.isfinite(cd))
    assert cd[-20:].mean() > 0.0
    # the run reports the coefficients on the coarse frontal area (2 x 2 cells)
    assert s.ref_area == pytest.approx(4.0)


@pytest.mark.slow
def test_refining_near_the_body_recovers_the_fine_grid_drag():
    """
    A small sphere at Re = 20: the tunnel grid alone, the same with the box
    around the body refined, and the whole tunnel refined.  Refining near the
    body must land much closer to the all-fine answer than the tunnel grid
    does (at Re = 100 on the default sphere: +6.4% plain, +0.3% refined).
    """
    pytest.importorskip("numba")
    from aero.lbm.solver3d import Solver3D

    r, (nz, ny, nx), re, u0, steps = 3.0, (24, 24, 48), 20.0, 0.05, 1500

    def omega(d):
        return 1.0 / (3.0 * u0 * d / re + 0.5)

    def mean_cd(s, n):
        s.run(steps=n, check_every=10 ** 9, verbose=False)
        return float(np.mean(s.Cd_history[n // 2:]))

    geom = Sphere(radius=r)
    solid = geom.mark_solid(nz, ny, nx)
    coarse = mean_cd(Solver3D(Nz=nz, Ny=ny, Nx=nx, solid=solid, omega=omega(2 * r), u0=u0, D=2 * r,
                              ref_area=geom.reference_area()), steps)
    box = refinement_box(solid, 2 * r)
    refined = mean_cd(RefinedSolver3D(nz, ny, nx, solid, box, fine_body_mask(geom, (nz, ny, nx), box),
                                      omega=omega(2 * r), u0=u0, D=2 * r, ref_area=geom.reference_area()),
                      steps)
    big = Sphere(radius=2 * r)
    fine = mean_cd(Solver3D(Nz=2 * nz, Ny=2 * ny, Nx=2 * nx, solid=big.mark_solid(2 * nz, 2 * ny, 2 * nx),
                            omega=omega(4 * r), u0=u0, D=4 * r, ref_area=big.reference_area()), 2 * steps)
    assert abs(refined - fine) < abs(coarse - fine) / 3.0, (coarse, refined, fine)

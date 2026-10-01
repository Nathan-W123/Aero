"""
Local 2:1 grid refinement around the body, for the 3D tunnel.

The tunnel runs on the coarse grid.  A box around the body -- a little
upstream and to the sides of it, a diameter or two into the wake -- runs at
twice the resolution in every direction, two fine steps to each coarse one.
Under that (acoustic) scaling the lattice velocity is the same on both grids
and the lattice viscosity doubles, so both run at the same Reynolds number
(see :func:`aero.lbm.multiblock._omega_fine`).  The body is voxelized afresh
on the fine grid, so its surface is a staircase of half-size steps, and the
forces are taken there.

What it buys: the cells that decide the drag -- the boundary layer, the
separation, the near wake -- at twice the resolution, for a fraction of
refining everything.  Refining the whole tunnel twice over takes 16x the
cell updates (8x the cells, 2x the steps) and, the arrays being larger,
20-30x the time; a box a few diameters long costs what its cells do, about
5-7x in a crowded tunnel and much less in a roomy one.

Coupling
--------
An overlapping-block scheme in the manner of Dupuis & Chopard (2003), with
the non-equilibrium rescaling of Filippova & Haenel (1998):

coarse -> fine
    The fine block has one ghost layer of fine cells around it, lying outside
    the block in the inner half of the first coarse cell beyond each face.
    Before each fine step the ghost layer is set from the coarse grid:
    tricubic in space, linear in time (the second fine step starts half-way
    through the coarse one), and with the non-equilibrium part rescaled by
    ``tau_f / (2 tau_c)``.  The fine collision and streaming carry it in.

fine -> coarse
    After both fine steps, every coarse cell under the block becomes the
    mean of its eight fine cells corrected to fourth order by the ring
    around them, non-equilibrium rescaled back.  The coarse cells around the
    block then stream in what the fine grid computed, and the coarse field
    under the block -- what the viewer and figures show -- is the fine
    solution.

Why the non-equilibrium part needs rescaling at all: density and velocity
are the same on both grids, so the equilibrium carries over as it is, but
``f_neq`` is proportional to ``tau`` times the velocity gradient *per cell*,
and that halves when the cell does.  Copying ``f`` across unchanged gets the
viscous stress at the interface wrong by that factor.

Validation
----------
A sphere at Re = 100 in the web UI's default tunnel (96 x 48 x 48, the
sphere 14 cells across, regularized collision, 3000 steps, Cd averaged over
the second half), against the same tunnel refined everywhere:

    ==========================================  =======  =============
    grid                                         Cd       vs. all-fine
    ==========================================  =======  =============
    tunnel grid alone, 14 cells across           1.3645   +6.4%
    refined near the body, 28 across in the box  1.2860   +0.3%
    whole tunnel refined, 28 across (192x96x96)  1.2820
    ==========================================  =======  =============

The box is 43 x 29 x 29 tunnel cells, so a step updates the tunnel's cells
and, twice, the box's eight-fold ones: 3.9x the plain grid's cell updates,
against 16x for refining everything.  Time per step on four cores:

    ===========  =========  ==================  ==================
    collision    plain      refined near body   whole tunnel
    ===========  =========  ==================  ==================
    regularized  15.1 ms    82 ms (5.4x)        307 ms (20x)
    BGK           8.6 ms    60 ms (7.0x)        262 ms (31x)
    ===========  =========  ==================  ==================

A fine cell update costs about 1.5 tunnel ones (:data:`FINE_UPDATE_COST`):
the fine grid takes the forces over the body's surface, at four times the
links, and drives the interface.  A decaying shear wave whose crests sit on
the block's faces decays at the plain solver's rate (0.4% off the analytic
one); see the transfer operators below for why that took fourth-order
transfers.

Limits
------
Two levels only, and the box must clear the tunnel's walls, inlet and outlet
by three coarse cells.  The scheme is not exactly mass-conserving across the
interface; the inlet and outlet absorb the difference.  Body forces, the
immersed boundary, interpolated bounce-back and the thermal model are not
coupled across the interface and are refused.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lattice3d import compute_feq, compute_macroscopic
from .multiblock import REFINEMENT, _omega_fine, neq_scale_coarse_to_fine

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:                                        # pragma: no cover
    _HAS_NUMBA = False

#: ``(z0, z1, y0, y1, x0, x1)``: the refined box in coarse cells, half-open.
Box = Tuple[int, int, int, int, int, int]

#: Coarse cells kept between the box and the tunnel's boundary planes: the
#: interpolation reads two cells beyond each face, and neither may be one a
#: boundary condition writes.
BOX_GAP = 3

#: Default margins around the body, in body lengths: the boundary layer and
#: separation need the sides and a little of the front; the near wake --
#: about one diameter of recirculation behind a sphere at Re = 100, 1.4 at
#: 200 -- needs more behind.
MARGIN_UPSTREAM = 0.5
MARGIN_LATERAL = 0.5
MARGIN_DOWNSTREAM = 1.5

#: Time per fine cell update, in tunnel cell updates: the fine grid also
#: carries the body -- four times the surface links to take forces over --
#: and the interface.  Measured; see Validation above.
FINE_UPDATE_COST = 1.5


# ---------------------------------------------------------------------------
# Where the block goes, and the body on it
# ---------------------------------------------------------------------------

def refinement_box(
    solid: np.ndarray,
    D: float,
    *,
    upstream: float = MARGIN_UPSTREAM,
    lateral: float = MARGIN_LATERAL,
    downstream: float = MARGIN_DOWNSTREAM,
) -> Box:
    """
    The box to refine: the body's bounding box grown by the margins (in body
    lengths ``D``), clipped to the tunnel less :data:`BOX_GAP` cells.

    Raises ValueError when the body sits too close to the tunnel's walls,
    inlet or outlet for a block to fit around it.
    """
    solid = np.asarray(solid, dtype=bool)
    if not solid.any():
        raise ValueError("There is no body to refine around.")
    nz, ny, nx = solid.shape
    lo = [int(np.nonzero(solid.any(axis=tuple(a for a in range(3) if a != ax)))[0].min()) for ax in range(3)]
    hi = [int(np.nonzero(solid.any(axis=tuple(a for a in range(3) if a != ax)))[0].max()) + 1 for ax in range(3)]
    d = max(float(D), 1.0)
    grow_lo = [lateral, lateral, upstream]
    grow_hi = [lateral, lateral, downstream]
    box = []
    for ax, n in enumerate((nz, ny, nx)):
        a = max(lo[ax] - int(np.ceil(grow_lo[ax] * d)), BOX_GAP)
        b = min(hi[ax] + int(np.ceil(grow_hi[ax] * d)), n - BOX_GAP)
        # the body must clear the box's faces by a coarse cell, so the
        # interpolation stencil at the interface is all fluid
        if lo[ax] - a < 1 or b - hi[ax] < 1:
            name = "zyx"[ax]
            raise ValueError(
                f"The body is too close to the tunnel's {'walls' if name != 'x' else 'inlet or outlet'} "
                f"to refine around: it needs {BOX_GAP + 1} or more clear cells along {name}.")
        box += [a, b]
    return tuple(box)  # type: ignore[return-value]


def refined_cost(coarse_shape: Sequence[int], box: Box) -> float:
    """
    The time a step takes with the block, in steps of the tunnel without it:
    the tunnel, and two steps of the block, ghost layer included, at
    :data:`FINE_UPDATE_COST` a cell.  An estimate from the cell counts.
    """
    z0, z1, y0, y1, x0, x1 = box
    fine = (2 * (z1 - z0) + 2) * (2 * (y1 - y0) + 2) * (2 * (x1 - x0) + 2)
    return 1.0 + REFINEMENT * FINE_UPDATE_COST * fine / float(np.prod(coarse_shape))


def fine_body_mask(geom: Any, coarse_shape: Sequence[int], box: Box) -> np.ndarray:
    """
    The body voxelized on the fine grid of ``box``, shape ``2 x`` the box.

    Fine cells sit a quarter of a coarse cell either side of each coarse cell
    centre.  The analytic shapes are evaluated there directly; a mesh is
    voxelized again at twice the scale.  Anything else falls back to the
    coarse mask with each cell split in eight, which keeps the coarse
    staircase but is still correct.
    """
    from ..geometry3d.box import Box as BoxShape
    from ..geometry3d.cylinder3d import Cylinder3D
    from ..geometry3d.mesh_mask import MeshMask, voxelize_mesh
    from ..geometry3d.sphere import Sphere

    Nz, Ny, Nx = (int(n) for n in coarse_shape)
    z0, z1, y0, y1, x0, x1 = box
    if isinstance(geom, MeshMask):
        # mesh coordinates put cell i on [i, i+1]: doubling them puts the
        # fine cells 2i and 2i+1 exactly on its two halves
        tris, _ = geom._transform_to_grid(Nz, Ny, Nx)
        tris = 2.0 * np.asarray(tris, dtype=np.float64) - 2.0 * np.array([x0, y0, z0], dtype=np.float64)
        return voxelize_mesh(tris, 2 * (z1 - z0), 2 * (y1 - y0), 2 * (x1 - x0))

    # the analytic shapes put cell i's centre at i
    z = (z0 - 0.25 + 0.5 * np.arange(2 * (z1 - z0)))[:, None, None]
    y = (y0 - 0.25 + 0.5 * np.arange(2 * (y1 - y0)))[None, :, None]
    x = (x0 - 0.25 + 0.5 * np.arange(2 * (x1 - x0)))[None, None, :]
    if isinstance(geom, (Sphere, BoxShape, Cylinder3D)):
        cx, cy, cz = geom.center(Nz, Ny, Nx)
        if isinstance(geom, Sphere):
            return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= geom.radius ** 2
        if isinstance(geom, BoxShape):
            return ((np.abs(x - cx) <= geom.width / 2.0) & (np.abs(y - cy) <= geom.height / 2.0)
                    & (np.abs(z - cz) <= geom.depth / 2.0))
        return ((x - cx) ** 2 + (y - cy) ** 2 <= geom.radius ** 2) & (np.abs(z - cz) <= geom.length / 2.0)

    coarse = np.asarray(geom.mark_solid(Nz, Ny, Nx), dtype=bool)[z0:z1, y0:y1, x0:x1]
    return np.repeat(np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1), 2, axis=2)


# ---------------------------------------------------------------------------
# Transfer operators
# ---------------------------------------------------------------------------
#
# Both directions are fourth-order accurate on smooth data, not second.  The
# simple choices -- trilinear interpolation onto the ghosts, the plain mean
# of the eight fine cells -- are each a slight low-pass filter, and the
# interface applies them every step: on a decaying shear wave whose crests
# sat on the block's faces they took 3% off its amplitude in 200 steps, with
# the error growing with the size of the block.  Cubic interpolation and a
# corrected mean bring that to the 0.4% the plain solver itself makes.

#: Coarse layers inside each face of the block restricted every step -- the
#: ones the coupling reads -- and how often the whole block is.
RESTRICT_BAND = 2
FULL_RESTRICT_EVERY = 16

#: Weights of the fourth-order restriction along one axis, over the fine
#: cells at -3/4, -1/4, +1/4, +3/4 of a coarse cell from its centre: the mean
#: of the two inside, less an eighth of the curvature the outer two measure.
RESTRICT_WEIGHTS = np.array([-1.0, 9.0, 9.0, -1.0]) / 16.0


def _axis_stencil(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cubic (four-point Lagrange) stencils for the fine cells ``j = -1 .. 2n``
    along an axis the block spans ``n`` coarse cells of, the two ghosts
    included: ``(index, weight)``, each ``(2n + 2, 4)``.

    Indices are into the block's coarse cells *with a two-cell halo*, so the
    coarse cell two before the block is 0.  Fine cell ``j`` sits at coarse
    coordinate ``a - 1/4 + j/2``: ``7/4 + j/2`` from there.
    """
    j = np.arange(-1, 2 * n + 1)
    loc = 1.75 + 0.5 * j
    base = np.floor(loc).astype(np.intp)
    t = loc - base
    idx = np.stack([base - 1, base, base + 1, base + 2], axis=1)
    w = np.stack([-t * (t - 1.0) * (t - 2.0) / 6.0, (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
                  -(t + 1.0) * t * (t - 2.0) / 2.0, (t + 1.0) * t * (t - 1.0) / 6.0], axis=1)
    return idx, w


def _interp(a: np.ndarray, axis: int, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Interpolate ``a`` along ``axis`` with per-point stencils ``(idx, w)``, each (M, P)."""
    shape = [1] * a.ndim
    shape[axis] = -1
    out = np.take(a, idx[:, 0], axis=axis) * w[:, 0].reshape(shape)
    for p in range(1, idx.shape[1]):
        out += np.take(a, idx[:, p], axis=axis) * w[:, p].reshape(shape)
    return out


def _rescale_neq(f: np.ndarray, factor: float, lattice, use_numba: bool = _HAS_NUMBA) -> np.ndarray:
    """``feq + factor * (f - feq)``: the moments kept, the non-equilibrium part scaled."""
    if use_numba and _HAS_NUMBA:
        out = np.array(f, dtype=np.float64, order="C")
        lat = lattice
        _rescale_kernel(out.reshape(out.shape[0], -1), factor, lat.ex.astype(np.float64),
                        lat.ey.astype(np.float64), lat.ez.astype(np.float64), lat.W, lat.h3_factor)
        return out
    squeeze = f.ndim == 3
    g = f[:, None] if squeeze else f
    feq = compute_feq(*compute_macroscopic(g, lattice), lattice)
    out = feq + factor * (g - feq)
    return out[:, 0] if squeeze else out


if _HAS_NUMBA:
    @nb.njit(cache=True)
    def _rescale_kernel(f, scale, ex, ey, ez, w, h3):                       # pragma: no cover
        """In place on ``(Q, N)``: each column's non-equilibrium part times ``scale``."""
        Q, N = f.shape
        for n in range(N):
            rho = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in range(Q):
                s = f[q, n]
                rho += s
                mx += ex[q] * s
                my += ey[q] * s
                mz += ez[q] * s
            inv = 1.0 / rho if rho > 0.0 else 0.0
            ux = mx * inv
            uy = my * inv
            uz = mz * inv
            usq = ux * ux + uy * uy + uz * uz
            for q in range(Q):
                eu = ex[q] * ux + ey[q] * uy + ez[q] * uz
                feq = w[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq
                                    + h3 * 4.5 * (eu * eu * eu - eu * usq))
                f[q, n] = feq + scale * (f[q, n] - feq)

    @nb.njit(cache=True, parallel=True)
    def _face_kernel(h, axis, ie, we, ia, wa, ib, wb, out):               # pragma: no cover
        """
        One ghost face, tricubic, into ``out`` (Q, Ma, Mb): across ``axis`` of
        the halo ``h`` (1 z, 2 y, 3 x) with the four-point stencil ``(ie, we)``,
        then along the face's own two axes, in (z, y, x) order, with
        ``(ia, wa)`` and ``(ib, wb)``.  One population per thread.
        """
        Q = h.shape[0]
        if axis == 3:
            Ha, Hb = h.shape[1], h.shape[2]
        elif axis == 2:
            Ha, Hb = h.shape[1], h.shape[3]
        else:
            Ha, Hb = h.shape[2], h.shape[3]
        Ma, Mb = ia.shape[0], ib.shape[0]
        for q in nb.prange(Q):
            plane = np.empty((Ha, Hb))
            if axis == 3:
                for i in range(Ha):
                    for j in range(Hb):
                        plane[i, j] = (we[0] * h[q, i, j, ie[0]] + we[1] * h[q, i, j, ie[1]]
                                       + we[2] * h[q, i, j, ie[2]] + we[3] * h[q, i, j, ie[3]])
            elif axis == 2:
                for i in range(Ha):
                    for j in range(Hb):
                        plane[i, j] = (we[0] * h[q, i, ie[0], j] + we[1] * h[q, i, ie[1], j]
                                       + we[2] * h[q, i, ie[2], j] + we[3] * h[q, i, ie[3], j])
            else:
                for i in range(Ha):
                    for j in range(Hb):
                        plane[i, j] = (we[0] * h[q, ie[0], i, j] + we[1] * h[q, ie[1], i, j]
                                       + we[2] * h[q, ie[2], i, j] + we[3] * h[q, ie[3], i, j])
            half = np.empty((Ha, Mb))
            for i in range(Ha):
                for m in range(Mb):
                    half[i, m] = (wb[m, 0] * plane[i, ib[m, 0]] + wb[m, 1] * plane[i, ib[m, 1]]
                                  + wb[m, 2] * plane[i, ib[m, 2]] + wb[m, 3] * plane[i, ib[m, 3]])
            for m in range(Ma):
                r0, r1, r2, r3 = ia[m, 0], ia[m, 1], ia[m, 2], ia[m, 3]
                w0, w1, w2, w3 = wa[m, 0], wa[m, 1], wa[m, 2], wa[m, 3]
                for j in range(Mb):
                    out[q, m, j] = w0 * half[r0, j] + w1 * half[r1, j] + w2 * half[r2, j] + w3 * half[r3, j]

    @nb.njit(cache=True, parallel=True)
    def _write_face(f, axis, pos, A, B, wb, scale, ex, ey, ez, w, h3):      # pragma: no cover
        """
        One ghost face into ``f``: ``(1 - wb) A + wb B`` (two time levels),
        its non-equilibrium part scaled by ``scale``, at index ``pos`` along
        ``axis`` (0 z, 1 y, 2 x); the face's own axes follow in (z, y, x) order.
        """
        Q, M1, M2 = A.shape
        wa = 1.0 - wb
        for a in nb.prange(M1):
            tmp = np.empty(Q)
            for b in range(M2):
                rho = 0.0
                mx = 0.0
                my = 0.0
                mz = 0.0
                for q in range(Q):
                    v = wa * A[q, a, b] + wb * B[q, a, b]
                    tmp[q] = v
                    rho += v
                    mx += ex[q] * v
                    my += ey[q] * v
                    mz += ez[q] * v
                inv = 1.0 / rho if rho > 0.0 else 0.0
                ux = mx * inv
                uy = my * inv
                uz = mz * inv
                usq = ux * ux + uy * uy + uz * uz
                for q in range(Q):
                    eu = ex[q] * ux + ey[q] * uy + ez[q] * uz
                    feq = w[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq
                                        + h3 * 4.5 * (eu * eu * eu - eu * usq))
                    val = feq + scale * (tmp[q] - feq)
                    if axis == 2:
                        f[q, a, b, pos] = val
                    elif axis == 1:
                        f[q, a, pos, b] = val
                    else:
                        f[q, pos, a, b] = val

    @nb.njit(cache=True, parallel=True)
    def _restrict_kernel(ff, fc, z0, y0, x0, rw, ex, ey, ez, w, scale, h3, band):  # pragma: no cover
        """
        The fourth-order restriction of a block, non-equilibrium rescaled, in
        one pass: every cell, or with ``band`` > 0 only the cells that many
        layers in from the block's faces.
        """
        Q = ff.shape[0]
        nz = (ff.shape[1] - 2) // 2
        ny = (ff.shape[2] - 2) // 2
        nx = (ff.shape[3] - 2) // 2
        for k in nb.prange(nz):
            avg = np.empty(Q)
            edge_k = band <= 0 or k < band or k >= nz - band
            for j in range(ny):
                edge_kj = edge_k or j < band or j >= ny - band
                for i in range(nx):
                    if not (edge_kj or i < band or i >= nx - band):
                        continue
                    rho = 0.0
                    mx = 0.0
                    my = 0.0
                    mz = 0.0
                    for q in range(Q):
                        s = 0.0
                        for dz in range(4):
                            sy = 0.0
                            for dy in range(4):
                                sx = 0.0
                                for dx in range(4):
                                    sx += rw[dx] * ff[q, 2 * k + dz, 2 * j + dy, 2 * i + dx]
                                sy += rw[dy] * sx
                            s += rw[dz] * sy
                        avg[q] = s
                        rho += s
                        mx += ex[q] * s
                        my += ey[q] * s
                        mz += ez[q] * s
                    inv = 1.0 / rho if rho > 0.0 else 0.0
                    ux = mx * inv
                    uy = my * inv
                    uz = mz * inv
                    usq = ux * ux + uy * uy + uz * uz
                    for q in range(Q):
                        eu = ex[q] * ux + ey[q] * uy + ez[q] * uz
                        feq = w[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq
                                            + h3 * 4.5 * (eu * eu * eu - eu * usq))
                        fc[q, z0 + k, y0 + j, x0 + i] = feq + scale * (avg[q] - feq)


def restrict_block(f_fine: np.ndarray, factor: float, lattice) -> np.ndarray:
    """
    The coarse cells under a fine block, from ``f_fine`` with its ghost layer:
    the fourth-order weighted mean of the 4 x 4 x 4 fine cells around each
    coarse centre (its own eight and the ring around them), non-equilibrium
    scaled by ``factor``.  The ghost layer must hold valid values.
    """
    q = f_fine.shape[0]
    n = [(m - 2) // 2 for m in f_fine.shape[1:]]
    out = f_fine
    for axis in (3, 2, 1):
        cnt = n[axis - 1]
        acc = 0.0
        for p in range(4):
            sl = [slice(None)] * 4
            sl[axis] = slice(p, p + 2 * cnt, 2)
            acc = acc + out[tuple(sl)] * RESTRICT_WEIGHTS[p]
        out = acc
    return _rescale_neq(np.ascontiguousarray(out).reshape(q, *n), factor, lattice)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

#: Solver3D options the refined block cannot couple across its interface.
_UNSUPPORTED = ("ibm_enabled", "bouzidi", "thermal", "buoyancy", "wall_model", "sem_inlet",
                "synthetic_inflow")
#: Options the fine block shares with the coarse grid.  Everything else --
#: inlet, outlet, walls, sponge, inlet noise -- belongs to the tunnel.
_SHARED = ("collision", "lattice", "backend", "trt_lambda", "les", "les_cs", "les_model",
           "van_driest", "van_driest_A")


class RefinedSolver3D:
    """
    The 3D tunnel with a 2x finer block around the body.

    Parameters
    ----------
    Nz, Ny, Nx  : the tunnel, in coarse cells
    solid       : the body on the coarse grid, (Nz, Ny, Nx); used for display
                  and checks -- the fine grid carries the body that counts
    box         : the refined block (see :func:`refinement_box`)
    solid_fine  : the body on the fine grid, ``2 x`` the box's shape (see
                  :func:`fine_body_mask`)
    omega, u0, D: the coarse grid's relaxation rate, inlet speed and the
                  body's length in coarse cells
    ref_area    : frontal area for the coefficients, in coarse cells^2; None
                  measures it from the fine body
    **kw        : the rest of Solver3D's options
    """

    def __init__(
        self,
        Nz: int,
        Ny: int,
        Nx: int,
        solid: np.ndarray,
        box: Box,
        solid_fine: np.ndarray,
        omega: float,
        u0: float,
        D: float,
        ref_area: Optional[float] = None,
        rho0: float = 1.0,
        **kw: Any,
    ) -> None:
        from .solver3d import Solver3D

        refused = [k for k in _UNSUPPORTED if kw.get(k)]
        if refused:
            raise ValueError(f"Local refinement does not support {', '.join(refused)}.")
        self.Nz, self.Ny, self.Nx = int(Nz), int(Ny), int(Nx)
        z0, z1, y0, y1, x0, x1 = (int(v) for v in box)
        for a, b, n, name in ((z0, z1, Nz, "z"), (y0, y1, Ny, "y"), (x0, x1, Nx, "x")):
            if not BOX_GAP <= a < b <= n - BOX_GAP:
                raise ValueError(f"The refined block must clear the tunnel's edges by {BOX_GAP} "
                                 f"cells along {name}: got [{a}, {b}) in {n}.")
        self.box: Box = (z0, z1, y0, y1, x0, x1)
        self._n = (z1 - z0, y1 - y0, x1 - x0)
        nz, ny, nx = self._n

        solid = np.ascontiguousarray(solid, dtype=bool)
        outside = solid.copy()
        outside[z0:z1, y0:y1, x0:x1] = False
        if outside.any():
            raise ValueError("Part of the body lies outside the refined block.")
        solid_fine = np.asarray(solid_fine, dtype=bool)
        if solid_fine.shape != (2 * nz, 2 * ny, 2 * nx):
            raise ValueError(f"solid_fine must have shape {(2 * nz, 2 * ny, 2 * nx)}, got {solid_fine.shape}")
        clear = np.zeros_like(solid_fine)
        clear[2:-2, 2:-2, 2:-2] = True
        if (solid_fine & ~clear).any():
            raise ValueError("The body must clear the refined block's faces by at least a coarse cell.")
        if solid.any() and not solid_fine.any():
            raise ValueError("The body covers no cells on the fine grid.")

        self.solid = solid
        self.solid_fine = solid_fine
        self.omega = float(omega)
        self.omega_fine = _omega_fine(self.omega)
        self.u0 = float(u0)
        self.D = float(D)
        self.rho0 = float(rho0)
        self._c2f = neq_scale_coarse_to_fine(self.omega, self.omega_fine)
        self._f2c = 1.0 / self._c2f

        # The coarse grid carries no body: under the block it is overwritten
        # every step, and outside it there is none.
        self._coarse = Solver3D(Nz=Nz, Ny=Ny, Nx=Nx, solid=np.zeros_like(solid), omega=omega,
                                u0=u0, D=D, rho0=rho0, ref_area=1.0, **kw)
        fine_kw = {k: kw[k] for k in _SHARED if k in kw}
        padded = np.zeros((2 * nz + 2, 2 * ny + 2, 2 * nx + 2), dtype=bool)
        padded[1:-1, 1:-1, 1:-1] = solid_fine
        # Every face of the fine block is driven by the interface: no inlet,
        # outlet or walls ("periodic" and an unknown wall_bc apply nothing,
        # and whatever streams into the ghost layer is overwritten).
        self._fine = Solver3D(
            Nz=2 * nz + 2, Ny=2 * ny + 2, Nx=2 * nx + 2, solid=padded,
            omega=self.omega_fine, u0=u0, D=REFINEMENT * D, rho0=rho0,
            ref_area=None if ref_area is None else REFINEMENT ** 2 * float(ref_area),
            wall_bc="interface", streamwise_bc="periodic", **fine_kw,
        )
        self.ref_area = float(ref_area) if ref_area is not None else self._fine.ref_area / REFINEMENT ** 2
        self.lattice = self._coarse.lattice
        self.collision = self._coarse.collision
        self.backend = self._coarse.backend
        self._stencils = [_axis_stencil(n) for n in self._n]
        self._use_numba = _HAS_NUMBA and self._coarse.backend == "numba"
        lat = self.lattice
        self._lat_e = (lat.ex.astype(np.float64), lat.ey.astype(np.float64), lat.ez.astype(np.float64))

        self.step_count = 0
        self.Cd_history: List[float] = []
        self.Cly_history: List[float] = []
        self.Clz_history: List[float] = []
        self.Cd_p_history: List[float] = []
        self.Cd_v_history: List[float] = []
        self._full_at = -1
        self._set_shell(self._shell())
        self._restrict(full=True)             # the coarse cells under the block start as the fine grid's

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _halo(self) -> np.ndarray:
        """The coarse cells the interface reads: the block and two cells around it."""
        z0, z1, y0, y1, x0, x1 = self.box
        return self._coarse.f[:, z0 - 2:z1 + 2, y0 - 2:y1 + 2, x0 - 2:x1 + 2]

    def _shell(self) -> List[np.ndarray]:
        """
        The fine ghost layer interpolated from the coarse grid as it stands:
        six faces, (x-, x+, y-, y+, z-, z+), each spanning the whole face
        including the edges -- raw, before the non-equilibrium rescaling.
        Tricubic, one axis at a time.
        """
        h = self._halo()
        sz, sy, sx = self._stencils
        # each face: across its normal, then along its own axes in (z, y, x) order
        plan = ((3, sx, sz, sy), (2, sy, sz, sx), (1, sz, sy, sx))
        faces = []
        if self._use_numba:
            for axis, normal, (ia, wa), (ib, wb) in plan:
                for e in (0, -1):
                    out = np.empty((h.shape[0], ia.shape[0], ib.shape[0]))
                    _face_kernel(h, axis, normal[0][e], normal[1][e], ia, wa, ib, wb, out)
                    faces.append(out)
            return faces
        for axis, (idx, w), (ia, wa), (ib, wb) in plan:
            for e in (0, -1):
                at = [slice(None)] * 4
                plane = 0.0
                for p in range(4):
                    at[axis] = idx[e, p]
                    plane = plane + h[tuple(at)] * w[e, p]
                faces.append(_interp(_interp(plane, 2, ib, wb), 1, ia, wa))
        return faces

    def prolong(self) -> None:
        """
        Set the whole fine block, ghosts included, from the coarse grid --
        tricubic, non-equilibrium rescaled -- for a coarse state set from
        outside (an initial condition, a restart).  The body's fine cells
        take the interpolated values too, which the first steps wash out.
        """
        sz, sy, sx = self._stencils
        fine = _interp(_interp(_interp(self._halo(), 3, *sx), 2, *sy), 1, *sz)
        self._fine.f[...] = _rescale_neq(fine, self._c2f, self.lattice, self._use_numba)

    def _set_shell(self, faces: Sequence[np.ndarray], later: Optional[Sequence[np.ndarray]] = None,
                   wb: float = 0.0) -> None:
        """
        Write the ghost layer: ``faces``, or between them and ``later`` at
        weight ``wb`` (a time level part-way through the coarse step), each
        with its non-equilibrium part rescaled for the fine grid.
        """
        f = self._fine.f
        later = faces if later is None else later
        if self._use_numba:
            ex, ey, ez = self._lat_e
            lat = self.lattice
            last = (f.shape[3] - 1, f.shape[2] - 1, f.shape[1] - 1)
            for k, (A, B) in enumerate(zip(faces, later)):
                axis = 2 - k // 2
                pos = 0 if k % 2 == 0 else last[k // 2]
                _write_face(f, axis, pos, A, B, wb, self._c2f, ex, ey, ez, lat.W, lat.h3_factor)
            return
        lat, s = self.lattice, self._c2f
        mixed = [a if wb == 0.0 else (1.0 - wb) * a + wb * b for a, b in zip(faces, later)]
        xm, xp, ym, yp, zm, zp = (_rescale_neq(face, s, lat, False) for face in mixed)
        f[:, :, :, 0] = xm
        f[:, :, :, -1] = xp
        f[:, :, 0, :] = ym
        f[:, :, -1, :] = yp
        f[:, 0, :, :] = zm
        f[:, -1, :, :] = zp

    def _restrict(self, full: bool = True) -> None:
        """
        The fine solution onto the coarse cells under the block (the ghost
        layer must be valid).  The coupling reads only the two coarse layers
        inside each face, so a step restricts those (``full=False``); the
        rest of the block -- what the viewer shows -- is brought up to date
        every :data:`FULL_RESTRICT_EVERY` steps and whenever it is read.
        """
        z0, z1, y0, y1, x0, x1 = self.box
        ff, fc = self._fine.f, self._coarse.f
        if self._use_numba:
            ex, ey, ez = self._lat_e
            _restrict_kernel(ff, fc, z0, y0, x0, RESTRICT_WEIGHTS, ex, ey, ez, self.lattice.W,
                             self._f2c, self.lattice.h3_factor, 0 if full else RESTRICT_BAND)
        else:
            fc[:, z0:z1, y0:y1, x0:x1] = restrict_block(ff, self._f2c, self.lattice)
        if full or not self._use_numba:
            self._full_at = self.step_count

    def _up_to_date(self) -> None:
        """Restrict the whole block if a step has moved on since it last was."""
        if self._full_at != self.step_count:
            self._restrict(full=True)

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def _step(self) -> Tuple[float, float, float]:
        """One coarse step: the tunnel, two fine steps, then the fine solution back onto the tunnel."""
        before = self._shell()                     # the coarse grid at t
        self._coarse._step()
        after = self._shell()                      # ... and at t + 1
        self._set_shell(before)
        cd1, cy1, cz1 = self._fine._step()
        p1, v1 = self._fine._last_cd_p, self._fine._last_cd_v
        self._set_shell(before, after, 0.5)
        cd2, cy2, cz2 = self._fine._step()
        # the restriction reads a ring of fine cells round each coarse one,
        # so the ghost layer must hold the coarse grid at t + 1, not what
        # streamed into it
        self._set_shell(after)
        self.step_count += 1
        self._restrict(full=self.step_count % FULL_RESTRICT_EVERY == 0)
        self._last_cd_p = 0.5 * (p1 + self._fine._last_cd_p)
        self._last_cd_v = 0.5 * (v1 + self._fine._last_cd_v)
        return 0.5 * (cd1 + cd2), 0.5 * (cy1 + cy2), 0.5 * (cz1 + cz2)

    def step(self) -> Tuple[float, float, float]:
        return self._step()

    def run(self, steps: int, check_every: int = 500, verbose: bool = True,
            callback: Optional[Callable] = None, **_ignored: Any) -> Dict[str, Any]:
        """Run ``steps`` coarse steps; the histories accumulate across calls, as Solver3D's do."""
        for k in range(1, int(steps) + 1):
            cd, cly, clz = self._step()
            self.Cd_history.append(float(cd))
            self.Cly_history.append(float(cly))
            self.Clz_history.append(float(clz))
            self.Cd_p_history.append(float(self._last_cd_p))
            self.Cd_v_history.append(float(self._last_cd_v))
            if k % max(int(check_every), 1) == 0:
                if verbose:
                    print(f"  step {self.step_count:6d}  Cd={cd:+.4f}  Cl_y={cly:+.4f}", flush=True)
                if callback is not None:
                    callback(self.step_count, cd, cly, clz)
        window = max(1, len(self.Cd_history) // 5)
        tail = {k: np.asarray(h[-window:]) for k, h in (
            ("Cd", self.Cd_history), ("Cly", self.Cly_history), ("Clz", self.Clz_history),
            ("Cd_p", self.Cd_p_history), ("Cd_v", self.Cd_v_history))}
        rho, ux, uy, uz = self.macroscopic()
        return {
            "Cd_mean": float(tail["Cd"].mean()), "Cd_std": float(tail["Cd"].std()),
            "Cly_mean": float(tail["Cly"].mean()), "Cly_std": float(tail["Cly"].std()),
            "Clz_mean": float(tail["Clz"].mean()), "Clz_std": float(tail["Clz"].std()),
            "Cd_p_mean": float(tail["Cd_p"].mean()), "Cd_v_mean": float(tail["Cd_v"].mean()),
            "Cd_history": self.Cd_history, "Cly_history": self.Cly_history,
            "Clz_history": self.Clz_history, "steps_completed": self.step_count,
            "stop_reason": "max_steps",
            "omega_coarse": self.omega, "omega_fine": self.omega_fine, "box": list(self.box),
            "rho": rho, "ux": ux, "uy": uy, "uz": uz,
        }

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    @property
    def f(self) -> np.ndarray:
        """The tunnel's distributions; under the block, the fine solution averaged."""
        self._up_to_date()
        return self._coarse.f

    @property
    def coarse(self):
        return self._coarse

    @property
    def fine(self):
        return self._fine

    def macroscopic(self, f: Optional[np.ndarray] = None) -> tuple:
        """``(rho, ux, uy, uz)`` on the tunnel grid."""
        if f is None:
            self._up_to_date()
        return self._coarse.macroscopic(f)

    def fine_macroscopic(self) -> tuple:
        """``(rho, ux, uy, uz)`` on the fine block, ghost layer stripped."""
        rho, ux, uy, uz = self._fine.macroscopic()
        inner = (slice(1, -1),) * 3
        return rho[inner], ux[inner], uy[inner], uz[inner]

    @property
    def cells(self) -> Dict[str, int]:
        """Cell counts, and the cell updates one coarse step costs, against refining everywhere."""
        coarse = self.Nz * self.Ny * self.Nx
        fine = int(np.prod(self._fine.f.shape[1:]))
        return {"coarse": coarse, "fine": fine, "updates_per_step": coarse + REFINEMENT * fine,
                "uniform_fine_updates_per_step": REFINEMENT ** 4 * coarse}

    @property
    def cost(self) -> float:
        """Estimated time per step, in steps of the tunnel without the block (:func:`refined_cost`)."""
        return refined_cost((self.Nz, self.Ny, self.Nx), self.box)

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
refining everything.  Refining the whole tunnel twice over costs 16x (8x
the cells, 2x the steps); a box a few diameters long costs what its cells
do, typically 3-5x in a crowded tunnel and much less in a roomy one.

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

The box is 43 x 29 x 29 tunnel cells, so a step costs the tunnel's cells
plus twice the box's eight-fold ones -- about 3.6x the plain grid's cell
updates, against 16x for refining everything.  A decaying shear wave whose
crests sit on the block's faces decays at the plain solver's rate (0.4% off
the analytic one); see the transfer operators below for why that took
fourth-order transfers.

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

    @nb.njit(cache=True)
    def _interp_kernel(src, axis, idx, w, out):                             # pragma: no cover
        """``_interp`` on (Q, A, B) along axis 1 or 2, into ``out``."""
        Q, A, B = src.shape
        P = idx.shape[1]
        for q in range(Q):
            if axis == 1:
                for m in range(idx.shape[0]):
                    for b in range(B):
                        acc = 0.0
                        for p in range(P):
                            acc += w[m, p] * src[q, idx[m, p], b]
                        out[q, m, b] = acc
            else:
                for a in range(A):
                    for m in range(idx.shape[0]):
                        acc = 0.0
                        for p in range(P):
                            acc += w[m, p] * src[q, a, idx[m, p]]
                        out[q, a, m] = acc

    @nb.njit(cache=True, parallel=True)
    def _restrict_kernel(ff, fc, z0, y0, x0, rw, ex, ey, ez, w, scale, h3):  # pragma: no cover
        """The fourth-order restriction of a whole block, non-equilibrium rescaled, in one pass."""
        Q = ff.shape[0]
        nz = (ff.shape[1] - 2) // 2
        ny = (ff.shape[2] - 2) // 2
        nx = (ff.shape[3] - 2) // 2
        for k in nb.prange(nz):
            avg = np.empty(Q)
            for j in range(ny):
                for i in range(nx):
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
        self._set_shell(self._shell())
        self._restrict()                      # the coarse cells under the block start as the fine grid's

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _halo(self) -> np.ndarray:
        """The coarse cells the interface reads: the block and two cells around it."""
        z0, z1, y0, y1, x0, x1 = self.box
        return self._coarse.f[:, z0 - 2:z1 + 2, y0 - 2:y1 + 2, x0 - 2:x1 + 2]

    def _along(self, a: np.ndarray, axis: int, stencil) -> np.ndarray:
        """Interpolate a (Q, A, B) plane along axis 1 or 2."""
        idx, w = stencil
        if not self._use_numba:
            return _interp(a, axis, idx, w)
        a = np.ascontiguousarray(a)
        shape = list(a.shape)
        shape[axis] = idx.shape[0]
        out = np.empty(shape)
        _interp_kernel(a, axis, idx, w, out)
        return out

    def _shell(self) -> List[np.ndarray]:
        """
        The fine ghost layer interpolated from the coarse grid as it stands:
        six faces, (x-, x+, y-, y+, z-, z+), each spanning the whole face
        including the edges -- raw, before the non-equilibrium rescaling.
        Tricubic, one axis at a time.
        """
        h = self._halo()
        sz, sy, sx = self._stencils

        def plane(axis, stencil, e):                          # the ghost plane across one axis
            idx, w = stencil
            out = None
            for p in range(4):
                at = [slice(None)] * 4
                at[axis] = idx[e, p]
                term = h[tuple(at)] * w[e, p]
                out = term if out is None else out + term
            return out

        faces = []
        for e in (0, -1):                                     # x faces: (Q, Mz, My)
            faces.append(self._along(self._along(plane(3, sx, e), 2, sy), 1, sz))
        for e in (0, -1):                                     # y faces: (Q, Mz, Mx)
            faces.append(self._along(self._along(plane(2, sy, e), 2, sx), 1, sz))
        for e in (0, -1):                                     # z faces: (Q, My, Mx)
            faces.append(self._along(self._along(plane(1, sz, e), 2, sx), 1, sy))
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

    def _set_shell(self, faces: Sequence[np.ndarray]) -> None:
        f = self._fine.f
        lat, s = self.lattice, self._c2f
        xm, xp, ym, yp, zm, zp = (_rescale_neq(face, s, lat, self._use_numba) for face in faces)
        f[:, :, :, 0] = xm
        f[:, :, :, -1] = xp
        f[:, :, 0, :] = ym
        f[:, :, -1, :] = yp
        f[:, 0, :, :] = zm
        f[:, -1, :, :] = zp

    def _restrict(self) -> None:
        """The fine solution onto the coarse cells under the block (the ghost layer must be valid)."""
        z0, z1, y0, y1, x0, x1 = self.box
        ff, fc = self._fine.f, self._coarse.f
        if self._use_numba:
            ex, ey, ez = self._lat_e
            _restrict_kernel(ff, fc, z0, y0, x0, RESTRICT_WEIGHTS, ex, ey, ez, self.lattice.W,
                             self._f2c, self.lattice.h3_factor)
        else:
            fc[:, z0:z1, y0:y1, x0:x1] = restrict_block(ff, self._f2c, self.lattice)

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
        self._set_shell([0.5 * (a + b) for a, b in zip(before, after)])
        cd2, cy2, cz2 = self._fine._step()
        # the restriction reads a ring of fine cells round each coarse one,
        # so the ghost layer must hold the coarse grid at t + 1, not what
        # streamed into it
        self._set_shell(after)
        self._restrict()
        self.step_count += 1
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
        return self._coarse.f

    @property
    def coarse(self):
        return self._coarse

    @property
    def fine(self):
        return self._fine

    def macroscopic(self, f: Optional[np.ndarray] = None) -> tuple:
        """``(rho, ux, uy, uz)`` on the tunnel grid."""
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

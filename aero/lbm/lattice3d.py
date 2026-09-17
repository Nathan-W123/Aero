"""
3D lattice descriptors: D3Q19 and D3Q27.

Why a second lattice
--------------------
D3Q19 drops the eight corner directions of the full 27-velocity cube.  It keeps
enough isotropy for the Navier-Stokes stress -- every moment Chapman-Enskog
needs through fourth order is exact -- so it is the right default, and it is
cheaper: measured 50.3 vs 41.0 MLUPS on a 48^3 box (numba, one core count),
so D3Q27 costs about 23% more per cell update -- less than the 42% the
direction count suggests, because the per-cell overhead amortises.

What it loses is one specific moment.  A single-speed lattice has components in
``{-1, 0, 1}``, so ``c_a^(2k) = c_a^2`` and any moment with a power above 2
collapses; no such lattice can represent those, D3Q27 included.  Of the moments
that *are* representable -- every power at most 2 -- exactly one differs between
the two velocity sets::

    sum_i w_i c_ix^2 c_iy^2 c_iz^2  =  0        (D3Q19)
                                    =  cs^6     (D3Q27, exact)

D3Q19 has no direction with all three components non-zero, so it scores that
moment as zero instead of ``1/27``.  Every other representable moment agrees
between them.  That single hole is the origin of D3Q19's ``O(u^3)`` Galilean-
invariance defect: the missing term is fully three-dimensional, so the error it
leaves behind depends on the *orientation* of the flow relative to the lattice
axes.  A laminar or axis-aligned run barely notices.  Resolved turbulence at
moderate Mach does: a shear layer or jet skewed to the axes picks up a spurious
anisotropy that grid refinement does not remove, because it is a property of
the velocity set and not of the discretisation error.

D3Q27 restores the full cubic symmetry, for the ~23% above.  Use D3Q19 for
everything routine and D3Q27 when the answer is a turbulence statistic.

What is lattice-specific and what is not
----------------------------------------
Almost nothing is.  The BGK, TRT and regularized collisions, streaming and
bounce-back are all written against ``(E, W, OPP)`` and loop to ``f.shape[0]``,
so they take either lattice unchanged.  The parts that genuinely depend on the
velocity set are:

* the Zou-He inlet closure, which needs to know which directions are unknown at
  an ``x = 0`` face and how to spread transverse momentum over them;
* the specular-reflection walls, which need the y-mirror permutation;
* MRT, whose 19x19 moment matrix is built for D3Q19 specifically.

The first two are derived here from the velocity vectors rather than written
out by hand, which is what makes a second lattice cheap.  MRT stays D3Q19-only
and the solver rejects the combination rather than silently transforming in the
wrong basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

CS2 = 1.0 / 3.0


def _opposite_index(e: np.ndarray) -> np.ndarray:
    """Index of -e_i for every i.  Every 3D lattice here is symmetric."""
    lookup = {tuple(v): i for i, v in enumerate(e.tolist())}
    return np.array([lookup[tuple(-v for v in row)] for row in e.tolist()],
                    dtype=np.int8)


def _mirror_index(e: np.ndarray, axis: int) -> np.ndarray:
    """Index of e_i with component ``axis`` negated (specular reflection)."""
    lookup = {tuple(v): i for i, v in enumerate(e.tolist())}
    out = []
    for row in e.tolist():
        mirrored = list(row)
        mirrored[axis] = -mirrored[axis]
        out.append(lookup[tuple(mirrored)])
    return np.array(out, dtype=np.int8)


@dataclass(frozen=True)
class Lattice3D:
    """
    A 3D velocity set plus everything derived from it.

    The derived index groups are what the boundary conditions consume; deriving
    rather than hand-listing them is the whole point, because a transcription
    slip in a 27-entry table is invisible until it shows up as a momentum leak.
    """

    name: str
    E: np.ndarray                      # (Q, 3) int8, columns (ex, ey, ez)
    W: np.ndarray                      # (Q,) float64
    #: carry the third-order Hermite term in the equilibrium.  Only legitimate
    #: where the lattice has the ``c_x^2 c_y^2 c_z^2`` moment to support it,
    #: i.e. D3Q27.  See :func:`compute_feq`.
    h3: bool = False
    OPP: np.ndarray = field(init=False)
    Y_MIR: np.ndarray = field(init=False)
    Z_MIR: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "OPP", _opposite_index(self.E))
        object.__setattr__(self, "Y_MIR", _mirror_index(self.E, 1))
        object.__setattr__(self, "Z_MIR", _mirror_index(self.E, 2))

    # -- basic shape ------------------------------------------------------
    @property
    def Q(self) -> int:
        return int(self.E.shape[0])

    @property
    def h3_factor(self) -> float:
        """``h3`` as a float, for kernels that take it as a scalar multiplier."""
        return 1.0 if self.h3 else 0.0

    @property
    def ex(self) -> np.ndarray:
        return self.E[:, 0].astype(np.float64)

    @property
    def ey(self) -> np.ndarray:
        return self.E[:, 1].astype(np.float64)

    @property
    def ez(self) -> np.ndarray:
        return self.E[:, 2].astype(np.float64)

    # -- Zou-He index groups for an x-normal face -------------------------
    @property
    def x_plus(self) -> np.ndarray:
        """Directions with ex > 0 — the unknowns at an inlet face x = 0."""
        return np.nonzero(self.E[:, 0] > 0)[0]

    @property
    def x_minus(self) -> np.ndarray:
        return np.nonzero(self.E[:, 0] < 0)[0]

    @property
    def x_zero(self) -> np.ndarray:
        return np.nonzero(self.E[:, 0] == 0)[0]

    @property
    def transverse_norm(self) -> float:
        """
        ``sum_{ex>0} w_i e_iy^2``, the normalisation for spreading transverse
        momentum over the unknown directions.

        It is 1/18 for both lattices, but it is computed rather than assumed:
        the identity that makes it so (``sum_i w_i e_ia^2 = cs^2`` with
        ``|e_ia| <= 1``) is a property of the velocity set, not a constant.
        """
        u = self.x_plus
        return float((self.W[u] * self.ey[u] ** 2).sum())

    # -- y-normal wall index groups ---------------------------------------
    @property
    def y_plus(self) -> np.ndarray:
        """Directions with ey > 0 — the unknowns at a bottom wall y = 0."""
        return np.nonzero(self.E[:, 1] > 0)[0]

    @property
    def y_minus(self) -> np.ndarray:
        return np.nonzero(self.E[:, 1] < 0)[0]

    def moment_errors(self) -> Dict[str, float]:
        """
        Departure from the isotropic Maxwellian for each moment order.

        Returns max-abs errors keyed ``"mass"``, ``"m1"`` ... ``"m4"`` plus
        ``"m222"`` for ``sum_i w_i c_ix^2 c_iy^2 c_iz^2``, the one moment that
        separates the two lattices.  Everything through ``m4`` must be zero for
        the hydrodynamics to be right; ``m222`` is zero only for D3Q27.

        Returned rather than asserted so a caller can report the numbers --
        this is the only thing standing between a mistyped weight and a
        viscosity that is quietly wrong by a few percent.
        """
        w, e = self.W, np.stack([self.ex, self.ey, self.ez])
        d = np.eye(3)
        m4_want = CS2 ** 2 * (
            np.einsum("ab,cd->abcd", d, d)
            + np.einsum("ac,bd->abcd", d, d)
            + np.einsum("ad,bc->abcd", d, d)
        )
        return {
            "mass": abs(float(w.sum()) - 1.0),
            "m1": float(np.abs(np.einsum("i,ai->a", w, e)).max()),
            "m2": float(np.abs(
                np.einsum("i,ai,bi->ab", w, e, e) - CS2 * d
            ).max()),
            "m3": float(np.abs(np.einsum("i,ai,bi,ci->abc", w, e, e, e)).max()),
            "m4": float(np.abs(
                np.einsum("i,ai,bi,ci,di->abcd", w, e, e, e, e) - m4_want
            ).max()),
            "m222": abs(
                float((w * self.ex ** 2 * self.ey ** 2 * self.ez ** 2).sum())
                - CS2 ** 3
            ),
        }


# ---------------------------------------------------------------------------
# D3Q19 — the default.  Ordering is load-bearing: checkpoints on disk and the
# MRT moment matrix in mrt3d.py both index into it.
# ---------------------------------------------------------------------------

_E19 = np.array([
    [ 0,  0,  0],
    [ 1,  0,  0], [-1,  0,  0], [ 0,  1,  0], [ 0, -1,  0],
    [ 0,  0,  1], [ 0,  0, -1],
    [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0],
    [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
    [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1],
], dtype=np.int8)

_W19 = np.array(
    [1.0 / 3.0]
    + [1.0 / 18.0] * 6
    + [1.0 / 36.0] * 12,
    dtype=np.float64,
)


def _build_d3q27() -> Tuple[np.ndarray, np.ndarray]:
    """
    The full 27-velocity cube, generated rather than typed out.

    Ordering extends D3Q19: the first 19 entries are exactly D3Q19, so a D3Q19
    checkpoint's populations keep their meaning and the eight corners are
    appended.  Weights come from the 1D product rule ``w(0) = 2/3``,
    ``w(+-1) = 1/6``, which is what makes the velocity set a Gauss-Hermite
    quadrature of the Maxwellian and hence isotropic to fourth order.
    """
    w1d = {0: 2.0 / 3.0, 1: 1.0 / 6.0, -1: 1.0 / 6.0}
    corners = [
        [x, y, z]
        for x in (1, -1) for y in (1, -1) for z in (1, -1)
    ]
    e = np.concatenate([_E19, np.array(corners, dtype=np.int8)], axis=0)
    w = np.array(
        [w1d[int(a)] * w1d[int(b)] * w1d[int(c)] for a, b, c in e.tolist()],
        dtype=np.float64,
    )
    return e, w


_E27, _W27 = _build_d3q27()

D3Q19 = Lattice3D("d3q19", _E19, _W19, h3=False)
D3Q27 = Lattice3D("d3q27", _E27, _W27, h3=True)

LATTICES: Dict[str, Lattice3D] = {"d3q19": D3Q19, "d3q27": D3Q27}


def get_lattice3d(name) -> Lattice3D:
    """Look up a lattice by name; accepts a :class:`Lattice3D` unchanged."""
    if isinstance(name, Lattice3D):
        return name
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key not in LATTICES:
        raise ValueError(
            f"unknown 3D lattice {name!r}; choose from {sorted(LATTICES)}"
        )
    return LATTICES[key]


# ---------------------------------------------------------------------------
# Lattice-generic moments
# ---------------------------------------------------------------------------

def compute_macroscopic(
    f: np.ndarray, lattice: Lattice3D = D3Q19
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rho, ux, uy, uz from f of shape ``(Q, Nz, Ny, Nx)``."""
    from .physics import inv_positive

    rho = f.sum(axis=0)
    inv_rho = inv_positive(rho)
    return (
        rho,
        inv_rho * np.einsum("i,izyx->zyx", lattice.ex, f),
        inv_rho * np.einsum("i,izyx->zyx", lattice.ey, f),
        inv_rho * np.einsum("i,izyx->zyx", lattice.ez, f),
    )


def compute_feq(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    lattice: Lattice3D = D3Q19,
) -> np.ndarray:
    """
    Hermite equilibrium, second order plus (on D3Q27) the third-order term.

    The second-order part is the usual
    ``w_i rho (1 + 3 e.u + 4.5 (e.u)^2 - 1.5 u.u)``.

    On a lattice with ``h3`` set, the next Hermite term is added::

        + w_i rho / (6 cs^6) * [ (e.u)^3 - 3 cs^2 (e.u) u^2 ]
        = + w_i rho * 4.5 * [ (e.u)^3 - (e.u) u^2 ]      for cs^2 = 1/3

    This is the whole point of carrying 27 velocities.  The term is what
    cancels the ``O(u^3)`` Galilean-invariance error, and it needs the
    ``c_x^2 c_y^2 c_z^2`` moment to be correct -- which is exactly the moment
    D3Q19 does not have.  Measured on an advected shear wave, enabling it on
    D3Q27 cuts the advection-induced viscosity error from -4.0e-2 to -8.3e-5
    at ``u = 0.2``, a factor of ~490.  Enabling it on D3Q19 instead only gets
    to -1.9e-2, because the lattice cannot represent what the term asks for --
    hence ``h3`` is a property of the lattice and not a user option.

    It also means D3Q27 with the second-order equilibrium alone is pointless:
    it reproduces D3Q19 to the last bit while costing ~23% more.
    """
    usq = ux * ux + uy * uy + uz * uz
    feq = np.empty((lattice.Q, *rho.shape), dtype=np.float64)
    h3 = lattice.h3
    for i in range(lattice.Q):
        eu = lattice.ex[i] * ux + lattice.ey[i] * uy + lattice.ez[i] * uz
        poly = 1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq
        if h3:
            poly = poly + 4.5 * (eu * eu * eu - eu * usq)
        feq[i] = lattice.W[i] * rho * poly
    return feq

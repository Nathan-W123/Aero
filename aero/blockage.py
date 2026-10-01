"""
Blockage: how much the tunnel's walls raise the drag, and the drag corrected
to free air.

What the tunnel is
------------------
The side walls are slip walls (mirrors) in y, and z is periodic, so a body in
this tunnel is one of an infinite array of copies -- a row in 2D, a square
array in 3D -- and "blockage" is its share of the cross-section.  The copies
speed the flow past the body up and steepen the pressure drop behind it, so
the drag reads high, by more than the textbook high-Reynolds-number
corrections (Maskell 1963; Allen & Vincenti 1944) allow for: at the Reynolds
numbers this tunnel runs, viscosity carries the walls' influence much
further.

The model
---------
::

    Cd_tunnel = Cd_free * (1 + K),      K = k1 b + k2 b^2

``b`` is the linear blockage: in 2D the body's frontal height over the
tunnel's; in 3D the diameter of the disc with the body's frontal area over
that of the disc with the tunnel's cross-section, ``sqrt(S / C)`` with the
same constant, which for a sphere in a square tunnel is just ``D / span``.

``k1`` and ``k2`` were measured with this solver (:data:`CALIBRATION`), per
shape: the same body, at fixed resolution, in tunnels of three or four
widths, fitted for ``Cd_free`` together with the coefficients (a negative
``k1`` -- a law that would lower the drag a little at small blockage -- is
dropped and the fit repeated).  They depend on the Reynolds number and are
interpolated in ``log Re`` between the measured points, held at the
nearest outside them.  A shape measured in only two tunnels takes its
reference shape's law, scaled to fit its two points.

What the measurements say: confinement is a far stronger effect in 2D than
in 3D at the same ``b`` -- a row of cylinders chokes the flow between them,
an array of spheres lets it round them -- and in 3D it grows as ``b^2``, so
it is small below 15% and climbs fast above 25%.  An earlier calibration
fitted a straight line in ``b`` through two or three points and put +10% on
a sphere's drag at 10% blockage, where the measured law gives +2%.  The
shapes differ: a square cylinder is confined about a third more than a
circular one, and a cube about half as much again as a sphere of the same
frontal area (1.5x at Re 20, 1.65x at Re 100) -- the sharp edges throw the
separated flow wider.  That, not the fit, is what limits a correction for
a shape nothing was measured for.

Measured
~~~~~~~~
Cylinder and square 20 cells across in tunnels 30 D long (2D); sphere 10
cells across and a cube of the same frontal area, in tunnels 16 D long
(3D); BGK, u0 = 0.05, even cell counts across (an odd one puts the body
half a cell off the grid's centre line, a different staircase, which moved
the 10-cell sphere's drag 5% -- as much as the blockage did).  Cd is the
mean once the run settles -- steady to 5e-4 per ten body passages, or for
shedding the last 100-200 passages -- on the nominal size.  Cd_free is the
fit's zero-blockage value, which carries the grid's own error: against the
references it is a few percent high, as a body this coarse should be.
The earlier sweeps of a 14-cell sphere in tunnels of other lengths agree
with the sphere laws' ratios between blockages to 0.3-1.7%.  The sphere at
Re 50 and the cube were run in two tunnels each, which fix one coefficient:
the sphere's at Re 50 is in the table because interpolating between Re 20
and 100 would have under-read it by a fifth.  A sphere at
Re 200 was not measured: ten cells across it needs omega = 1.97, past what
BGK holds, so above Re 100 the Re 100 law stands, with the extrapolation
allowance on its uncertainty.

.. measured-begin

2D cylinder, Cd by blockage b:

    =====  =======  =======  =======  =======
       Re    0.050    0.100    0.200    0.303
    =====  =======  =======  =======  =======
       10   3.0735   3.3972   4.4586   6.2558
       20   2.2073   2.4123   3.1002   4.2658
       40   1.6494   1.7980   2.3045   3.1568
      100   1.4514   1.5287   1.8618   2.5023
    =====  =======  =======  =======  =======

2D rectangle (a square), Cd by blockage b:

    =====  =======  =======  =======  =======
       Re    0.050    0.100    0.200    0.303
    =====  =======  =======  =======  =======
       20   2.4778   2.7648   3.7843   5.6964
      100   1.5780   1.6971   2.1878   3.1628
    =====  =======  =======  =======  =======

3D box (a cube), Cd by blockage b:

    =====  =======  =======
       Re    0.154    0.299
    =====  =======  =======
       20   3.3294   3.9589
      100   1.3807   1.5668
    =====  =======  =======

3D sphere, Cd by blockage b:

    =====  =======  =======  =======  =======
       Re    0.100    0.152    0.200    0.294
    =====  =======  =======  =======  =======
       20   2.8082   2.8559   2.9401   3.2346
       50       --       --   1.7018   1.8463
      100   1.1455   1.1533   1.1723   1.2546
    =====  =======  =======  =======  =======

Fits:

    ============  =====  ======  ======  =======  ============  ========
    shape            Re      k1      k2  Cd_free  reference     fit res.
    ============  =====  ======  ======  =======  ============  ========
    2d cylinder      10   0.485  10.743   2.9289  2.846  +2.9%     0.34%
    2d cylinder      20   0.383   9.762   2.1180  2.045  +3.6%     0.27%
    2d cylinder      40   0.375   9.570   1.5834  1.522  +4.0%     0.20%
    2d cylinder     100   0.000   8.270   1.4137  1.340  +5.5%     1.05%
    2d rectangle     20   0.000  14.923   2.3918            --     0.93%
    2d rectangle    100   0.000  11.505   1.5227            --     1.63%
    3d box           20   0.000   3.089   3.1022            --  2 tunnels
    3d box          100   0.000   2.158   1.3135            --  2 tunnels
    3d sphere        20   0.000   2.072   2.7336  2.610  +4.8%     0.68%
    3d sphere        50   0.000   1.973   1.5773  1.538  +2.5%  2 tunnels
    3d sphere       100   0.000   1.309   1.1227  1.092  +2.8%     0.79%
    ============  =====  ======  ======  =======  ============  ========

.. measured-end

Uncertainty
-----------
The correction comes with its own uncertainty, a fraction of ``K``: what the
fits and their form leave for a calibrated shape (:data:`CALIBRATED_SPREAD`);
for any other -- an uploaded mesh, a finite 3D cylinder, a rectangle or box
not proportioned like the square or cube -- more, plus the spread between
the calibrated shapes' laws (:data:`UNCALIBRATED_SPREAD`); and more outside
the calibrated Reynolds numbers and blockages.  The measured alternative is
a blockage study -- the same case in wider tunnels, extrapolated to zero
blockage -- which the web UI runs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: (Re, k1, k2) by mode and shape, from the sweeps in the module docstring.
CALIBRATION: Dict[str, Dict[str, Tuple[Tuple[float, float, float], ...]]] = {
    "2d": {
        "cylinder": ((10.0, 0.485, 10.743), (20.0, 0.383, 9.762), (40.0, 0.375, 9.570), (100.0, 0.000, 8.270)),
        "rectangle": ((20.0, 0.000, 14.923), (100.0, 0.000, 11.505)),
    },
    "3d": {
        "box": ((20.0, 0.000, 3.089), (100.0, 0.000, 2.158)),
        "sphere": ((20.0, 0.000, 2.072), (50.0, 0.000, 1.973), (100.0, 0.000, 1.309)),
    },
}

#: Relative uncertainty of K for a shape that was calibrated (what the fits
#: and their form leave), and for one that was not -- an uploaded mesh, a
#: finite 3D cylinder -- on top of how far the calibrated shapes differ.
CALIBRATED_SPREAD = 0.15
UNCALIBRATED_SPREAD = 0.25
#: The extra allowance per factor of e outside the calibrated Reynolds numbers.
RE_EXTRAPOLATION_SPREAD = 0.15
#: The largest blockage measured, by mode; beyond it the uncertainty grows
#: by BEYOND_SPREAD for each further multiple of it.
B_MAX = {"2d": 0.30, "3d": 0.30}
BEYOND_SPREAD = 1.0


# ---------------------------------------------------------------------------
# How much of the tunnel the body blocks
# ---------------------------------------------------------------------------

def linear_blockage_from_area(area: float, cross_section: float) -> float:
    """``sqrt(4 S / (pi C))`` in 3D -- the equal-area disc's diameter ratio."""
    return math.sqrt(4.0 * max(float(area), 0.0) / (math.pi * max(float(cross_section), 1e-30)))


def blockage_of_solid(solid: np.ndarray) -> Dict[str, Any]:
    """
    ``b`` (see the module docstring), the frontal-area ratio, and which
    law applies (``"2d"`` or ``"3d"``) for a voxel body: (Ny, Nx) in 2D,
    (Nz, Ny, Nx) in 3D.  A 3D body spanning the whole periodic width is an
    infinite one, and blocks the tunnel as a 2D body does.
    """
    solid = np.asarray(solid, dtype=bool)
    if solid.ndim == 2:
        ny = solid.shape[0]
        h = float(solid.any(axis=1).sum())
        return {"b": h / ny, "area_ratio": h / ny, "frontal": h, "law": "2d"}
    nz, ny, _ = solid.shape
    if solid.any(axis=(1, 2)).all():
        h = float(solid.any(axis=(0, 2)).sum())
        return {"b": h / ny, "area_ratio": h / ny, "frontal": h * nz, "law": "2d"}
    area = float(solid.any(axis=2).sum())
    return {"b": linear_blockage_from_area(area, ny * nz), "area_ratio": area / (ny * nz),
            "frontal": area, "law": "3d"}


def frontal_size(mode: str, shape: str, params: Dict[str, Any]) -> Optional[float]:
    """
    The body's frontal height (2D) or area (3D) in cells, from the case's
    numbers alone, for the shapes that have a formula; None otherwise.
    """
    def f(key, default):
        try:
            return float(params.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    shape = str(shape).lower()
    if mode == "2d":
        return 2.0 * f("radius", 20.0) if shape == "cylinder" else f("height", 20.0) if shape == "rectangle" else None
    if shape == "sphere":
        return math.pi * f("radius", 7.0) ** 2
    if shape == "box":
        return f("height", 10.0) * f("depth", 10.0)
    if shape == "cylinder":
        return 2.0 * f("radius", 8.0) * min(f("length", 24.0), f("nz", 48.0))
    return None


def blockage_of_case(mode: str, shape: str, params: Dict[str, Any],
                     solid: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """``b`` and the area ratio for a case, from its numbers or, failing that, its voxels."""
    if solid is not None:
        return blockage_of_solid(solid)
    size = frontal_size(mode, shape, params)
    if size is None:
        return None

    def f(key, default):
        try:
            return max(float(params.get(key, default)), 1.0)
        except (TypeError, ValueError):
            return float(default)

    if mode == "2d":
        b = size / f("ny", 200.0)
        return {"b": b, "area_ratio": b, "frontal": size, "law": "2d"}
    if str(shape).lower() == "cylinder" and f("length", 24.0) >= f("nz", 48.0):
        b = 2.0 * f("radius", 8.0) / f("ny", 48.0)          # spans the periodic width: a 2D body
        return {"b": b, "area_ratio": b, "frontal": size, "law": "2d"}
    cross = f("ny", 48.0) * f("nz", 48.0)
    return {"b": linear_blockage_from_area(size, cross), "area_ratio": size / cross, "frontal": size,
            "law": "3d"}


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------

def _coefficients(table: Sequence[Tuple[float, float, float]], re: float) -> Tuple[float, float, float]:
    """(k1, k2, how far outside the table's Re range, in ln units), interpolated in log Re."""
    lre = math.log(max(float(re), 1e-6))
    lo, hi = math.log(table[0][0]), math.log(table[-1][0])
    if lre <= lo:
        return table[0][1], table[0][2], lo - lre
    if lre >= hi:
        return table[-1][1], table[-1][2], lre - hi
    for (r0, a0, b0), (r1, a1, b1) in zip(table, table[1:]):
        if lre <= math.log(r1):
            t = (lre - math.log(r0)) / (math.log(r1) - math.log(r0))
            return a0 + t * (a1 - a0), b0 + t * (b1 - b0), 0.0
    return table[-1][1], table[-1][2], 0.0


def _proportioned_like_calibration(shape: str, params: Optional[Dict[str, Any]]) -> bool:
    """
    Whether a body has the proportions its shape was calibrated with: the
    rectangle and the box were measured as a square and a cube.  Anything
    else of those shapes takes their law with an uncalibrated shape's
    uncertainty.
    """
    if not params or shape not in ("rectangle", "box"):
        return True

    def f(key):
        try:
            return float(params.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    sides = [f("width"), f("height")] + ([f("depth")] if shape == "box" else [])
    if min(sides) <= 0.0:
        return True
    return max(sides) / min(sides) <= 1.25


def blockage_correction(mode: str, re: float, b: float, shape: Optional[str] = None,
                        params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    How much the tunnel raises Cd at blockage ``b`` and Reynolds number
    ``re``: ``K`` (Cd_tunnel = Cd_free (1 + K)), its uncertainty ``dK``, and
    what the uncertainty is made of.  ``params``, the case's numbers, say
    whether a rectangle or box has the proportions it was calibrated with.

    A shape with its own calibration uses it.  Any other takes the mean of
    the mode's calibrated shapes, with an uncertainty that spans them: the
    shapes measured so far differ by more than any one fit's error, so the
    spread between them is the honest measure of not knowing.
    """
    mode = "3d" if str(mode).lower() == "3d" else "2d"
    tables = CALIBRATION.get(mode) or {}
    if not tables:
        raise ValueError(f"no blockage calibration for mode {mode!r}")
    b = max(float(b), 0.0)
    shape = (shape or "").lower()

    def k_of(table):
        k1, k2, out = _coefficients(table, re)
        return k1 * b + k2 * b * b, k1, k2, out

    if shape in tables:
        K, k1, k2, outside = k_of(tables[shape])
        if _proportioned_like_calibration(shape, params):
            spread, basis = CALIBRATED_SPREAD, "measured for this shape"
        else:
            like = "a square" if shape == "rectangle" else "a cube"
            spread, basis = UNCALIBRATED_SPREAD, f"measured for {like}, and this {shape}'s proportions differ"
        dK = spread * K
    else:
        each = [k_of(t) for t in tables.values()]
        Ks = [e[0] for e in each]
        K = float(np.mean(Ks))
        k1 = float(np.mean([e[1] for e in each]))
        k2 = float(np.mean([e[2] for e in each]))
        outside = max(e[3] for e in each)
        spread = UNCALIBRATED_SPREAD
        names = " and ".join(sorted(tables))
        basis = (f"not measured for this shape: the mean of the {names} laws, and their spread"
                 if len(tables) > 1 else f"not measured for this shape: the {names} law")
        dK = spread * K + 0.5 * (max(Ks) - min(Ks))
    extra = 0.0
    if outside > 0:
        extra += RE_EXTRAPOLATION_SPREAD * outside
        basis += f"; Re {re:g} is outside the range measured"
    if b > B_MAX[mode]:
        extra += BEYOND_SPREAD * (b / B_MAX[mode] - 1.0)
        basis += f"; blockage beyond the {B_MAX[mode]:.0%} measured"
    dK += extra * K
    return {"K": K, "dK": dK, "factor": 1.0 + K, "b": b, "k1": k1, "k2": k2,
            "relative_uncertainty": dK / K if K > 0 else 0.0, "basis": basis}


def free_air_cd(cd: float, corr: Dict[str, Any]) -> Tuple[float, float]:
    """Cd corrected to free air, and its uncertainty from the correction alone."""
    f = corr["factor"]
    cd_free = float(cd) / f
    return cd_free, cd_free * corr["dK"] / f

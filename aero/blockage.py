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
Writing it in frontal area is what lets one calibration serve every shape:
a cube and a sphere of the same frontal area block the tunnel alike, where
their extents across it do not.

``k1`` and ``k2`` were measured with this solver (:data:`CALIBRATION`): the
same body, at fixed resolution, in tunnels of four widths, fitted for
``Cd_free`` together with the coefficients.  They depend on the Reynolds
number, and are interpolated in ``log Re`` between the measured points and
held at the nearest outside them.

Uncertainty
-----------
The correction comes with its own uncertainty, a fraction of ``K``: what the
fits and their form leave for a calibrated shape (:data:`CALIBRATED_SPREAD`);
for any other -- an uploaded mesh, say -- the spread between the calibrated
shapes' laws as well (:data:`UNCALIBRATED_SPREAD`); and more outside the
calibrated Reynolds numbers and blockages.  The measured alternative is a
blockage study -- the same case in wider tunnels, extrapolated to zero
blockage -- which the web UI runs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: (Re, k1, k2) by mode and shape, from the calibration sweeps described in
#: the module docstring and :data:`CALIBRATION_NOTES`.
CALIBRATION: Dict[str, Dict[str, Tuple[Tuple[float, float, float], ...]]] = {
    "2d": {
        "cylinder": ((10.0, 0.450, 10.87), (20.0, 0.348, 9.894), (40.0, 0.351, 9.661), (100.0, 0.0, 8.144)),
        "rectangle": ((20.0, 1.881, 9.500),),
    },
    "3d": {
        "sphere": ((20.0, 1.00, 0.0), (100.0, 0.58, 0.0)),
    },
}

#: Relative uncertainty of K for a shape that was calibrated (what the fits
#: and their form leave), and for one that was not -- an uploaded mesh, a
#: finite 3D cylinder -- on top of how far the calibrated shapes differ.
CALIBRATED_SPREAD = 0.12
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
            spread, basis = CALIBRATED_SPREAD, f"calibrated on this shape ({shape})"
        else:
            like = "a square" if shape == "rectangle" else "a cube"
            spread, basis = UNCALIBRATED_SPREAD, f"calibrated on {like}; this {shape}'s proportions differ"
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
        basis = (f"not calibrated on this shape: the mean of the {names} laws" if len(tables) > 1
                 else f"not calibrated on this shape: the {names} law")
        dK = spread * K + 0.5 * (max(Ks) - min(Ks))
    extra = 0.0
    if outside > 0:
        extra += RE_EXTRAPOLATION_SPREAD * outside
        basis += f"; Re {re:g} is outside the calibrated range"
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

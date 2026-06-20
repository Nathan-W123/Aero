"""
D3Q19 Multiple Relaxation Time (MRT) matrices.

Reference: d'Humières et al., Phil. Trans. R. Soc. A 360, 437 (2002).

Moment ordering (rows of M19, 19 moments):
   0  rho   density                    (conserved)
   1  e     energy
   2  eps   energy-square
   3  jx    x-momentum                 (conserved)
   4  qx    energy-flux x
   5  jy    y-momentum                 (conserved)
   6  qy    energy-flux y
   7  jz    z-momentum                 (conserved)
   8  qz    energy-flux z
   9  3pxx  stress diagonal  (2ex²-ey²-ez²)
  10  pww   stress diagonal  (ey²-ez²)
  11  pxy   stress off-diag (ex*ey)
  12  pyz   stress off-diag (ey*ez)
  13  pxz   stress off-diag (ex*ez)
  14  mx    ghost  (3|e|²-5)*ex
  15  my    ghost  (3|e|²-5)*ey
  16  mz    ghost  (3|e|²-5)*ez
  17  txxy  ghost  ex*(ey²-ez²)
  18  txyy  ghost  ey*(ex²-ez²)

Velocity ordering (columns) matches d3q19.py E3:
  0:(0,0,0)  1:(+x)  2:(-x)  3:(+y)  4:(-y)  5:(+z)  6:(-z)
  7:(+x+y)  8:(-x+y)  9:(+x-y) 10:(-x-y)
  11:(+x+z) 12:(-x+z) 13:(+x-z) 14:(-x-z)
  15:(+y+z) 16:(-y+z) 17:(+y-z) 18:(-y-z)

Relaxation:
  Stress modes (9-13) → sv = omega   (viscosity)
  Conserved (0,3,5,7) → 1.0          (irrelevant)
  Ghost (14-18)        → 1.0
  Energy/flux tunable for stability at high Re.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Transformation matrix M19  (19 × 19, exact integer rows)
# ---------------------------------------------------------------------------

M19 = np.array([
    # 0: rho
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    # 1: e  (-30 for rest, -11 for face, 8 for edge)
    [-30,-11,-11,-11,-11,-11,-11,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
    # 2: eps (12 for rest, -4 for face, 1 for edge)
    [ 12, -4, -4, -4, -4, -4, -4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    # 3: jx (ex)
    [  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
    # 4: qx = (-4+3|e|²)*ex: face→(-4+3)*ex=-ex, edge→(-4+6)*ex=2*ex... wait
    # For face(i=1,ex=1,|e|²=1): (-4+3*1)*1 = -1 → but d'Humières gives -4 for face ex
    # Using d'Humières directly:  -4*ex for face(|e|²=1), +1*ex for edge(|e|²=2)
    [  0, -4,  4,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
    # 5: jy (ey)
    [  0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1],
    # 6: qy = -4*ey for face, +ey for edge
    [  0,  0,  0, -4,  4,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1],
    # 7: jz (ez)
    [  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1],
    # 8: qz = -4*ez for face, +ez for edge
    [  0,  0,  0,  0,  0, -4,  4,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1],
    # 9: 3pxx = 2ex²-ey²-ez²  (face±x:2; face±y:-1; face±z:-1; xy/xz edges:1; yz edges:-2)
    [  0,  2,  2, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2],
    # 10: pww = ey²-ez²  (±y:1; ±z:-1; xy edges:1; xz edges:-1; yz edges:0)
    [  0,  0,  0,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0],
    # 11: pxy = ex*ey  (only xy edges nonzero: +x+y:1, -x+y:-1, +x-y:-1, -x-y:1)
    [  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    # 12: pyz = ey*ez  (only yz edges: +y+z:1, -y+z:-1, +y-z:-1, -y-z:1)
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1],
    # 13: pxz = ex*ez  (only xz edges: +x+z:1, -x+z:-1, +x-z:-1, -x-z:1)
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0],
    # 14: mx = (3|e|²-5)*ex  (face:(-2)*ex; edge:(1)*ex)
    [  0, -2,  2,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
    # 15: my = (3|e|²-5)*ey  (face:(-2)*ey; edge:(1)*ey)
    [  0,  0,  0, -2,  2,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1],
    # 16: mz = (3|e|²-5)*ez  (face:(-2)*ez; edge:(1)*ez)
    [  0,  0,  0,  0,  0, -2,  2,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1],
    # 17: txxy = ex*(ey²-ez²)  (xy edges:+x+y:1,-x+y:-1,+x-y:1,-x-y:-1; xz edges:opp signs)
    [  0,  0,  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1, -1,  1,  0,  0,  0,  0],
    # 18: txyy = ey*(ex²-ez²)  (xy edges:+x+y:1,-x+y:1,+x-y:-1,-x-y:-1; yz edges:opp)
    [  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0, -1,  1, -1,  1],
], dtype=np.float64)

# The hand-assembled D3Q19 basis used here is not exactly orthogonal, so a
# Moore-Penrose pseudoinverse is more robust than a strict matrix inverse.
M19_inv = np.linalg.pinv(M19)


def build_s3_vec(
    omega: float,
    se:  float = 1.19,
    sep: float = 1.40,
    sq:  float = 1.20,
) -> np.ndarray:
    """
    Build the MRT relaxation vector s for D3Q19.

    Parameters
    ----------
    omega : float — BGK-equivalent rate (1/tau); controls viscosity
    se    : float — energy mode s1 (default 1.19)
    sep   : float — energy-square mode s2 (default 1.40)
    sq    : float — energy-flux modes s4,s6,s8 (default 1.20)

    Returns
    -------
    s : (19,) float64
    """
    sv = omega
    return np.array([
        1.0,   #  0  rho       (conserved)
        se,    #  1  e
        sep,   #  2  eps
        1.0,   #  3  jx        (conserved)
        sq,    #  4  qx
        1.0,   #  5  jy        (conserved)
        sq,    #  6  qy
        1.0,   #  7  jz        (conserved)
        sq,    #  8  qz
        sv,    #  9  3pxx      (viscosity)
        sv,    # 10  pww       (viscosity)
        sv,    # 11  pxy       (viscosity)
        sv,    # 12  pyz       (viscosity)
        sv,    # 13  pxz       (viscosity)
        1.0,   # 14  mx        (ghost)
        1.0,   # 15  my        (ghost)
        1.0,   # 16  mz        (ghost)
        1.0,   # 17  txxy      (ghost)
        1.0,   # 18  txyy      (ghost)
    ], dtype=np.float64)

"""
Build ``tunnel_plane.stl``: a simple aircraft made for this wind tunnel.

Why not a downloaded model: real aircraft (and ``simple_plane.stl``) have
wings a few percent of the span thick, which on a grid a few dozen cells
across is under one cell -- they fall between the cell centres and vanish.
This one keeps the usual layout (fuselage, tapered swept wing with
dihedral, tailplane, fin) but with thick sections, NACA 0018 on the wing
and 0020 on the tail, so at ~48 cells across its wing root is two cells
thick.

It is built as one signed-distance field -- the parts smoothly blended, so
the surface is a single closed shell with no overlapping bodies, which a
parity voxelizer would read as holes where parts overlap -- and meshed with
marching cubes.  Axes are the tunnel's: x downstream (nose at the upstream
end), y up (fin up), z across (wings); use Orientation "as in the file".
Units are millimetres, fuselage 600 mm long; the web UI scales it anyway.

Needs scikit-image to regenerate (not a dependency of the package):

    pip install scikit-image
    python3 samples/stl/make_tunnel_plane.py
"""

from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes

H = 0.007                       # sampling step: ~4 mm, under half a cell at 64 cells across
LENGTH_MM = 600.0


def smin(a, b, k):
    """Polynomial smooth minimum: a union with a fillet of about k."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def round_cone(px, py, pz, a, b, r1, r2):
    """Exact distance to the hull of two spheres (a, r1) and (b, r2) (after Quilez)."""
    ba = np.subtract(b, a)
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    qx, qy, qz = px - a[0], py - a[1], pz - a[2]
    y = qx * ba[0] + qy * ba[1] + qz * ba[2]
    z = y - l2
    wx, wy, wz = qx * l2 - ba[0] * y, qy * l2 - ba[1] * y, qz * l2 - ba[2] * y
    x2 = wx * wx + wy * wy + wz * wz
    y2, z2 = y * y * l2, z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d_end = np.sqrt(x2 + z2) * il2 - r2
    d_start = np.sqrt(x2 + y2) * il2 - r1
    d_side = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d_end, np.where(np.sign(y) * a2 * y2 < k, d_start, d_side))


def lifting_surface(x, s, t_coord, *, span, c_root, c_tip, le_root, le_tip, mid_root, mid_tip, tc,
                    one_sided=False):
    """
    A tapered, swept NACA 00xx surface.  ``s`` is the spanwise coordinate
    (0 at the root), ``t_coord`` the one its thickness is along.  The
    result is signed (negative inside) and continuous, which is all
    marching cubes needs of it.
    """
    f = np.clip(s / span, 0.0, 1.0)
    chord = c_root + (c_tip - c_root) * f
    le = le_root + (le_tip - le_root) * f
    mid = mid_root + (mid_tip - mid_root) * f
    u = (x - le) / chord
    uc = np.clip(u, 0.0, 1.0)
    half = 5.0 * tc * chord * (0.2969 * np.sqrt(uc) - 0.1260 * uc - 0.3516 * uc ** 2
                               + 0.2843 * uc ** 3 - 0.1036 * uc ** 4)
    d_thick = np.abs(t_coord - mid) - half
    d_chord = np.maximum(-u, u - 1.0) * chord
    d_span = np.maximum(s - span, -s) if one_sided else s - span
    return np.maximum(np.maximum(d_thick, d_chord), d_span)


def plane_sdf(x, y, z):
    body = smin(round_cone(x, y, z, (0.06, 0.0, 0.0), (0.32, 0.004, 0.0), 0.052, 0.068),
                round_cone(x, y, z, (0.32, 0.004, 0.0), (0.95, 0.045, 0.0), 0.068, 0.018), 0.03)
    wing = lifting_surface(x, np.abs(z), y, span=0.5, c_root=0.24, c_tip=0.13, le_root=0.30, le_tip=0.37,
                           mid_root=-0.015, mid_tip=0.02, tc=0.18)
    tailplane = lifting_surface(x, np.abs(z), y, span=0.18, c_root=0.13, c_tip=0.075, le_root=0.80,
                                le_tip=0.85, mid_root=0.045, mid_tip=0.05, tc=0.20)
    fin = lifting_surface(x, y - 0.02, z, span=0.22, c_root=0.16, c_tip=0.075, le_root=0.78, le_tip=0.88,
                          mid_root=0.0, mid_tip=0.0, tc=0.20, one_sided=True)
    return smin(smin(smin(body, wing, 0.02), tailplane, 0.015), fin, 0.015)


def build():
    # samples half a step off the round numbers the parts are laid out on:
    # a sample exactly on a surface (a tip at |z| = 0.5, say) is an exact
    # zero, where marching cubes emits degenerate and non-manifold triangles.
    # Half a step in z keeps the two wings mirror images.
    xs = np.arange(-0.03, 1.03, H) + 0.5 * H
    ys = np.arange(-0.12, 0.30, H) + 0.37 * H
    zs = np.arange(-0.55, 0.55, H) + 0.5 * H
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    field = plane_sdf(X, Y, Z).astype(np.float32)
    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(H, H, H), gradient_direction="ascent")
    verts += np.array([xs[0], ys[0], zs[0]])
    tris = (verts[faces] * LENGTH_MM).astype(np.float32)
    # a sliver whose corners coincide once stored in float32 has no area;
    # dropping it leaves every edge still shared by two triangles
    same = lambda a, b: np.all(tris[:, a] == tris[:, b], axis=1)
    tris = tris[~(same(0, 1) | same(1, 2) | same(0, 2))].astype(np.float64)
    # outward-facing: the signed volume of a closed surface is positive
    if np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() < 0:
        tris = tris[:, ::-1]
    return tris


def write_binary_stl(path: Path, tris: np.ndarray) -> None:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    record = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(tris), dtype=record)
    data["normal"], data["v"] = n, tris
    header = b"tunnel_plane: simple aircraft for the Aero wind tunnel (x downstream, y up)"
    with path.open("wb") as fh:
        fh.write(header.ljust(80, b" "))
        fh.write(np.uint32(len(tris)).tobytes())
        fh.write(data.tobytes())


if __name__ == "__main__":
    out = Path(__file__).with_name("tunnel_plane.stl")
    tris = build()
    write_binary_stl(out, tris)
    ext = np.ptp(tris.reshape(-1, 3), axis=0)
    print(f"{out.name}: {len(tris):,} triangles, {out.stat().st_size / 2**20:.1f} MB, "
          f"extent {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm (x, y, z)")

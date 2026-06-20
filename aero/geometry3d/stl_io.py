"""Minimal STL reader (ASCII and binary) — numpy only."""

from __future__ import annotations

import pathlib
import struct
from typing import Tuple

import numpy as np


def load_stl_triangles(path: str) -> np.ndarray:
    """
    Load an STL mesh as triangle vertex arrays.

    Returns
    -------
    triangles : ndarray, shape (T, 3, 3)
        Each row is three (x, y, z) vertices.
    """
    data = pathlib.Path(path).read_bytes()
    if _looks_like_binary_stl(data):
        return _load_stl_binary(data)
    if data[:5].lower().startswith(b"solid"):
        return _load_stl_ascii(data)
    return _load_stl_binary(data)


def _looks_like_binary_stl(data: bytes) -> bool:
    """True when byte length matches the binary STL layout."""
    if len(data) < 84:
        return False
    tri_count = struct.unpack_from("<I", data, 80)[0]
    return len(data) == 84 + tri_count * 50


def _load_stl_binary(data: bytes) -> np.ndarray:
    if len(data) < 84:
        raise ValueError("STL file too small to be valid binary STL")
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if len(data) < expected:
        raise ValueError("Binary STL truncated")

    tris = np.empty((tri_count, 3, 3), dtype=np.float64)
    offset = 84
    for i in range(tri_count):
        vals = struct.unpack_from("<12f", data, offset)
        tris[i, 0] = vals[3:6]
        tris[i, 1] = vals[6:9]
        tris[i, 2] = vals[9:12]
        offset += 50
    return tris


def _load_stl_ascii(data: bytes) -> np.ndarray:
    text = data.decode("utf-8", errors="replace").splitlines()
    tris = []
    verts = []
    for line in text:
        parts = line.strip().split()
        if not parts:
            continue
        tag = parts[0].lower()
        if tag == "vertex" and len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(verts) == 3:
                tris.append(verts)
                verts = []
    if not tris:
        raise ValueError("No triangles found in ASCII STL")
    return np.asarray(tris, dtype=np.float64)


def triangle_bounds(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) over all triangle vertices."""
    mins = triangles.reshape(-1, 3).min(axis=0)
    maxs = triangles.reshape(-1, 3).max(axis=0)
    return mins, maxs

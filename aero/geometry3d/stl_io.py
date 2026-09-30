"""Minimal STL reader (ASCII and binary) — numpy only."""

from __future__ import annotations

import pathlib
import re
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
    return parse_stl(pathlib.Path(path).read_bytes())


def parse_stl(data: bytes) -> np.ndarray:
    """The triangles of an STL file's contents (ASCII or binary), shape (T, 3, 3)."""
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

    # 50 bytes a triangle: normal, three vertices, attribute count
    record = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data, dtype=record, count=tri_count, offset=84)["v"].astype(np.float64)


_ASCII_VERTEX = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)


def _load_stl_ascii(data: bytes) -> np.ndarray:
    verts = _ASCII_VERTEX.findall(data)
    n = len(verts) // 3 * 3                   # a trailing partial triangle is dropped
    if n == 0:
        raise ValueError("No triangles found in ASCII STL")
    try:
        return np.array(verts[:n], dtype=np.float64).reshape(-1, 3, 3)
    except ValueError as exc:
        raise ValueError(f"Unreadable vertex in ASCII STL: {exc}") from None


def triangle_bounds(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) over all triangle vertices."""
    mins = triangles.reshape(-1, 3).min(axis=0)
    maxs = triangles.reshape(-1, 3).max(axis=0)
    return mins, maxs

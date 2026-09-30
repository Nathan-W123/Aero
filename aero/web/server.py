"""
Local web front end for the Aero solver.

A browser UI you run on your own machine::

    python3 webui.py          # then open http://localhost:8017

Built on ``http.server`` from the standard library rather than Flask or
FastAPI, deliberately: the solver's only hard dependencies are NumPy and
Matplotlib, and a web UI is not a good reason to add a third.  The whole
server is one file with no install step beyond the package itself.

Why this exists alongside the Qt app
------------------------------------
The desktop GUI needs PySide6 and a display server, which rules it out over
SSH, in a container, or on a headless box -- exactly where a long solve wants
to run.  This front end needs a browser and nothing else.

How a run works
---------------
Each run gets a :class:`Job` executing on its own thread.  The solver is
driven in *chunks* rather than one long call: after every chunk the worker
checks a cancel flag, samples the coefficient history, and renders a preview
frame.  ``Solver.run`` accumulates ``step_count`` and the coefficient
histories across calls, so chunking costs nothing and buys both live progress
and a stop button that actually stops.

The options the form offers are read from the solver's own tables at import
(:data:`COLLISIONS`, lattice names, geometry classes), so this UI cannot drift
out of step with the solver the way a hand-maintained list does.
"""

from __future__ import annotations

import functools
import io
import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

import matplotlib
matplotlib.use("Agg")                      # no display server anywhere near this
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ..benchmarks import (
    build_uncertainty_report, literature_cd_range, mean_uncertainty,
    reference_length_cells, schiller_naumann_cd, sphere_expected_cd,
)
from ..lbm.lattice3d import LATTICES
from ..lbm.physics import base_nu_from_omega

_HERE = Path(__file__).resolve().parent

#: Collision operators the solvers accept.  Kept beside the solver's own
#: validation list; :func:`_self_check` asserts they still agree.
COLLISIONS = ["bgk", "trt", "mrt", "regularized"]
LATTICES_3D = sorted(LATTICES)
BACKENDS = ["auto", "numpy", "numba"]
SHAPES_2D = ["cylinder", "rectangle"]
SHAPES_3D = ["sphere", "box", "cylinder"]
WALL_BCS = ["slip", "noslip"]
OUTLET_BCS = ["convective", "zerogradient"]


def _self_check() -> None:
    """Fail loudly at import if the option lists have drifted from the solver."""
    from ..lbm.solver3d import Solver3D
    import inspect

    src = inspect.getsource(Solver3D.__init__)
    for name in COLLISIONS:
        if f'"{name}"' not in src:
            raise RuntimeError(
                f"collision {name!r} is offered by the web UI but the solver "
                f"no longer lists it — update aero/web/server.py"
            )


_self_check()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    params: Dict[str, Any]
    state: str = "queued"                  # queued | running | done | error | stopped
    step: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    preview_png: Optional[bytes] = None
    figures: Dict[str, bytes] = field(default_factory=dict)
    figure_step: int = -1
    #: packed geometry (solid mask) and the latest downsampled velocity field,
    #: for the browser-side 3D viewer -- see _pack for the wire format
    geometry: Optional[bytes] = None
    field_bytes: Optional[bytes] = None
    field_step: int = -1
    #: Cd and Cl at every step (entry i is step i + 1), for the convergence
    #: chart to zoom into; replaced whole after each chunk, so a reader always
    #: sees a matching pair
    series: Tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.zeros(0, np.float32), np.zeros(0, np.float32)))
    #: replay frames, (step, vorticity scale, wake bytes, vorticity bytes);
    #: see _record_frame
    frames: List[Tuple[int, float, np.ndarray, np.ndarray]] = field(default_factory=list)
    frame_shape: List[int] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event)

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "step": self.step,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "history": self.history,
            "result": self.result,
            "has_preview": self.preview_png is not None,
            "figures": sorted(self.figures),
            "figure_step": self.figure_step,
            "has_geometry": self.geometry is not None,
            "field_step": self.field_step,
            "frame_step": self.frames[-1][0] if self.frames else -1,
            "params": self.params,
        }


JOBS: Dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Building a case
# ---------------------------------------------------------------------------

def _f(params, key, default):
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _i(params, key, default):
    try:
        return int(float(params.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


#: Relaxation rate above which a run is refused.  The strict limit is 2 (where
#: tau reaches 1/2 and the viscosity vanishes), but that bound is useless as a
#: guard: Re = 1e5 on a coarse grid lands at omega = 1.99995, passes a naive
#: ``omega < 2`` check, and then diverges a thousand steps in.  Below roughly
#: tau = 0.505 nothing survives contact with a real obstacle, so refuse there
#: and say why.
OMEGA_CEILING = 1.98


def _check_omega(omega: float, re: float) -> None:
    if not 0.0 < omega < 2.0:
        raise ValueError(
            f"This case gives omega={omega:.4f}, outside the valid range (0, 2). "
            "Lower Re, raise u0, or use a finer grid."
        )
    if omega > OMEGA_CEILING:
        raise ValueError(
            f"This case gives omega={omega:.5f} (tau={1/omega:.5f}), too close to "
            f"the tau=1/2 singularity to stay stable — Re={re:g} is too high for "
            "this grid. Raise the grid resolution, raise u0, or lower Re."
        )


def _case(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and check a case's parameters, allocating nothing.

    Returns the derived numbers (reference length, relaxation rate, grid)
    and a ``make`` callable for the geometry; raises ValueError with a
    message for the user on a case the solver would refuse.  Shared by
    :func:`_build_solver` and the pre-run checks, which call it on every
    edit of the form.
    """
    mode = p.get("mode", "2d")
    re = _f(p, "re", 100.0)
    u0 = _f(p, "u0", 0.05)
    if re <= 0:
        raise ValueError("Reynolds number must be positive.")
    if not (0.0 < u0 <= 0.2):
        raise ValueError("u0 must be in (0, 0.2] — beyond that the flow is too compressible.")

    if mode == "2d":
        from ..geometry.cylinder import Cylinder
        from ..geometry.rectangle import Rectangle

        ny, nx = _i(p, "ny", 200), _i(p, "nx", 400)
        shape = p.get("shape", "cylinder")
        if shape == "rectangle":
            w, h = _f(p, "width", 40.0), _f(p, "height", 20.0)
            D, label, make = h, f"rectangle {w:g}x{h:g}", (lambda: Rectangle(width=w, height=h))
        else:
            r = _f(p, "radius", 20.0)
            D, label, make = 2.0 * r, f"cylinder r={r:g}", (lambda: Cylinder(radius=r))
        if D >= ny:
            raise ValueError("The body is taller than the domain.")
        grid = (ny, nx)
    else:
        from ..geometry3d.box import Box
        from ..geometry3d.cylinder3d import Cylinder3D
        from ..geometry3d.sphere import Sphere

        nz, ny, nx = _i(p, "nz", 48), _i(p, "ny", 48), _i(p, "nx", 96)
        shape = p.get("shape", "sphere")
        if shape == "box":
            w, h, d = _f(p, "width", 10.0), _f(p, "height", 10.0), _f(p, "depth", 10.0)
            D, label, make = h, f"box {w:g}x{h:g}x{d:g}", (lambda: Box(width=w, height=h, depth=d))
        elif shape == "cylinder":
            r, L = _f(p, "radius", 8.0), _f(p, "length", 24.0)
            D, label, make = 2.0 * r, f"cylinder r={r:g}", (lambda: Cylinder3D(radius=r, length=L))
        else:
            r = _f(p, "radius", 7.0)
            D, label, make = 2.0 * r, f"sphere r={r:g}", (lambda: Sphere(radius=r))
        if D >= ny or (shape == "sphere" and D >= nz):
            raise ValueError("The body is larger than the tunnel's cross-section.")
        grid = (nz, ny, nx)

    nu = u0 * D / re
    omega = 1.0 / (3.0 * nu + 0.5)
    _check_omega(omega, re)
    if mode == "3d" and p.get("lattice", "d3q19") != "d3q19" and p.get("collision", "bgk") == "mrt":
        raise ValueError("MRT is implemented for D3Q19 only — pick another collision operator.")
    return {"mode": mode, "shape": shape, "D": D, "label": label, "make": make,
            "grid": grid, "re": re, "u0": u0, "nu": nu, "omega": omega}


def _build_solver(p: Dict[str, Any]):
    """Return (solver, geometry_label). Raises ValueError on a bad case."""
    case = _case(p)
    common = dict(
        u0=case["u0"],
        backend=p.get("backend", "auto"),
        collision=p.get("collision", "bgk"),
        wall_bc=p.get("wall_bc", "slip"),
        outlet_bc=p.get("outlet_bc", "convective"),
        les=bool(p.get("les")),
        les_cs=_f(p, "les_cs", 0.16),
        inlet_perturbation=_f(p, "inlet_perturbation", 0.0),
    )
    geom, D, omega = case["make"](), case["D"], case["omega"]
    if case["mode"] == "2d":
        from ..lbm.solver import Solver

        ny, nx = case["grid"]
        solid = geom.mark_solid(ny, nx)
        return Solver(Ny=ny, Nx=nx, solid=solid, omega=omega, D=D, **common), case["label"]

    from ..lbm.solver3d import Solver3D

    nz, ny, nx = case["grid"]
    solid = geom.mark_solid(nz, ny, nx)
    return Solver3D(Nz=nz, Ny=ny, Nx=nx, solid=solid, omega=omega, D=D,
                    lattice=p.get("lattice", "d3q19"), ref_area=geom.reference_area(),
                    **common), case["label"]


#: The pre-run stability check warns above this relaxation rate: the grid
#: then resolves the viscosity by only a few percent of a cell, and whether a
#: run survives depends on the collision operator and the body.  Measured
#: with the regularized inlet: BGK holds on the default 2D rectangle up to
#: omega = 1.942 (Re 200) and on the default sphere at 1.919.  (The solver
#: refuses outright above OMEGA_CEILING.)
OMEGA_WARN = 1.95
#: ... and above this inlet speed, where compressibility errors (~Ma^2) pass 3%.
U0_WARN = 0.1

#: The uncertainty-budget checks that depend only on the settings.
PREFLIGHT_CHECKS = ("blockage", "domain_length", "discretization", "bc_sensitivity")


def _stability_check(p: Dict[str, Any], case: Optional[Dict[str, Any]], error: Optional[str]):
    u0 = _f(p, "u0", 0.05)
    ma = u0 * np.sqrt(3.0)
    if case is None:
        return {"status": "fail", "short": "cannot run as set", "message": error or ""}
    omega = case["omega"]
    short = f"ω = {omega:.3f} · Ma = {ma:.2f}"
    notes = []
    if omega > OMEGA_WARN:
        notes.append(
            f"ω = {omega:.3f} (τ = {1 / omega:.3f}) is close to 2, where the grid barely "
            "resolves the viscosity. If the run diverges, use a finer grid (a larger body "
            "in cells), a lower Re or u0, or another collision operator.")
    if u0 > U0_WARN:
        notes.append(f"Ma = {ma:.2f}: compressibility errors grow as Ma² (~{ma * ma * 100:.0f}% here); "
                     "a lower u0 is closer to incompressible, at more steps for the same flow time.")
    if notes:
        return {"status": "warn", "short": short, "message": " ".join(notes)}
    return {"status": "pass", "short": short,
            "message": f"ω = {omega:.3f} (τ = {1 / omega:.3f}) and Ma = {ma:.2f} are well inside "
                       "the stable, nearly incompressible range."}


_NUMERIC_KEYS = frozenset({"re", "u0", "nx", "ny", "nz", "radius", "width", "height", "depth",
                           "length", "steps", "inlet_perturbation", "les_cs"})
#: The numbers :func:`_case` falls back on, for the checks' benefit.
_FORM_DEFAULTS = {
    "2d": {"re": 100.0, "u0": 0.05, "nx": 400, "ny": 200, "radius": 20.0, "width": 40.0, "height": 20.0},
    "3d": {"re": 100.0, "u0": 0.05, "nx": 96, "ny": 48, "nz": 48, "radius": 7.0, "width": 10.0,
           "height": 10.0, "depth": 10.0, "length": 24.0},
}


def _is_number(v: Any) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _preflight(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    What can be said about a case before it runs: the checks that depend only
    on its settings -- blockage, domain length, resolution, boundary
    conditions, stability -- and the drag it should give.  Allocates nothing,
    so the page can ask on every edit.
    """
    # a form mid-edit sends "" or "1e"; read those as the defaults, so the
    # checks stay meaningful rather than the whole list failing
    mode = p.get("mode", "2d")
    p = {**_FORM_DEFAULTS[mode if mode in _FORM_DEFAULTS else "2d"],
         **{k: v for k, v in p.items() if k not in _NUMERIC_KEYS or _is_number(v)}}
    shape = p.get("shape", "cylinder" if mode == "2d" else "sphere")
    case, error = None, None
    try:
        case = _case(p)
    except ValueError as exc:
        error = str(exc)
    checks: Dict[str, Any] = {"stability": _stability_check(p, case, error)}
    try:
        rep = build_uncertainty_report(mode=mode, shape=shape, params=dict(p), result={})
        for name in PREFLIGHT_CHECKS:
            comp = rep.components.get(name) or {}
            if comp.get("status") not in (None, "n/a"):
                checks[name] = {"status": comp["status"], "message": comp.get("message", ""),
                                "short": _short_label(name, comp.get("value"), mode, shape,
                                                      _f(p, "re", 100.0))}
    except Exception as exc:                 # a half-typed form must not break the page
        checks["budget"] = {"status": "warn", "short": "checks unavailable", "message": str(exc)}
    out: Dict[str, Any] = {"uncertainty": checks, "error": error}
    try:
        out["reference"] = _reference(p, mode, None)
    except Exception:
        out["reference"] = None
    if case is not None:
        out["setup"] = {"label": case["label"], "omega": _num(case["omega"]), "nu": _num(case["nu"], 6),
                        "grid": list(case["grid"]), "cells": int(np.prod(case["grid"]))}
    return out


# ---------------------------------------------------------------------------
# Binary payloads for the browser-side 3D viewer
# ---------------------------------------------------------------------------

def _pack(header: Dict[str, Any], *arrays: np.ndarray) -> bytes:
    """
    One JSON header line, padded so the data starts 8-byte aligned, then the
    arrays back to back.  ``header["parts"]`` names each array with its dtype
    and element count, so the client can walk the buffer with typed views and
    no copies.  JSON for the numbers would be ~5x the bytes and a parse of a
    few hundred thousand floats on every refresh.
    """
    parts = []
    blobs = []
    for name, arr in zip(header["names"], arrays):
        arr = np.ascontiguousarray(arr)
        parts.append({"name": name, "dtype": str(arr.dtype), "count": int(arr.size)})
        blobs.append(arr.tobytes())
    h = dict(header)
    h.pop("names")
    h["parts"] = parts
    head = json.dumps(h).encode()
    pad = (-(len(head) + 1)) % 8
    return head + b" " * pad + b"\n" + b"".join(blobs)


def _series(cd_hist, cl_hist) -> Tuple[np.ndarray, np.ndarray]:
    """Per-step Cd and Cl as float32, the same length (Cl padded with NaN)."""
    cd = np.asarray(cd_hist, dtype=np.float32)
    cl = np.full(cd.shape, np.nan, dtype=np.float32)
    n = min(len(cl_hist), cd.size)
    cl[:n] = np.asarray(cl_hist[:n], dtype=np.float32)
    return cd, cl


def _series_payload(job: "Job", start: int) -> bytes:
    """The per-step coefficients from step ``start + 1`` on (entry i is step i + 1)."""
    cd, cl = job.series
    start = min(max(int(start), 0), cd.size)
    return _pack({"from": start, "total": int(cd.size), "names": ["cd", "cl"]}, cd[start:], cl[start:])


def _geometry_payload(solid: np.ndarray) -> bytes:
    if solid.ndim == 2:
        solid = solid[None]
    return _pack({"shape": list(solid.shape), "names": ["solid"]},
                 solid.astype(np.uint8))


#: Voxels sent to the browser's volume renderer per refresh; larger grids are
#: strided down to about this.  Six bytes a voxel, so ~12 MB at most.
FIELD_VOXELS_MAX = 2_000_000

#: Velocity range carried in a byte for the viewer's animation, in u0.
VEL_FULL_SCALE = 1.5

#: Disturbance speed |u - U_inf| at the top of the colour scale, in units of u0.
WAKE_FULL_SCALE = 1.2


def _field_payload(solver, mode: str) -> bytes:
    header, *volumes = _field_volumes(solver, mode)
    return _pack(header, *volumes)


def _field_volumes(solver, mode: str):
    """
    Two scalar volumes for the viewer's volume renderer, one byte a voxel,
    and the velocity that animates them.

    ``wake`` is |u - U_inf| / u0: how much the body disturbs the flow.  It is
    zero in the free stream, so the free stream renders as clear space and the
    wake as a plume; full scale is WAKE_FULL_SCALE u0.  ``vorticity`` is
    |curl u| D / u0, scaled to its 99.5th percentile over the fluid so the
    shear layers show at any Reynolds number (the scale is in the header, for
    the legend).  Inside the body both are zero: the body is drawn from the
    geometry, as a surface.  ``vel`` is u/u0 as RGBA bytes (A unused),
    ``(u/u0) / (2 VEL_FULL_SCALE) + 1/2``; the viewer carries its smoke
    texture along it, so the motion on screen is the computed flow's.
    """
    if mode == "2d":
        _, ux, uy = solver.macroscopic()
        ux, uy = ux[None], uy[None]
        uz = np.zeros_like(ux)
    else:
        _, ux, uy, uz = solver.macroscopic()
    solid = solver.solid.get() if hasattr(solver.solid, "get") else solver.solid
    solid = np.asarray(solid, dtype=bool).reshape(ux.shape)
    nz, ny, nx = ux.shape
    f = max(1, int(np.ceil((ux.size / FIELD_VOXELS_MAX) ** (1.0 / 3.0))))
    sl = (slice(None, None, f),) * 3
    ux, uy, uz = (np.nan_to_num(a[sl].astype(np.float32)) for a in (ux, uy, uz))
    solid = solid[sl]
    u0 = float(getattr(solver, "u0", 0.05)) or 0.05
    d_ref = float(getattr(solver, "D", 1.0)) or 1.0

    def ddx(a, axis):                                  # axes are (z, y, x)
        return np.gradient(a, float(f), axis=axis) if a.shape[axis] > 1 else np.zeros_like(a)

    wake = np.sqrt((ux - u0) ** 2 + uy * uy + uz * uz) / u0
    wx = ddx(uz, 1) - ddx(uy, 0)
    wy = ddx(ux, 0) - ddx(uz, 2)
    wz = ddx(uy, 2) - ddx(ux, 1)
    vort = np.sqrt(wx * wx + wy * wy + wz * wz) * (d_ref / u0)
    wake[solid] = 0.0
    vort[solid] = 0.0
    fluid = vort[~solid]
    v_scale = max(float(np.percentile(fluid, 99.5)) if fluid.size else 1.0, 1e-6)

    def byte(a, full_scale):
        return np.clip(a * (255.0 / full_scale) + 0.5, 0.0, 255.0).astype(np.uint8)

    vel = np.zeros(wake.shape + (4,), dtype=np.float32)
    for i, comp in enumerate((ux, uy, uz)):
        vel[..., i] = np.where(solid, 0.0, comp / u0) / (2.0 * VEL_FULL_SCALE) + 0.5
    return (
        {"shape": list(wake.shape), "factor": f, "full": [nz, ny, nx], "u0": u0,
         "scales": {"wake": WAKE_FULL_SCALE, "vorticity": _num(v_scale, 3),
                    "vel": VEL_FULL_SCALE},
         "names": ["wake", "vorticity", "vel"]},
        byte(wake, WAKE_FULL_SCALE), byte(vort, v_scale), byte(vel, 1.0),
    )


#: The viewer replays the wake and vorticity of a run as a time-lapse, one
#: frame a chunk.  Frames cost two bytes a voxel; past this many bytes a run
#: drops every other frame (keeping the first and the latest), and only the
#: newest REPLAY_RUNS_KEPT runs keep theirs.
REPLAY_BYTES_MAX = 48 * 2 ** 20
REPLAY_RUNS_KEPT = 3


def _record_frame(job: "Job", step: int, header: Dict[str, Any], wake: np.ndarray,
                  vort: np.ndarray) -> None:
    frames = job.frames + [(int(step), float(header["scales"]["vorticity"] or 1.0), wake, vort)]
    if sum(w.nbytes + v.nbytes for _, _, w, v in frames) > REPLAY_BYTES_MAX and len(frames) > 2:
        frames = frames[:-1:2] + [frames[-1]]
    job.frame_shape = list(header["shape"])
    job.frames = frames                       # replaced whole: readers see one list or the other


def _frames_payload(job: "Job", after: int) -> bytes:
    """The replay frames after step ``after``: steps, vorticity scales, then the volumes."""
    frames = [fr for fr in job.frames if fr[0] > after]
    empty = np.zeros(0, np.uint8)
    return _pack({"shape": job.frame_shape, "steps": [fr[0] for fr in frames],
                  "scales": [_num(fr[1], 4) for fr in frames], "names": ["wake", "vorticity"]},
                 np.concatenate([fr[2].ravel() for fr in frames]) if frames else empty,
                 np.concatenate([fr[3].ravel() for fr in frames]) if frames else empty)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Field figures are drawn at about twice the size the page shows them, so
#: they stay sharp on dense screens; type is sized for the shown size (the plot
#: panes are ~300-440 px wide).
FIELD_FIG_WIDTH = 3.6          # inches
FIELD_FIG_DPI = 200
_BODY_FILL = "#a3adb9"         # the body, as the 3D view draws it
_BODY_EDGE = "#2f3a48"
_FRAME = "#8d9aab"             # the tunnel walls
_TEXT = "#1b2a3d"


@functools.lru_cache(maxsize=None)
def _ui_font() -> str:
    """The page's typeface where Matplotlib can find it, else its default."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in ("Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans")
                 if n in have), "DejaVu Sans")


def _mid_plane(solver, mode: str):
    """``(rho, ux, uy, speed, solid)`` on the plane the figures show."""
    if mode == "2d":
        rho, ux, uy = solver.macroscopic()
        return rho, ux, uy, np.hypot(ux, uy), solver.solid
    rho, ux, uy, uz = solver.macroscopic()
    k = ux.shape[0] // 2
    speed = np.sqrt(ux[k] ** 2 + uy[k] ** 2 + uz[k] ** 2)
    return rho[k], ux[k], uy[k], speed, solver.solid[k]


def _smoothed(mask: np.ndarray, passes: int = 1) -> np.ndarray:
    """A 3x3 box blur of a 0/1 mask, so its 0.5 contour is round, not a staircase."""
    a = mask.astype(float)
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0
    return a


def _extend_into(solid: np.ndarray, a: np.ndarray, passes: int = 3) -> np.ndarray:
    """
    ``a`` with the solid cells near the surface given their fluid neighbours'
    mean, so the field runs smoothly under the drawn outline of the body
    instead of stopping at the staircase of cells.
    """
    a = np.where(solid, np.nan, a)
    for _ in range(passes):
        gap = np.isnan(a)
        if not gap.any():
            break
        p = np.pad(a, 1, mode="edge")
        near = np.stack([p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]])
        known = ~np.isnan(near)
        count = known.sum(axis=0)
        mean = np.where(known, near, 0.0).sum(axis=0) / np.maximum(count, 1)
        a = np.where(gap & (count > 0), mean, a)
    rest = np.isnan(a)
    if rest.any():                           # deep inside: hidden under the body
        a[rest] = np.nanmean(a) if (~rest).any() else 0.0
    return a


def _field_png(solver, mode: str, what: str = "speed", streamlines: bool = False) -> bytes:
    """
    Velocity magnitude, vorticity or pressure over the mid-plane.

    Each is scaled by the free stream -- ``|u|/U∞``, ``ωD/U∞`` and ``Cp`` -- so
    the numbers mean the same on any grid or inlet speed.  The pressure plot
    carries isobars; ``streamlines`` adds them to the velocity plot (for the
    final figure: tracing them costs more than the rest of the figure).
    """
    rho, ux, uy, speed, solid = _mid_plane(solver, mode)
    u0 = float(getattr(solver, "u0", 0.0) or 0.05)
    D = float(getattr(solver, "D", 0.0) or 1.0)
    fluid = ~solid
    has_fluid = bool(fluid.any())

    def symmetric(a):
        lim = float(np.nanpercentile(np.abs(a[fluid]), 99)) if has_fluid else 0.0
        return lim if np.isfinite(lim) and lim > 0 else 1.0

    isobars = None
    if what == "vorticity":
        data = (np.gradient(uy, axis=1) - np.gradient(ux, axis=0)) * (D / u0)
        cmap, label = "RdBu_r", r"$\omega D/U_\infty$"
        vmax = symmetric(data); vmin = -vmax
    elif what == "pressure":
        # against the static pressure just inside the inlet, as a tunnel's
        # upstream tapping would read it; p = rho c_s^2 with c_s^2 = 1/3
        p = rho / 3.0
        tap = fluid[:, 1]
        p_inf = float(np.mean(p[:, 1][tap])) if tap.any() else float(np.mean(p))
        data = (p - p_inf) / (0.5 * (3.0 * p_inf) * u0 ** 2)
        cmap, label = "RdBu_r", r"$C_p$"
        vmax = symmetric(data); vmin = -vmax
        isobars = np.linspace(vmin, vmax, 13)
    else:
        data = speed / u0
        cmap, label = "viridis", r"$|u|/U_\infty$"
        vmin, vmax = 0.0, (float(np.nanmax(data[fluid])) if has_fluid else 1.0)
        if not (np.isfinite(vmax) and vmax > 0):
            vmax = 1.0

    ny, nx = data.shape
    font = _ui_font()
    fig = Figure(figsize=(FIELD_FIG_WIDTH, max(1.1, 0.87 * FIELD_FIG_WIDTH * ny / nx)),
                 dpi=FIELD_FIG_DPI, facecolor="#ffffff")
    ax = fig.subplots()
    shown = _extend_into(solid, data) if has_fluid else data
    im = ax.imshow(shown, origin="lower", aspect="equal", interpolation="bilinear",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    if isobars is not None and has_fluid:
        ax.contour(shown, levels=isobars, colors=_TEXT, linewidths=0.3, alpha=0.4)
    if streamlines and what == "speed" and has_fluid:
        d = 1.25
        ax.streamplot(np.arange(nx), np.arange(ny), np.where(solid, 0.0, ux), np.where(solid, 0.0, uy),
                      density=(d, max(d * ny / nx, 0.3)), color="#ffffff", linewidth=0.4,
                      arrowsize=0.45, zorder=2)
    if solid.any():
        # the body as a filled, smoothed outline, as the 3D view draws it
        mask = _smoothed(solid)
        ax.contourf(mask, levels=[0.5, 2.0], colors=[_BODY_FILL], zorder=3)
        ax.contour(mask, levels=[0.5], colors=[_BODY_EDGE], linewidths=0.6, zorder=4)
    ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(_FRAME); side.set_linewidth(0.6)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025, aspect=13)
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(_FRAME)
    cb.locator = MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1.5, colors=_TEXT)
    for tick in cb.ax.get_yticklabels():
        tick.set_fontfamily(font)
    cb.ax.set_title(label, fontsize=8, color=_TEXT, pad=4, loc="left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIELD_FIG_DPI, bbox_inches="tight", pad_inches=0.03,
                facecolor="#ffffff")
    return buf.getvalue()


def _uncertainty(p: Dict[str, Any], mode: str, solver, window: int) -> Dict[str, Any]:
    """The run's uncertainty budget, reduced to status + message per check."""
    shape = p.get("shape", "cylinder" if mode == "2d" else "sphere")
    result = {
        "Cd_history": list(getattr(solver, "Cd_history", [])),
        "Cl_history" if mode == "2d" else "Cly_history": _lift_history(solver),
        "analysis_window": window,
    }
    try:
        rep = build_uncertainty_report(mode=mode, shape=shape, params=dict(p), result=result)
    except Exception as exc:                        # never let the budget sink a finished run
        return {"error": str(exc)}
    out = {}
    for name, comp in rep.components.items():
        if comp.get("status") == "n/a":
            continue
        out[name] = {"status": comp["status"], "message": comp.get("message", ""),
                     "short": _short_label(name, comp.get("value"), mode, shape, _f(p, "re", 100.0))}
    return out


def _short_label(name: str, v: Any, mode: str, shape: str, re: float = 100.0) -> str:
    """A few words for the chip, so the finding is readable without hovering."""
    try:
        if name == "statistical":
            s = f"±{v['cd_relative_sem95']*100:.1f}% stat. · {v['n_eff']:.0f} indep. samples"
            return s + (" · still drifting" if v.get("stationary") is False else "")
        if name == "blockage":
            s = f"blockage {v*100:.0f}%"
            if mode == "3d" and shape == "sphere":
                from ..benchmarks import sphere_confinement_factor
                s += f" · ~+{(sphere_confinement_factor(v, re)-1)*100:.0f}% on Cd"
            return s
        if name == "domain_length":
            return f"{v['upstream_D']:.1f}D upstream · {v['downstream_D']:.1f}D downstream"
        if name == "discretization":
            if isinstance(v, dict) and "cells_across_body" in v:
                return (f"{v['cells_across_body']:.0f} cells across · "
                        f"boundary layer ≈ {v['cells_across_boundary_layer']:.1f} cells")
            return "grid study"
        if name == "bc_sensitivity":
            w = (v or {}).get("warnings") or []
            return "boundary conditions ok" if not w else f"{len(w)} boundary-condition note{'s' if len(w) > 1 else ''}"
    except Exception:
        pass
    return name.replace("_", " ")


def _reference(p: Dict[str, Any], mode: str, cd: Optional[float]) -> Optional[Dict[str, Any]]:
    """
    What the drag *should* be for this case, including the tunnel's confinement.

    The comparison a user makes by eye -- against the unconfined textbook value
    -- is the wrong one for a confined tunnel, and a statistical error bar can
    never cover a systematic bias.  This states the expected value for the run
    as configured, so the two can be compared honestly.
    """
    shape = p.get("shape", "cylinder" if mode == "2d" else "sphere")
    re = _f(p, "re", 100.0)
    d = reference_length_cells(mode, shape, p)
    if mode == "3d":
        span = min(_f(p, "ny", 48.0), _f(p, "nz", 48.0))
        blockage = d / max(span, 1.0)
    else:
        blockage = d / max(_f(p, "ny", 200.0), 1.0)
    band = literature_cd_range(mode, shape, re, blockage if mode == "3d" else None)
    if band is None:
        return None
    lo, hi, note = band
    out = {"band": [_num(lo, 3), _num(hi, 3)], "note": note, "blockage": _num(blockage, 3)}
    if mode == "3d" and shape == "sphere":
        sn, conf, _ = sphere_expected_cd(re, blockage)
        out.update({"unconfined": _num(sn, 3), "expected": _num(conf, 3)})
    if cd is not None:
        out["status"] = "pass" if lo <= cd <= hi else "fail"
    return out


def _lift_history(solver) -> List[float]:
    """2D keeps Cl_history; 3D keeps Cly (wall-normal) and Clz (spanwise)."""
    for name in ("Cl_history", "Cly_history"):
        h = getattr(solver, name, None)
        if h is not None:
            return list(h)
    return []


def _num(x, nd: int = 5) -> Optional[float]:
    """
    A float the browser can parse, or None.

    Python's json.dumps writes NaN/Infinity as bare tokens, which are not JSON;
    ``response.json()`` in the browser throws on them and the page silently
    stops updating.  Every number that reaches a payload goes through here.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _run_job(job: Job) -> None:
    p = job.params
    mode = p.get("mode", "2d")
    try:
        job.state = "running"
        job.message = "building the case"
        solver, label = _build_solver(p)
        nu = base_nu_from_omega(solver.omega)
        job.result["setup"] = {
            "label": label,
            "omega": round(float(solver.omega), 5),
            "nu": round(float(nu), 6),
            "tau": round(1.0 / float(solver.omega), 4),
            "grid": (list(solver.solid.shape)),
            # the backend that was actually resolved ("auto" is not an answer)
            "backend": getattr(solver, "backend", p.get("backend", "auto")),
            "cells": int(np.prod(solver.solid.shape)),
            "solid_cells": int(solver.solid.sum()),
            # frontal area the coefficients are normalised by; 3D only
            "ref_area": _num(getattr(solver, "ref_area", None), 2),
        }

        job.geometry = _geometry_payload(solver.solid)
        header, *volumes = _field_volumes(solver, mode)
        job.field_bytes = _pack(header, *volumes)
        job.field_step = 0
        _record_frame(job, 0, header, volumes[0], volumes[1])

        total = job.total = max(_i(p, "steps", 3000), 1)
        chunk = max(total // 60, 50)
        done = 0
        t0 = time.time()
        while done < total:
            if job._cancel.is_set():
                job.state = "stopped"
                job.message = f"stopped at step {done}"
                break
            n = min(chunk, total - done)
            solver.run(steps=n, check_every=10 ** 9, verbose=False)
            done += n
            job.step = done

            cd_hist = list(getattr(solver, "Cd_history", []))
            cl_hist = _lift_history(solver)
            cd = float(cd_hist[-1]) if cd_hist else float("nan")
            cl = float(cl_hist[-1]) if cl_hist else float("nan")
            # A blown-up run can stay finite for a while (Cd ~ 1e40) before
            # it reaches NaN; no physical drag coefficient is anywhere near 1e4.
            if not np.isfinite(cd) or abs(cd) > 1e4:
                raise RuntimeError(
                    f"the solution diverged at step {done} — lower Re or u0, raise the "
                    f"grid resolution (more cells across the body), or try another "
                    f"collision operator"
                )
            job.history.append({"step": done, "cd": _num(cd), "cl": _num(cl)})
            job.series = _series(cd_hist, cl_hist)
            rate = done / max(time.time() - t0, 1e-9)
            mlups = rate * float(np.prod(solver.solid.shape)) / 1e6
            job.message = (f"step {done:,} of {total:,} · {rate:,.0f} steps/s · "
                           f"{mlups:,.1f} MLUPS on {getattr(solver, 'backend', '?')}")
            try:
                header, *volumes = _field_volumes(solver, mode)
                job.field_bytes = _pack(header, *volumes)
                job.field_step = done
                _record_frame(job, done, header, volumes[0], volumes[1])
            except Exception:
                pass
            if len(job.history) % 3 == 1:
                try:
                    for what in ("speed", "vorticity", "pressure"):
                        job.figures[what] = _field_png(solver, mode, what)
                    job.figure_step = done
                except Exception:
                    pass

        if job.state != "stopped":
            job.state = "done"
            job.message = "complete"

        # final numbers over the settled tail
        cd_hist = np.asarray(getattr(solver, "Cd_history", []), dtype=float)
        cl_hist = np.asarray(_lift_history(solver), dtype=float)
        if cd_hist.size:
            n = cd_hist.size
            window = n - n // 2                        # second half: first half is start-up
            st = mean_uncertainty(cd_hist[-window:])
            sl = mean_uncertainty(cl_hist[-window:]) if cl_hist.size else {}
            job.result["coefficients"] = {
                "cd": _num(st["mean"]),
                # sigma is the size of the *fluctuation*, not an uncertainty; the
                # page used to print it as "±", which read as an error bar
                "cd_std": _num(st["sigma"]),
                "cd_sem95": _num(st["sem95"]),
                "cd_n_eff": _num(st["n_eff"], 1),
                "cd_stationary": st["stationary"],
                "cl": _num(sl.get("mean")),
                "cl_std": _num(sl.get("sigma")),
                "cl_sem95": _num(sl.get("sem95")),
                "note": "averaged over the second half of the run",
            }
            job.result["uncertainty"] = _uncertainty(p, mode, solver, window)
            job.result["reference"] = _reference(p, mode, job.result["coefficients"]["cd"])
        for what in ("speed", "vorticity", "pressure"):
            try:
                job.figures[what] = _field_png(solver, mode, what, streamlines=True)
            except Exception:
                pass
        job.figure_step = job.step
        try:
            header, *volumes = _field_volumes(solver, mode)
            job.field_bytes = _pack(header, *volumes)
            job.field_step = job.step
            if not job.frames or job.frames[-1][0] != job.step:
                _record_frame(job, job.step, header, volumes[0], volumes[1])
        except Exception:
            pass
        job.preview_png = job.figures.get(p.get("field", "speed")) or job.preview_png

    except Exception as exc:                       # surfaced verbatim in the UI
        job.state = "error"
        job.error = str(exc) or exc.__class__.__name__
        job.message = "failed"
        traceback.print_exc()
    finally:
        job.finished = time.time()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "AeroWeb/1.0"

    def log_message(self, fmt, *args):             # quieter than the default
        if not any(k in (args[0] if args else "") for k in ("/api/status", "/api/field", "/api/figure")):
            print(f"  {self.address_string()} {fmt % args}")

    # -- helpers --
    def _send(self, code, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

    # -- routes --
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        if u.path in ("/", "/index.html"):
            html = (_HERE / "index.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/api/schema":
            return self._json({
                "collisions": COLLISIONS, "lattices": LATTICES_3D,
                "backends": BACKENDS, "shapes_2d": SHAPES_2D,
                "shapes_3d": SHAPES_3D, "wall_bcs": WALL_BCS,
                "outlet_bcs": OUTLET_BCS,
            })
        if u.path == "/api/status":
            job = JOBS.get((q.get("id") or [""])[0])
            if job is None:
                return self._json({"error": "no such run"}, 404)
            return self._json(job.public())
        if u.path in ("/api/geometry", "/api/field"):
            job = JOBS.get((q.get("id") or [""])[0])
            if job is None:
                return self._json({"error": "no such run"}, 404)
            blob = job.geometry if u.path == "/api/geometry" else job.field_bytes
            if not blob:
                return self._json({"error": "not ready yet"}, 404)
            return self._send(200, blob, "application/octet-stream")
        if u.path == "/api/frames":
            job = JOBS.get((q.get("id") or [""])[0])
            if job is None:
                return self._json({"error": "no such run"}, 404)
            after = (q.get("after") or ["-1"])[0]
            try:
                after = int(after)
            except ValueError:
                after = -1
            return self._send(200, _frames_payload(job, after), "application/octet-stream")
        if u.path == "/api/series":
            job = JOBS.get((q.get("id") or [""])[0])
            if job is None:
                return self._json({"error": "no such run"}, 404)
            start = (q.get("from") or ["0"])[0]
            return self._send(200, _series_payload(job, int(start) if start.isdigit() else 0),
                              "application/octet-stream")
        if u.path == "/api/figure":
            job = JOBS.get((q.get("id") or [""])[0])
            if job is None:
                return self._json({"error": "no such run"}, 404)
            name = (q.get("name") or ["preview"])[0]
            png = job.preview_png if name == "preview" else job.figures.get(name)
            if not png:
                return self._json({"error": "not rendered yet"}, 404)
            return self._send(200, png, "image/png")
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad JSON"}, 400)

        if u.path == "/api/run":
            job = Job(id=uuid.uuid4().hex[:12], params=payload)
            with _JOBS_LOCK:
                JOBS[job.id] = job
                for old in sorted(JOBS.values(), key=lambda j: j.started)[:-12]:
                    JOBS.pop(old.id, None)        # keep the last dozen
                for old in sorted(JOBS.values(), key=lambda j: j.started)[:-REPLAY_RUNS_KEPT]:
                    old.frames = []               # and the replays of the last few
            threading.Thread(target=_run_job, args=(job,), daemon=True).start()
            return self._json({"id": job.id})

        if u.path == "/api/preflight":
            return self._json(_preflight(payload if isinstance(payload, dict) else {}))

        if u.path == "/api/stop":
            job = JOBS.get(payload.get("id", ""))
            if job is None:
                return self._json({"error": "no such run"}, 404)
            job._cancel.set()
            return self._json({"ok": True})

        return self._json({"error": "not found"}, 404)


def serve(host: str = "127.0.0.1", port: int = 8017) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print("\n  Aero CFD — web UI")
    print(f"  Open {url}")
    print("  Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        httpd.server_close()

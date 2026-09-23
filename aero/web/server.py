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

import io
import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

import matplotlib
matplotlib.use("Agg")                      # no display server anywhere near this
import matplotlib.pyplot as plt

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


def _build_solver(p: Dict[str, Any]):
    """Return (solver, geometry_label). Raises ValueError on a bad case."""
    mode = p.get("mode", "2d")
    re = _f(p, "re", 100.0)
    u0 = _f(p, "u0", 0.05)
    if re <= 0:
        raise ValueError("Reynolds number must be positive.")
    if not (0.0 < u0 <= 0.2):
        raise ValueError("u0 must be in (0, 0.2] — beyond that the flow is too compressible.")

    common = dict(
        u0=u0,
        backend=p.get("backend", "auto"),
        collision=p.get("collision", "bgk"),
        wall_bc=p.get("wall_bc", "slip"),
        outlet_bc=p.get("outlet_bc", "convective"),
        les=bool(p.get("les")),
        les_cs=_f(p, "les_cs", 0.16),
        inlet_perturbation=_f(p, "inlet_perturbation", 0.0),
    )

    if mode == "2d":
        from ..geometry.cylinder import Cylinder
        from ..geometry.rectangle import Rectangle
        from ..lbm.solver import Solver

        ny, nx = _i(p, "ny", 200), _i(p, "nx", 400)
        shape = p.get("shape", "cylinder")
        if shape == "rectangle":
            w, h = _f(p, "width", 40.0), _f(p, "height", 20.0)
            geom, D, label = Rectangle(width=w, height=h), h, f"rectangle {w:g}x{h:g}"
        else:
            r = _f(p, "radius", 20.0)
            geom, D, label = Cylinder(radius=r), 2.0 * r, f"cylinder r={r:g}"
        solid = geom.mark_solid(ny, nx)
        if D >= ny:
            raise ValueError("The body is taller than the domain.")
        nu = u0 * D / re
        omega = 1.0 / (3.0 * nu + 0.5)
        _check_omega(omega, re)
        return Solver(Ny=ny, Nx=nx, solid=solid, omega=omega, D=D, **common), label

    from ..geometry3d.box import Box
    from ..geometry3d.cylinder3d import Cylinder3D
    from ..geometry3d.sphere import Sphere
    from ..lbm.solver3d import Solver3D

    nz, ny, nx = _i(p, "nz", 48), _i(p, "ny", 48), _i(p, "nx", 96)
    shape = p.get("shape", "sphere")
    if shape == "box":
        w, h, d = _f(p, "width", 10.0), _f(p, "height", 10.0), _f(p, "depth", 10.0)
        geom, D, label = Box(width=w, height=h, depth=d), h, f"box {w:g}x{h:g}x{d:g}"
    elif shape == "cylinder":
        r, L = _f(p, "radius", 8.0), _f(p, "length", 24.0)
        geom, D, label = Cylinder3D(radius=r, length=L), 2.0 * r, f"cylinder r={r:g}"
    else:
        r = _f(p, "radius", 7.0)
        geom, D, label = Sphere(radius=r), 2.0 * r, f"sphere r={r:g}"
    solid = geom.mark_solid(nz, ny, nx)
    nu = u0 * D / re
    omega = 1.0 / (3.0 * nu + 0.5)
    _check_omega(omega, re)
    lattice = p.get("lattice", "d3q19")
    if lattice != "d3q19" and common["collision"] == "mrt":
        raise ValueError("MRT is implemented for D3Q19 only — pick another collision operator.")
    return Solver3D(Nz=nz, Ny=ny, Nx=nx, solid=solid, omega=omega, D=D,
                    lattice=lattice, **common), label


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _field_png(solver, mode: str, what: str = "speed") -> bytes:
    """Velocity magnitude / vorticity / pressure over the mid-plane."""
    if mode == "2d":
        rho, ux, uy = solver.macroscopic()
        solid = solver.solid
    else:
        rho, ux, uy, uz = solver.macroscopic()
        k = ux.shape[0] // 2
        rho, ux, uy, solid = rho[k], ux[k], uy[k], solver.solid[k]

    if what == "vorticity":
        dvdx = np.gradient(uy, axis=1)
        dudy = np.gradient(ux, axis=0)
        data, cmap, label = dvdx - dudy, "RdBu_r", "vorticity"
        lim = np.nanpercentile(np.abs(data[~solid]), 99) if (~solid).any() else 1.0
        vmin, vmax = -lim, lim
    elif what == "pressure":
        data, cmap, label = (rho - 1.0) / 3.0, "coolwarm", "pressure"
        vals = data[~solid] if (~solid).any() else data
        lim = np.nanpercentile(np.abs(vals), 99) or 1e-6
        vmin, vmax = -lim, lim
    else:
        data, cmap, label = np.sqrt(ux ** 2 + uy ** 2), "viridis", "|u|"
        vmin, vmax = 0.0, float(np.nanmax(data)) or 1.0

    data = np.ma.masked_where(solid, data)
    ny, nx = data.shape
    fig, ax = plt.subplots(figsize=(min(11, nx / 44), min(6, ny / 44)))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear", aspect="equal")
    ax.contour(solid.astype(float), levels=[0.5], colors="#1b1b1b", linewidths=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return _png(fig)


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
            "cells": int(np.prod(solver.solid.shape)),
            "solid_cells": int(solver.solid.sum()),
        }

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
            cl_hist = list(getattr(solver, "Cl_history", []))
            cd = float(cd_hist[-1]) if cd_hist else float("nan")
            cl = float(cl_hist[-1]) if cl_hist else float("nan")
            if not np.isfinite(cd):
                raise RuntimeError(
                    f"the solution diverged at step {done} — lower Re, raise the "
                    f"grid resolution, or try the regularized collision operator"
                )
            job.history.append({"step": done, "cd": round(cd, 5), "cl": round(cl, 5)})
            rate = done / max(time.time() - t0, 1e-9)
            job.message = f"step {done:,} of {total:,} · {rate:,.0f} steps/s"
            if len(job.history) % 3 == 1:
                try:
                    job.preview_png = _field_png(solver, mode, p.get("field", "speed"))
                except Exception:
                    pass

        if job.state != "stopped":
            job.state = "done"
            job.message = "complete"

        # final numbers over the settled tail
        cd_hist = np.asarray(getattr(solver, "Cd_history", []), dtype=float)
        cl_hist = np.asarray(getattr(solver, "Cl_history", []), dtype=float)
        if cd_hist.size:
            tail = cd_hist[max(cd_hist.size // 2, 0):]
            tail_l = cl_hist[max(cl_hist.size // 2, 0):] if cl_hist.size else np.array([np.nan])
            job.result["coefficients"] = {
                "cd": round(float(np.nanmean(tail)), 5),
                "cd_std": round(float(np.nanstd(tail)), 5),
                "cl": round(float(np.nanmean(tail_l)), 5),
                "cl_std": round(float(np.nanstd(tail_l)), 5),
                "note": "averaged over the second half of the run",
            }
        for what in ("speed", "vorticity", "pressure"):
            try:
                job.figures[what] = _field_png(solver, mode, what)
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
        if "/api/status" not in (args[0] if args else ""):
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
        self._send(code, json.dumps(obj).encode(), "application/json")

    # -- routes --
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
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
            threading.Thread(target=_run_job, args=(job,), daemon=True).start()
            return self._json({"id": job.id})

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

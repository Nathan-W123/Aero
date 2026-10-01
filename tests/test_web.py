"""Tests for the local web UI."""

import json
import threading
import time

import numpy as np
import pytest

from aero.web import server as web


# ---------------------------------------------------------------------------
# The option lists must track the solver
# ---------------------------------------------------------------------------

def test_offered_collisions_are_all_accepted_by_the_solver():
    """
    The Qt GUI drifted out of step with the solver — its collision list was
    missing 'regularized' for a week. This pins the web UI against that.
    """
    from aero.lbm.solver import Solver

    solid = np.zeros((16, 24), dtype=bool)
    for name in web.COLLISIONS:
        Solver(Ny=16, Nx=24, solid=solid, omega=1.5, u0=0.05, D=4.0,
               collision=name, backend="numpy")


def test_offered_lattices_are_all_real():
    from aero.lbm.lattice3d import get_lattice3d

    for name in web.LATTICES_3D:
        assert get_lattice3d(name).name == name


def test_self_check_runs_at_import():
    web._self_check()          # raises if the lists have drifted


# ---------------------------------------------------------------------------
# Case building
# ---------------------------------------------------------------------------

def test_builds_a_2d_case():
    solver, label = web._build_solver(
        {"mode": "2d", "shape": "cylinder", "radius": 8, "re": 80,
         "u0": 0.05, "nx": 120, "ny": 60, "backend": "numpy"}
    )
    assert solver.solid.shape == (60, 120)
    assert "cylinder" in label
    assert 0.0 < solver.omega < 2.0


def test_builds_a_3d_case_on_either_lattice():
    for lattice, q in (("d3q19", 19), ("d3q27", 27)):
        solver, _ = web._build_solver(
            {"mode": "3d", "shape": "sphere", "radius": 4, "re": 20,
             "u0": 0.05, "nx": 40, "ny": 24, "nz": 24,
             "lattice": lattice, "backend": "numpy"}
        )
        assert solver.f.shape[0] == q


@pytest.mark.parametrize("shape", web.SHAPES_2D)
def test_every_offered_2d_shape_builds(shape):
    web._build_solver({"mode": "2d", "shape": shape, "re": 60, "u0": 0.05,
                       "nx": 120, "ny": 60, "backend": "numpy"})


@pytest.mark.parametrize("shape", web.SHAPES_3D)
def test_every_offered_3d_shape_builds(shape):
    extra = {}
    if shape == "mesh":                   # needs an upload to build from
        from pathlib import Path
        stl = Path(__file__).resolve().parents[1] / "samples" / "stl" / "unit_sphere.stl"
        extra = {"mesh_id": web._store_mesh("unit_sphere.stl", stl.read_bytes())["id"], "mesh_size": 8}
    web._build_solver({"mode": "3d", "shape": shape, "re": 20, "u0": 0.05,
                       "nx": 40, "ny": 24, "nz": 24, "backend": "numpy", **extra})


# ---------------------------------------------------------------------------
# Bad input is rejected with something a user can act on
# ---------------------------------------------------------------------------

def test_unstable_omega_is_refused_before_the_run_starts():
    """
    Re this high on this grid puts omega past 2, where LBM diverges.  Catching
    it up front beats letting the solve blow up ten thousand steps in.
    """
    with pytest.raises(ValueError, match="too close to|omega"):
        web._build_solver({"mode": "2d", "shape": "cylinder", "radius": 4,
                           "re": 100000, "u0": 0.05, "nx": 120, "ny": 60})


def test_body_larger_than_the_domain_is_refused():
    with pytest.raises(ValueError, match="taller than the domain"):
        web._build_solver({"mode": "2d", "shape": "cylinder", "radius": 60,
                           "re": 50, "u0": 0.05, "nx": 120, "ny": 60})


def test_mrt_on_d3q27_is_refused_with_a_useful_message():
    with pytest.raises(ValueError, match="D3Q19"):
        web._build_solver({"mode": "3d", "shape": "sphere", "radius": 4,
                           "re": 20, "u0": 0.05, "nx": 40, "ny": 24, "nz": 24,
                           "lattice": "d3q27", "collision": "mrt"})


@pytest.mark.parametrize("u0", [0.0, -0.1, 0.9])
def test_out_of_range_u0_is_refused(u0):
    with pytest.raises(ValueError, match="u0"):
        web._build_solver({"mode": "2d", "shape": "cylinder", "radius": 6,
                           "re": 50, "u0": u0, "nx": 120, "ny": 60})


def test_nonnumeric_input_falls_back_instead_of_crashing():
    """The form posts strings; a stray one must not 500 the server."""
    solver, _ = web._build_solver(
        {"mode": "2d", "shape": "cylinder", "radius": "", "re": "80",
         "u0": "0.05", "nx": "abc", "ny": "60", "backend": "numpy"}
    )
    assert solver.solid.shape[0] == 60


# ---------------------------------------------------------------------------
# Running a job
# ---------------------------------------------------------------------------

def test_a_short_job_runs_to_completion_and_reports_coefficients():
    job = web.Job(id="t1", params={
        "mode": "2d", "shape": "cylinder", "radius": 6, "re": 60, "u0": 0.05,
        "nx": 120, "ny": 60, "steps": 200, "backend": "numpy",
    })
    web._run_job(job)
    assert job.state == "done", job.error
    assert job.step == 200
    c = job.result["coefficients"]
    assert np.isfinite(c["cd"]) and c["cd"] > 0
    assert set(job.figures) == {"speed", "vorticity", "pressure"}
    for png in job.figures.values():
        assert png.startswith(b"\x89PNG")


def test_a_failing_case_reports_the_reason_rather_than_hanging():
    job = web.Job(id="t2", params={
        "mode": "2d", "shape": "cylinder", "radius": 60, "re": 50,
        "u0": 0.05, "nx": 120, "ny": 60, "steps": 50,
    })
    web._run_job(job)
    assert job.state == "error"
    assert "domain" in job.error
    assert job.finished is not None


def test_a_run_that_blows_up_is_reported_even_before_it_turns_nan(monkeypatch):
    """
    A diverging run's drag can stay finite -- ~1e40 -- for a while before it
    reaches NaN, so a NaN check alone let such a run finish as "complete" with
    garbage fields.  BGK on the default sphere used to do exactly that, until
    the inlet was regularized; here the blow-up is injected.
    """
    build = web._build_solver

    def diverging(p):
        solver, label = build(p)
        run = solver.run

        def run_then_blow_up(*args, **kwargs):
            out = run(*args, **kwargs)
            solver.Cd_history[-1] = 1e40
            return out
        solver.run = run_then_blow_up
        return solver, label

    monkeypatch.setattr(web, "_build_solver", diverging)
    job = web.Job(id="t_blowup", params={
        "mode": "2d", "shape": "cylinder", "radius": 6, "re": 20, "u0": 0.05,
        "nx": 80, "ny": 40, "steps": 200, "backend": "numpy",
    })
    web._run_job(job)
    assert job.state == "error"
    assert "diverged" in job.error and "collision operator" in job.error


def test_cancelling_stops_the_run_partway():
    job = web.Job(id="t3", params={
        "mode": "2d", "shape": "cylinder", "radius": 6, "re": 20, "u0": 0.05,
        "nx": 120, "ny": 60, "steps": 100000, "backend": "numpy",
    })
    t = threading.Thread(target=web._run_job, args=(job,), daemon=True)
    t.start()
    for _ in range(200):
        if job.step > 0:
            break
        time.sleep(0.05)
    job._cancel.set()
    t.join(timeout=60)
    assert not t.is_alive(), "cancel did not stop the worker"
    assert job.state == "stopped"
    assert 0 < job.step < 100000


def test_public_payload_is_json_serialisable():
    """It goes straight out over the wire — a stray ndarray would 500."""
    job = web.Job(id="t4", params={"mode": "2d"})
    job.history.append({"step": 1, "cd": 1.0, "cl": 0.0})
    json.dumps(job.public())


def test_public_payload_never_leaks_the_png_bytes():
    job = web.Job(id="t5", params={})
    job.preview_png = b"\x89PNG-not-really"
    pub = job.public()
    assert pub["has_preview"] is True
    assert "preview_png" not in pub


def test_3d_job_reports_lift_and_stays_strict_json():
    """
    3D keeps its lift in Cly_history, not Cl_history.  Reading the wrong name
    made cl NaN, json.dumps wrote a bare NaN token, and the browser's parser
    threw -- the page just stopped updating, with nothing in the log.
    """
    job = web.Job(id="t6", params={
        "mode": "3d", "shape": "sphere", "radius": 3, "re": 20, "u0": 0.05,
        "nx": 32, "ny": 16, "nz": 16, "steps": 120, "backend": "numpy",
    })
    web._run_job(job)
    assert job.state == "done", job.error
    payload = json.dumps(job.public(), allow_nan=False)      # raises on NaN
    assert "NaN" not in payload
    c = job.result["coefficients"]
    assert c["cl"] is not None and np.isfinite(c["cl"])
    assert all(h["cl"] is not None for h in job.history)


def test_num_turns_non_finite_into_none():
    assert web._num(float("nan")) is None
    assert web._num(float("inf")) is None
    assert web._num("x") is None
    assert web._num(1.23456789) == 1.23457


def test_job_reports_an_uncertainty_budget_and_a_reference():
    """
    The page used to print mean ± sigma, which read as an error bar and could
    never cover a systematic bias.  The job now carries the 95% statistical
    uncertainty, the budget's checks, and the expected value for the case as
    configured -- confinement included.
    """
    job = web.Job(id="t7", params={
        "mode": "3d", "shape": "sphere", "radius": 3, "re": 20, "u0": 0.05,
        "nx": 32, "ny": 16, "nz": 16, "steps": 200, "backend": "numpy",
    })
    web._run_job(job)
    assert job.state == "done", job.error
    c = job.result["coefficients"]
    assert c["cd_sem95"] is not None and c["cd_sem95"] >= 0
    assert c["cd_std"] is not None
    u = job.result["uncertainty"]
    for name in ("statistical", "blockage", "domain_length", "discretization"):
        assert u[name]["status"] in ("pass", "warn", "fail")
        assert u[name]["short"]
    r = job.result["reference"]
    assert r["expected"] > r["unconfined"]            # 6/16 blockage raises it
    assert r["status"] in ("pass", "fail")
    json.dumps(job.public(), allow_nan=False)


# ---------------------------------------------------------------------------
# The volume the 3D viewer renders
# ---------------------------------------------------------------------------

def _unpack(blob: bytes):
    nl = blob.index(b"\n")
    header = json.loads(blob[:nl])
    off, out = nl + 1, {}
    for part in header["parts"]:
        out[part["name"]] = np.frombuffer(blob, dtype=part["dtype"], count=part["count"], offset=off)
        off += part["count"] * np.dtype(part["dtype"]).itemsize
    return header, out


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_field_payload_is_the_wake_and_vorticity_as_bytes(mode):
    """
    The viewer volume-renders one byte per voxel.  The wake |u - U_inf| must
    be clear (zero) in an undisturbed stream and inside the body, which is
    drawn as a surface from the geometry instead.
    """
    p = {"mode": mode, "shape": "sphere" if mode == "3d" else "cylinder", "radius": 4,
         "re": 20, "u0": 0.05, "nx": 40, "ny": 24, "nz": 24, "backend": "numpy"}
    solver, _ = web._build_solver(p)
    header, parts = _unpack(web._field_payload(solver, mode))
    assert set(parts) == {"wake", "vorticity", "vel"}
    assert all(a.dtype == np.uint8 for a in parts.values())
    shape = header["shape"]
    assert parts["wake"].size == int(np.prod(shape))
    assert parts["vel"].size == 4 * int(np.prod(shape))            # RGBA per voxel
    solid = np.asarray(solver.solid, bool).reshape(shape)
    wake = parts["wake"].reshape(shape)
    assert wake[solid].max() == 0                        # the body is not fog
    assert wake[~solid].max() == 0                       # a fresh uniform stream is undisturbed
    assert header["scales"]["wake"] == web.WAKE_FULL_SCALE

    solver.run(steps=40, check_every=10 ** 9, verbose=False)
    header, parts = _unpack(web._field_payload(solver, mode))
    wake = parts["wake"].reshape(shape)
    assert wake[~solid].max() > 20                       # the body now disturbs the flow
    assert parts["vorticity"].max() == 255               # scaled to its own percentile
    assert header["scales"]["vorticity"] > 0
    vel = parts["vel"].reshape(shape + [4]).astype(float)
    ux = (vel[..., 0] / 255.0 - 0.5) * 2.0 * web.VEL_FULL_SCALE      # decoded, in u0
    assert np.all(np.abs(ux[solid]) <= 2.0 * web.VEL_FULL_SCALE / 255)  # at rest, to a byte
    assert np.median(ux[~solid]) == pytest.approx(1.0, abs=0.05)     # the free stream


def test_field_payload_is_strided_to_the_voxel_budget(monkeypatch):
    monkeypatch.setattr(web, "FIELD_VOXELS_MAX", 5000)
    solver, _ = web._build_solver({"mode": "3d", "shape": "sphere", "radius": 4, "re": 20,
                                   "u0": 0.05, "nx": 40, "ny": 24, "nz": 24, "backend": "numpy"})
    header, parts = _unpack(web._field_payload(solver, "3d"))
    assert header["factor"] >= 2 and header["full"] == [24, 24, 40]
    assert parts["wake"].size <= 5000 * 1.2


# ---------------------------------------------------------------------------
# Field figures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_field_figures_render_with_their_final_detail(mode):
    """
    The last figures of a run add streamlines and isobars.  The run loop
    swallows figure errors (a figure must never sink a run), so a failure in
    that path would only show as a figure quietly missing its detail.
    """
    p = {"mode": mode, "shape": "sphere" if mode == "3d" else "cylinder", "radius": 4,
         "re": 20, "u0": 0.05, "nx": 40, "ny": 24, "nz": 24, "backend": "numpy"}
    solver, _ = web._build_solver(p)
    solver.run(steps=30, check_every=10 ** 9, verbose=False)
    for what in ("speed", "vorticity", "pressure"):
        for detail in (False, True):
            png = web._field_png(solver, mode, what, streamlines=detail)
            assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_body_cells_take_their_fluid_neighbours_values():
    solid = np.zeros((5, 5), dtype=bool)
    solid[1:4, 1:4] = True
    a = np.arange(25, dtype=float).reshape(5, 5)
    out = web._extend_into(solid, a)
    assert np.array_equal(out[~solid], a[~solid])          # the fluid is untouched
    assert out[1, 2] == a[0, 2]                              # one fluid neighbour
    assert out[1, 1] == pytest.approx((a[0, 1] + a[1, 0]) / 2)
    assert np.isfinite(out).all()                            # the middle, a pass later


def test_smoothed_outline_keeps_a_straight_wall_in_place():
    solid = np.zeros((8, 8), dtype=bool)
    solid[:, :4] = True
    s = web._smoothed(solid)
    # the 0.5 level of a blurred half-plane sits on the original boundary
    assert np.all(s[:, 3] > 0.5) and np.all(s[:, 4] < 0.5)


# ---------------------------------------------------------------------------
# Pre-run checks and the per-step series
# ---------------------------------------------------------------------------

def test_preflight_reports_blockage_before_anything_runs():
    out = web._preflight({"mode": "3d", "shape": "sphere", "radius": "7", "nx": "96",
                          "ny": "48", "nz": "48", "re": "100", "u0": "0.05"})
    assert out["error"] is None
    checks = out["uncertainty"]
    assert {"stability", "blockage", "domain_length", "discretization"} <= set(checks)
    assert "29%" in checks["blockage"]["short"]
    assert out["reference"]["expected"] > out["reference"]["unconfined"]     # confinement
    assert out["setup"]["cells"] == 96 * 48 * 48
    json.dumps(out)


def test_preflight_says_why_a_case_would_be_refused():
    out = web._preflight({"mode": "2d", "shape": "cylinder", "radius": "120", "ny": "200"})
    assert out["error"] == "The body is taller than the domain."
    assert out["uncertainty"]["stability"]["status"] == "fail"


def test_a_3d_body_larger_than_the_tunnel_is_refused():
    out = web._preflight({"mode": "3d", "shape": "sphere", "radius": "30", "ny": "48", "nz": "48"})
    assert out["error"] == "The body is larger than the tunnel's cross-section."
    with pytest.raises(ValueError):
        web._build_solver({"mode": "3d", "shape": "sphere", "radius": 30, "ny": 48, "nz": 48})


def test_preflight_survives_a_half_typed_form():
    out = web._preflight({"mode": "2d", "shape": "cylinder", "radius": "", "nx": "1e", "re": "100"})
    assert out["error"] is None
    assert all(c["status"] in {"pass", "warn", "fail"} for c in out["uncertainty"].values())
    assert "budget" not in out["uncertainty"]


def test_preflight_warns_close_to_omega_two():
    out = web._preflight({"mode": "3d", "shape": "sphere", "radius": "3", "nx": "96",
                          "ny": "48", "nz": "48", "re": "100", "u0": "0.05"})
    stab = out["uncertainty"]["stability"]
    assert stab["status"] == "warn" and "1.965" in stab["short"]


def test_series_payload_serves_every_step_incrementally():
    job = web.Job(id="t_series", params={"mode": "2d", "shape": "cylinder", "radius": 6, "re": 20,
                                         "u0": 0.05, "nx": 80, "ny": 40, "steps": 150,
                                         "backend": "numpy"})
    web._run_job(job)
    assert job.state == "done"
    header, parts = _unpack(web._series_payload(job, 0))
    assert header["from"] == 0 and header["total"] == 150
    assert parts["cd"].size == parts["cl"].size == 150
    assert np.isfinite(parts["cd"]).all()
    header, parts = _unpack(web._series_payload(job, 140))
    assert header["from"] == 140 and parts["cd"].size == 10
    header, parts = _unpack(web._series_payload(job, 10 ** 6))      # past the end: nothing
    assert parts["cd"].size == 0


# ---------------------------------------------------------------------------
# Replay frames
# ---------------------------------------------------------------------------

def _small_job(job_id, steps=300):
    return web.Job(id=job_id, params={"mode": "2d", "shape": "cylinder", "radius": 6, "re": 20,
                                      "u0": 0.05, "nx": 80, "ny": 40, "steps": steps,
                                      "backend": "numpy"})


def test_replay_frames_are_recorded_and_served_after_a_step():
    job = _small_job("t_frames")
    web._run_job(job)
    steps = [fr[0] for fr in job.frames]
    assert steps[0] == 0 and steps[-1] == 300 and steps == sorted(set(steps))
    header, parts = _unpack(web._frames_payload(job, -1))
    voxels = int(np.prod(header["shape"]))
    assert header["steps"] == steps
    assert parts["wake"].size == parts["vorticity"].size == len(steps) * voxels
    assert parts["wake"][voxels:].max() > 0                   # the wake grows after step 0
    header, parts = _unpack(web._frames_payload(job, steps[-2]))
    assert header["steps"] == [steps[-1]] and parts["wake"].size == voxels
    assert web.Job(id="x", params={}).public()["frame_step"] == -1
    assert job.public()["frame_step"] == 300


def test_replay_frames_stay_inside_their_budget(monkeypatch):
    job = _small_job("t_budget", steps=600)
    voxels = 80 * 40
    monkeypatch.setattr(web, "REPLAY_BYTES_MAX", 5 * 2 * voxels)     # five frames' worth
    web._run_job(job)
    steps = [fr[0] for fr in job.frames]
    assert sum(w.nbytes + v.nbytes for _, _, w, v in job.frames) <= 6 * 2 * voxels
    assert steps[0] == 0 and steps[-1] == 600                  # the ends are kept
    assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# STL uploads
# ---------------------------------------------------------------------------

SAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "samples" / "stl"


def _upload(name="unit_sphere.stl"):
    return web._store_mesh(name, (SAMPLES / name).read_bytes())


def _mesh_params(mesh_id, **kw):
    p = {"mode": "3d", "shape": "mesh", "mesh_id": mesh_id, "mesh_size": "10", "nx": "48",
         "ny": "24", "nz": "24", "re": "20", "u0": "0.05", "backend": "numpy"}
    p.update(kw)
    return p


def test_an_uploaded_stl_is_parsed_and_described():
    info = _upload()
    assert info["triangles"] == 1280 and info["open_edges"] == 0
    assert info["extent"] == pytest.approx([2.0, 2.0, 2.0], rel=1e-3)
    assert web._mesh_info(info["id"])["name"] == "unit_sphere.stl"
    assert "tris" not in web._mesh_info(info["id"])


def test_an_open_mesh_is_flagged():
    from aero.geometry3d.stl_io import parse_stl
    tris = parse_stl((SAMPLES / "unit_cube.stl").read_bytes())[:-1]         # one face short
    assert web._open_edges(tris) == 3


@pytest.mark.parametrize("data, why", [(b"not an stl at all", "readable"),
                                       (b"solid x\nendsolid x\n", "readable")])
def test_a_bad_upload_says_why(data, why):
    with pytest.raises(ValueError, match=why):
        web._store_mesh("bad.stl", data)


def test_only_the_newest_meshes_are_kept():
    ids = [_upload()["id"] for _ in range(web.MESHES_KEPT + 2)]
    assert ids[0] not in web.MESHES and ids[-1] in web.MESHES


def test_a_mesh_case_uses_its_size_as_the_reference_length():
    info = _upload()
    case = web._case(_mesh_params(info["id"], mesh_size="12"))
    assert case["D"] == 12.0 and case["label"] == "mesh unit_sphere.stl"
    with pytest.raises(ValueError, match="Choose an STL"):
        web._case(_mesh_params("nope"))
    with pytest.raises(ValueError, match="cross-section"):
        web._case(_mesh_params(info["id"], mesh_size="30"))


def test_preflight_reports_how_the_mesh_landed_on_the_grid():
    info = _upload()
    out = web._preflight(_mesh_params(info["id"]))
    mesh = out["uncertainty"]["mesh"]
    assert mesh["status"] == "pass" and "solid cells" in mesh["short"] and "frontal area" not in mesh["short"]
    assert "by area" in out["uncertainty"]["blockage"]["short"]
    assert "blockage" in out["uncertainty"]
    json.dumps(out)
    thin = web._preflight(_mesh_params(_upload("simple_plane.stl")["id"], mesh_size="6"))
    assert thin["uncertainty"]["mesh"]["status"] in {"warn", "fail"}


def test_a_slender_mesh_is_graded_on_its_frontal_area():
    """A plane spanning 42% of the tunnel blocks ~3% of it; the extent would say "fail"."""
    plane = _upload("simple_plane.stl")["id"]
    out = web._preflight(_mesh_params(plane, mesh_size="20", nx="96", ny="48", nz="48", mesh_rot_x="90"))
    b = out["uncertainty"]["blockage"]
    assert "by area" in b["short"]
    area = float(b["short"].split()[1].rstrip("%"))
    assert area < 5.0
    assert "42%" in b["message"]                 # the extent is still reported


def test_preview_is_the_mask_the_run_uses():
    info = _upload()
    p = _mesh_params(info["id"])
    case = web._case(p)
    header, parts = _unpack(web._geometry_payload(web._solid_for(p, case)))
    assert header["shape"] == [24, 24, 48]
    solver, label = web._build_solver(p)
    assert np.array_equal(parts["solid"].reshape(24, 24, 48).astype(bool), solver.solid)
    assert solver.ref_area > 0                                   # measured from the voxels
    # every shape previews, in 2D too
    header, parts = _unpack(web._geometry_payload(web._solid_for(
        {"mode": "2d", "shape": "cylinder", "radius": 5, "nx": 60, "ny": 30},
        web._case({"mode": "2d", "shape": "cylinder", "radius": 5, "nx": 60, "ny": 30}))))
    assert header["shape"] == [1, 30, 60] and parts["solid"].sum() > 0


def test_a_mesh_run_completes():
    info = _upload()
    job = web.Job(id="t_mesh", params={**_mesh_params(info["id"]), "steps": 60})
    web._run_job(job)
    assert job.state == "done", job.error
    assert job.result["setup"]["label"] == "mesh unit_sphere.stl"
    assert np.isfinite(job.result["coefficients"]["cd"])


def test_a_part_too_thin_for_its_size_is_told_the_size_it_needs():
    """The toy aircraft at 12 cells across is thinner than a cell everywhere."""
    plane = _upload("simple_plane.stl")["id"]
    p = _mesh_params(plane, mesh_size="12", nx="192", ny="96", nz="96")
    m = web._preflight(p)["uncertainty"]["mesh"]
    assert m["status"] == "fail" and m["short"].startswith("no cells at Size 12")
    need = int(m["short"].rsplit(" ", 1)[1])
    assert 40 <= need <= 96
    assert f"Size {need}" in m["message"]
    # at that size it is on the grid
    ok = web._preflight(_mesh_params(plane, mesh_size=str(need), nx="192", ny="96", nz="96"))
    assert ok["uncertainty"]["mesh"]["status"] == "pass"


def test_the_sample_aircraft_keeps_its_shape_at_the_size_its_readme_gives():
    plane = _upload("tunnel_plane.stl")["id"]
    m = web._preflight(_mesh_params(plane, mesh_size="48", mesh_orient="none",
                                    nx="144", ny="72", nz="96"))["uncertainty"]["mesh"]
    assert m["status"] == "pass"
    small = web._preflight(_mesh_params(plane, mesh_size="24", mesh_orient="none",
                                        nx="144", ny="72", nz="96"))["uncertainty"]["mesh"]
    assert small["status"] == "warn" and "needs about 48" in small["short"]

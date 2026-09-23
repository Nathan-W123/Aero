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
    web._build_solver({"mode": "3d", "shape": shape, "re": 20, "u0": 0.05,
                       "nx": 40, "ny": 24, "nz": 24, "backend": "numpy"})


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

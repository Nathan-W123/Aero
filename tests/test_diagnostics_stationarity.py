"""Tests for rolling convergence and statistical-stationarity diagnostics."""

import math

import numpy as np

from aero.diagnostics import (
    analyze_convergence,
    analyze_strouhal,
    detect_statistical_stationarity,
)


def test_analyze_convergence_reports_ratio():
    history = [1.0 + 1.0e-4 * math.sin(i) for i in range(600)]
    report = analyze_convergence(history, window=500, tol=0.01)
    assert report.converged
    assert report.ratio < report.threshold


def test_analyze_strouhal_returns_peak():
    D = 20.0
    u0 = 0.05
    t = np.arange(8192, dtype=float)
    peak_bin = 4
    freq = peak_bin / t.size
    st_target = freq * D / u0
    cl = 0.2 * np.sin(2.0 * math.pi * freq * t)
    report = analyze_strouhal(list(cl), D=D, u0=u0, window=8192)
    assert report.strouhal is not None
    assert abs(report.strouhal - st_target) / st_target < 0.02


def test_detect_statistical_stationarity_requires_both_metrics():
    D = 20.0
    u0 = 0.05
    st_target = 0.17
    freq = st_target * u0 / D
    t = np.arange(5000, dtype=float)
    cd = list(1.5 + 1.0e-4 * np.sin(0.01 * t))
    cl = list(0.15 * np.sin(2.0 * math.pi * freq * t))
    cd_report, st_report, stop = detect_statistical_stationarity(
        cd,
        cl,
        D=D,
        u0=u0,
        convergence_window=2000,
        strouhal_window=2048,
    )
    assert cd_report.converged
    assert st_report.strouhal is not None
    assert stop

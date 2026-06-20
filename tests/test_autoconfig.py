"""Tests for shared preflight autoconfiguration."""

import argparse

import pytest

from cli import build_parser as build_parser_2d
from cli3d import build_parser as build_parser_3d
from aero.autoconfig import autoconfigure_2d, autoconfigure_3d


def test_cli2d_accepts_autoconfigure_flag():
    args = build_parser_2d().parse_args(["--autoconfigure", "safe"])
    assert args.autoconfigure == "safe"


def test_cli3d_accepts_autoconfigure_flag():
    args = build_parser_3d().parse_args(["--autoconfigure", "safe"])
    assert args.autoconfigure == "safe"


def test_autoconfigure_2d_recovers_low_tau_and_high_ma():
    args = argparse.Namespace(
        shape="cylinder",
        radius=10.0,
        height=20.0,
        width=40.0,
        ny=60,
        nx=120,
        re=400.0,
        u0=0.2,
        wall_bc="slip",
        collision="bgk",
        inlet_perturbation=0.0,
    )
    report = autoconfigure_2d(args, policy="safe")
    assert report is not None
    assert report.applied is True
    assert args.nx > 120
    assert args.ny > 60
    assert args.collision == "mrt"
    assert args.inlet_perturbation == pytest.approx(0.02)
    assert report.final_summary["tau"] >= 0.55
    assert report.final_summary["Ma"] <= 0.15


def test_autoconfigure_2d_expands_blocked_grid():
    args = argparse.Namespace(
        shape="cylinder",
        radius=30.0,
        height=20.0,
        width=40.0,
        ny=100,
        nx=200,
        re=100.0,
        u0=0.05,
        wall_bc="noslip",
        collision="bgk",
        inlet_perturbation=0.01,
    )
    report = autoconfigure_2d(args, policy="safe")
    assert report is not None
    assert args.ny > 100
    assert args.nx > 200
    assert report.final_summary["blockage"] <= 0.40


def test_autoconfigure_3d_expands_high_blockage_grid():
    args = argparse.Namespace(
        shape="sphere",
        radius=20.0,
        height=10.0,
        depth=10.0,
        length=20.0,
        ny=64,
        nz=64,
        nx=128,
        re=100.0,
        u0=0.05,
        wall_bc="slip",
        collision="bgk",
        inlet_perturbation=0.0,
        stl_fit=0.35,
    )
    report = autoconfigure_3d(args, policy="safe")
    assert report is not None
    assert report.applied is True
    assert args.ny > 64
    assert args.nz > 64
    assert args.nx > 128
    assert report.final_summary["blockage"] <= 0.30


def test_autoconfigure_off_returns_none():
    args = argparse.Namespace(
        shape="sphere",
        radius=10.0,
        height=10.0,
        depth=10.0,
        length=20.0,
        ny=64,
        nz=64,
        nx=128,
        re=100.0,
        u0=0.05,
        wall_bc="slip",
        collision="bgk",
        inlet_perturbation=0.0,
        stl_fit=0.35,
    )
    assert autoconfigure_3d(args, policy="off") is None

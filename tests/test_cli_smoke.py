"""
End-to-end smoke tests for the command-line entry points.

The existing CLI tests only build the argument parser, so a mistake anywhere
past ``build_parser`` went unnoticed: ``cli.py`` referenced ``args.van_driest_a``
for an option argparse names ``van_driest_A``, which made *every* 2D CLI
invocation die with an AttributeError — the first command in the README
included.  These tests actually call ``main()``.
"""

import sys

import numpy as np
import pytest


def _run_cli(module_main, argv, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    return module_main()


def test_cli_2d_runs_end_to_end(monkeypatch, tmp_path):
    from cli import main

    rc = _run_cli(
        main,
        ["cli.py", "--shape", "cylinder", "--nx", "48", "--ny", "24",
         "--radius", "4", "--re", "40", "--steps", "40",
         "--check-every", "20", "--backend", "numpy", "--no-plot"],
        monkeypatch, tmp_path,
    )
    assert rc in (0, None)


def test_cli_2d_runs_with_every_physics_option(monkeypatch, tmp_path):
    """Exercise the option-heavy path, which is where wiring mistakes hide."""
    from cli import main

    rc = _run_cli(
        main,
        ["cli.py", "--shape", "cylinder", "--nx", "48", "--ny", "24",
         "--radius", "4", "--re", "40", "--steps", "30", "--check-every", "15",
         "--backend", "numpy", "--no-plot",
         "--collision", "trt", "--les", "--les-model", "wale",
         "--van-driest", "--van-driest-A", "26.0",
         "--bouzidi", "--sponge-cells", "4",
         "--statistics", "--stats-start", "5",
         "--body-force-x", "1e-7"],
        monkeypatch, tmp_path,
    )
    assert rc in (0, None)


def test_cli_3d_runs_end_to_end(monkeypatch, tmp_path):
    from cli3d import main

    rc = _run_cli(
        main,
        ["cli3d.py", "--shape", "sphere", "--nx", "24", "--ny", "12",
         "--nz", "12", "--radius", "3", "--re", "10", "--steps", "20",
         "--check-every", "10", "--backend", "numpy", "--no-plot", "--allow-high-blockage"],
        monkeypatch, tmp_path,
    )
    assert rc in (0, None)


def test_cli_3d_runs_with_statistics_and_body_force(monkeypatch, tmp_path):
    from cli3d import main

    rc = _run_cli(
        main,
        ["cli3d.py", "--shape", "sphere", "--nx", "24", "--ny", "12",
         "--nz", "12", "--radius", "2", "--re", "10", "--steps", "20",
         "--check-every", "10", "--backend", "numpy", "--no-plot", "--allow-high-blockage",
         "--streamwise-bc", "periodic", "--body-force-x", "1e-6",
         "--statistics", "--stats-start", "2"],
        monkeypatch, tmp_path,
    )
    assert rc in (0, None)


@pytest.mark.parametrize("module_name", ["cli", "cli3d"])
def test_every_parser_dest_is_read_back_correctly(module_name):
    """
    Guard against ``args.<wrong_case>`` typos.

    argparse keeps the case of a long option, so ``--van-driest-A`` becomes
    ``van_driest_A``.  Referring to ``van_driest_a`` raises only at runtime,
    which is how the 2D CLI shipped broken.  This checks that every attribute
    the module reads off ``args`` is actually a dest the parser defines.
    """
    import ast
    import importlib
    import pathlib

    module = importlib.import_module(module_name)
    parser = module.build_parser()
    dests = {a.dest for a in parser._actions}

    source = pathlib.Path(module.__file__).read_text()
    tree = ast.parse(source)
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }
    missing = sorted(referenced - dests)
    assert not missing, f"{module_name} reads undefined arg dests: {missing}"

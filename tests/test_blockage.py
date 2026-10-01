"""Tests for the blockage model (aero.blockage)."""

import math

import numpy as np
import pytest

from aero.blockage import (
    B_MAX, CALIBRATION, blockage_correction, blockage_of_case, blockage_of_solid, free_air_cd,
    linear_blockage_from_area,
)
from aero.geometry.cylinder import Cylinder
from aero.geometry3d.box import Box
from aero.geometry3d.sphere import Sphere


def test_a_sphere_in_a_square_tunnel_blocks_d_over_span():
    assert linear_blockage_from_area(math.pi * 7 ** 2, 48 * 48) == pytest.approx(14 / 48)
    bl = blockage_of_case("3d", "sphere", {"radius": 7, "ny": 48, "nz": 48})
    assert bl["b"] == pytest.approx(14 / 48) and bl["law"] == "3d"
    assert bl["area_ratio"] == pytest.approx(math.pi * 49 / 48 ** 2)


def test_a_cube_and_a_sphere_of_the_same_frontal_area_block_alike():
    side = 14 * math.sqrt(math.pi) / 2
    cube = blockage_of_case("3d", "box", {"width": side, "height": side, "depth": side, "ny": 48, "nz": 48})
    sphere = blockage_of_case("3d", "sphere", {"radius": 7, "ny": 48, "nz": 48})
    assert cube["b"] == pytest.approx(sphere["b"])


def test_2d_blockage_is_the_frontal_height_over_the_tunnel():
    assert blockage_of_case("2d", "cylinder", {"radius": 20, "ny": 200})["b"] == pytest.approx(0.2)
    assert blockage_of_case("2d", "rectangle", {"width": 40, "height": 20, "ny": 200})["b"] == pytest.approx(0.1)


def test_a_cylinder_across_the_whole_periodic_width_blocks_as_a_2d_one():
    finite = blockage_of_case("3d", "cylinder", {"radius": 8, "length": 24, "ny": 48, "nz": 48})
    spanning = blockage_of_case("3d", "cylinder", {"radius": 8, "length": 48, "ny": 48, "nz": 48})
    assert finite["law"] == "3d"
    assert spanning["law"] == "2d" and spanning["b"] == pytest.approx(16 / 48)


def test_voxel_blockage_matches_the_formula():
    solid = Sphere(radius=7).mark_solid(48, 48, 96)
    vox = blockage_of_solid(solid)
    assert vox["law"] == "3d"
    assert vox["b"] == pytest.approx(14 / 48, rel=0.03)
    flat = blockage_of_solid(Cylinder(radius=20).mark_solid(200, 400))
    assert flat["law"] == "2d" and flat["b"] == pytest.approx(0.2, abs=0.01)
    slab = np.zeros((8, 30, 40), bool)
    slab[:, 10:16, 5:9] = True                       # touches every z: an infinite body
    assert blockage_of_solid(slab)["law"] == "2d" and blockage_of_solid(slab)["b"] == pytest.approx(6 / 30)


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_the_correction_grows_with_blockage_and_vanishes_without_it(mode):
    re = 50.0
    K = [blockage_correction(mode, re, b)["K"] for b in (0.0, 0.05, 0.1, 0.2, 0.3)]
    assert K[0] == 0.0
    assert all(a < b for a, b in zip(K, K[1:]))


def test_a_calibrated_shape_is_more_certain_than_an_uploaded_one():
    sphere = blockage_correction("3d", 100.0, 0.2, "sphere")
    mesh = blockage_correction("3d", 100.0, 0.2, "mesh")
    assert sphere["relative_uncertainty"] < mesh["relative_uncertainty"]
    assert "measured for this shape" in sphere["basis"] and "not measured" in mesh["basis"]


def test_an_uncalibrated_shape_takes_the_mean_of_the_calibrated_laws():
    tables = CALIBRATION["2d"]
    if len(tables) < 2:
        pytest.skip("one calibrated 2D shape")
    each = [blockage_correction("2d", 40.0, 0.2, name)["K"] for name in tables]
    other = blockage_correction("2d", 40.0, 0.2, "polygon")
    assert other["K"] == pytest.approx(np.mean(each))
    assert other["dK"] >= 0.5 * (max(each) - min(each))       # the spread between shapes is covered


def test_uncertainty_grows_outside_what_was_measured():
    table = CALIBRATION["3d"]["sphere"]
    lo_re, hi_re = table[0][0], table[-1][0]
    rel = lambda re, b: blockage_correction("3d", re, b, "sphere")["relative_uncertainty"]
    inside = rel(hi_re, 0.2)
    assert inside < rel(20 * hi_re, 0.2) and inside < rel(lo_re / 10, 0.2)
    assert inside < rel(hi_re, 1.5 * B_MAX["3d"])


def test_free_air_divides_out_the_confinement():
    corr = blockage_correction("2d", 40.0, 0.2, "cylinder")
    cd_free, unc = free_air_cd(2.0, corr)
    assert cd_free == pytest.approx(2.0 / (1.0 + corr["K"]))
    assert unc == pytest.approx(cd_free * corr["dK"] / (1.0 + corr["K"]))


def test_the_calibration_tables_are_ordered_and_physical():
    for mode, tables in CALIBRATION.items():
        for shape, pts in tables.items():
            res = [r for r, _, _ in pts]
            assert res == sorted(res) and len(set(res)) == len(res)
            for re, k1, k2 in pts:
                # confinement raises the drag at every blockage up to the largest measured
                for b in np.linspace(0.01, B_MAX[mode], 8):
                    assert k1 * b + k2 * b * b > 0.0, (mode, shape, re)


def test_a_rectangle_unlike_the_calibrated_square_is_less_certain():
    square = blockage_correction("2d", 40.0, 0.2, "rectangle", {"width": 20, "height": 20})
    long = blockage_correction("2d", 40.0, 0.2, "rectangle", {"width": 40, "height": 20})
    assert square["K"] == long["K"] and square["dK"] < long["dK"]
    assert "proportions differ" in long["basis"]

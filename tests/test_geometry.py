"""
Unit tests for geometry solid-mask generation.
"""

import numpy as np
import pytest
from aero.geometry.cylinder import Cylinder
from aero.geometry.rectangle import Rectangle


Ny, Nx = 200, 400


# ---------------------------------------------------------------------------
# Cylinder
# ---------------------------------------------------------------------------

class TestCylinder:
    def test_cell_count_approx_pi_r_squared(self):
        r = 20
        cyl   = Cylinder(radius=r)
        solid = cyl.mark_solid(Ny, Nx)
        expected = np.pi * r ** 2
        # Grid discretisation gives ≤2% error for r=20
        assert abs(solid.sum() - expected) / expected < 0.02

    def test_solid_is_boolean(self):
        solid = Cylinder(radius=20).mark_solid(Ny, Nx)
        assert solid.dtype == bool

    def test_shape(self):
        solid = Cylinder(radius=20).mark_solid(Ny, Nx)
        assert solid.shape == (Ny, Nx)

    def test_center_inside_solid(self):
        cyl   = Cylinder(radius=20, cx_frac=0.5, cy_frac=0.5)
        solid = cyl.mark_solid(Ny, Nx)
        cx, cy = cyl.center(Ny, Nx)
        assert solid[int(cy), int(cx)]

    def test_far_corner_is_fluid(self):
        solid = Cylinder(radius=20, cx_frac=0.5, cy_frac=0.5).mark_solid(Ny, Nx)
        assert not solid[0, 0]
        assert not solid[0, -1]
        assert not solid[-1, 0]
        assert not solid[-1, -1]

    def test_reference_length_is_diameter(self):
        cyl = Cylinder(radius=15)
        assert cyl.reference_length() == 30.0

    def test_no_solid_cells_when_radius_zero(self):
        solid = Cylinder(radius=0).mark_solid(Ny, Nx)
        # A point circle (radius=0) marks the center cell
        assert solid.sum() <= 1


# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------

class TestRectangle:
    def test_cell_count_approx_width_times_height(self):
        w, h  = 40, 20
        rect  = Rectangle(width=w, height=h)
        solid = rect.mark_solid(Ny, Nx)
        expected = (w + 1) * (h + 1)   # inclusive boundary
        assert abs(solid.sum() - expected) / expected < 0.05

    def test_solid_is_boolean(self):
        assert Rectangle(width=40, height=20).mark_solid(Ny, Nx).dtype == bool

    def test_shape(self):
        assert Rectangle(width=40, height=20).mark_solid(Ny, Nx).shape == (Ny, Nx)

    def test_center_inside_solid(self):
        rect  = Rectangle(width=40, height=20, cx_frac=0.5, cy_frac=0.5)
        solid = rect.mark_solid(Ny, Nx)
        cx, cy = rect.center(Ny, Nx)
        assert solid[int(cy), int(cx)]

    def test_reference_length_is_height(self):
        rect = Rectangle(width=60, height=25)
        assert rect.reference_length() == 25.0

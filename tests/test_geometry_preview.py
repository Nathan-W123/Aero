"""Tests for pre-run 3D geometry preview helpers."""

import numpy as np

from aero.gui.geometry_preview import solid_mask_for_3d_config
from aero.gui.state import GuiConfig


def test_solid_mask_sphere():
    config = GuiConfig(mode="3d", shape_3d="sphere")
    config.params_3d.update({"nx": "64", "ny": "48", "nz": "48", "radius": "7"})
    solid = solid_mask_for_3d_config(config)
    assert solid is not None
    assert solid.shape == (48, 48, 64)
    assert solid.sum() > 0


def test_solid_mask_mesh_missing_stl():
    config = GuiConfig(mode="3d", shape_3d="mesh")
    config.params_3d.update({"nx": "32", "ny": "32", "nz": "32", "stl_path": ""})
    assert solid_mask_for_3d_config(config) is None

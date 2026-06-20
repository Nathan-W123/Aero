"""3D geometry obstacles for the LBM wind tunnel."""
from .sphere import Sphere
from .box import Box
from .cylinder3d import Cylinder3D
from .mesh_mask import MeshMask

__all__ = ["Sphere", "Box", "Cylinder3D", "MeshMask"]

"""
Geometry defined by loading a grayscale or colour image and thresholding it.

Dark pixels (value < threshold) become solid; light pixels become fluid.
Pass invert=True to reverse (dark=fluid, light=solid).

Only matplotlib.image is used — no PIL / scipy dependency.
"""

import numpy as np
import matplotlib.image as mpimg
from .base import Geometry
from typing import Tuple


class ImageMask(Geometry):
    """
    Load any PNG/JPG supported by matplotlib and threshold it to a solid mask.

    Parameters
    ----------
    path      : str   — path to image file
    threshold : float — pixels with grayscale value > threshold are fluid (default 0.5)
    invert    : bool  — swap solid/fluid (default False: dark=solid, light=fluid)
    """

    def __init__(self, path: str, threshold: float = 0.5, invert: bool = False):
        self.path = path
        self.threshold = threshold
        self.invert = invert

    def mark_solid(self, Ny: int, Nx: int) -> np.ndarray:
        img = mpimg.imread(self.path)

        # Normalise uint8 images (0-255) to float (0.0-1.0)
        if img.dtype == np.uint8:
            img = img.astype(np.float64) / 255.0

        # Convert to grayscale
        if img.ndim == 3:
            # Use luminance weights for RGB or RGBA
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img.astype(np.float64)

        # Nearest-neighbour resize to (Ny, Nx)
        src_h, src_w = gray.shape
        yi = np.clip((np.arange(Ny) * src_h / Ny).astype(int), 0, src_h - 1)
        xi = np.clip((np.arange(Nx) * src_w / Nx).astype(int), 0, src_w - 1)
        resized = gray[yi[:, None], xi[None, :]]   # shape (Ny, Nx)

        solid = resized <= self.threshold
        if self.invert:
            solid = ~solid
        return solid

    def center(self, Ny: int, Nx: int) -> Tuple[float, float]:
        solid = self.mark_solid(Ny, Nx)
        ys, xs = np.where(solid)
        if len(xs) == 0:
            return Nx / 2.0, Ny / 2.0
        return float(xs.mean()), float(ys.mean())

    def reference_length(self) -> float:
        raise NotImplementedError(
            "ImageMask has no fixed reference length. "
            "Pass D explicitly when constructing the Solver."
        )

"""TRT parameters for D3Q19."""

from __future__ import annotations

import numpy as np

from .trt2d import trt_taus, trt_weights_2d


def trt_weights_3d(ex, ey, ez, tau_plus, tau_minus):
    return trt_weights_2d(
        (ex * ex + ey * ey + ez * ez).astype(ex.dtype),
        np.zeros_like(ex, dtype=ex.dtype),
        tau_plus,
        tau_minus,
    )

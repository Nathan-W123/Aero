"""Compact post-processing observables for coefficient histories."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def coefficient_spectrum(
    history: List[float],
    *,
    sample_spacing: float = 1.0,
    min_samples: int = 128,
    discard_fraction: float = 0.2,
    top_n: int = 8,
) -> Optional[Dict[str, object]]:
    """Return a compact FFT summary for a coefficient history."""
    values = np.asarray(history, dtype=np.float64)
    if values.size < min_samples:
        return None
    discard = int(values.size * discard_fraction)
    if discard >= values.size - min_samples:
        discard = max(0, values.size - min_samples)
    values = values[discard:]
    signal = values - float(np.mean(values))
    if np.max(np.abs(signal)) < 1e-12:
        return None

    freqs = np.fft.rfftfreq(signal.size, d=sample_spacing)
    amps = np.abs(np.fft.rfft(signal))
    if amps.size <= 1:
        return None
    ranked = np.argsort(amps[1:])[::-1] + 1
    ranked = ranked[:top_n]
    dominant = int(ranked[0])
    return {
        "sample_count": int(signal.size),
        "dominant_frequency": float(freqs[dominant]),
        "dominant_amplitude": float(amps[dominant]),
        "frequencies": [float(freqs[idx]) for idx in ranked],
        "amplitudes": [float(amps[idx]) for idx in ranked],
    }

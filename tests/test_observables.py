"""Tests for compact coefficient observables."""

import math

from aero.observables import coefficient_spectrum


def test_coefficient_spectrum_detects_frequency():
    target_freq = 0.025
    values = [0.2 * math.sin(2.0 * math.pi * target_freq * i) for i in range(512)]
    report = coefficient_spectrum(values, discard_fraction=0.0, min_samples=128)
    assert report is not None
    assert abs(report["dominant_frequency"] - target_freq) < 0.005

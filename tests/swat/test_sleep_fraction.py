"""Test 7 — sleep fraction in physiological window under healthy controls.

Spec calls for ~33% sleep time in scenario A. Allow 25%–40% to absorb
stochastic / dt sensitivity.
"""
import pytest
from model_validation.runner import sleep_fraction_under_controls


def test_healthy_sleep_fraction(model_with_noise):
    """Note: uses the WITH-noise model (sleep fraction is a stochastic stat)."""
    sf = sleep_fraction_under_controls(
        model_with_noise, V_h=1.0, V_n=0.3, V_c=0.0
    )
    print(f"\n  Healthy sleep fraction: {sf*100:.1f}% (target: 25%-40%)")
    assert 0.25 <= sf <= 0.40, (
        f"Sleep fraction {sf*100:.1f}% outside 25-40% target window. "
        f"This indicates beta_Z / c_tilde tuning has drifted."
    )

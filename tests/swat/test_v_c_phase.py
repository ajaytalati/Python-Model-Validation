"""Test 8 — V_c phase term enters E correctly.

V_c = ±6 hours collapses E to near zero (cos(2π·6/24) = 0).
Symmetry: T_end at V_c=+6 ≈ T_end at V_c=-6.
T_end at V_c=0 should be high (phase aligned).
"""
import pytest
from model_validation.runner import t_end_under_constant_controls


def test_v_c_zero_is_max(model):
    # V_n=0 (clean V_h-only signal for healthy)
    T_aligned    = t_end_under_constant_controls(model, V_h=1.0, V_n=0.0, V_c=0.0)
    T_shift_pos  = t_end_under_constant_controls(model, V_h=1.0, V_n=0.0, V_c=6.0)
    T_shift_neg  = t_end_under_constant_controls(model, V_h=1.0, V_n=0.0, V_c=-6.0)
    print(f"\n  V_c= 0:  T_end = {T_aligned:.3f}")
    print(f"  V_c=+6:  T_end = {T_shift_pos:.3f}")
    print(f"  V_c=-6:  T_end = {T_shift_neg:.3f}")
    assert T_aligned > T_shift_pos + 0.10, (
        f"V_c=0 should be substantially better than V_c=+6: "
        f"got T_aligned={T_aligned:.3f}, T_shift_pos={T_shift_pos:.3f}"
    )
    assert T_aligned > T_shift_neg + 0.10, (
        f"V_c=0 should be substantially better than V_c=-6: "
        f"got T_aligned={T_aligned:.3f}, T_shift_neg={T_shift_neg:.3f}"
    )


def test_v_c_symmetric(model):
    """T_end should be symmetric in V_c (cos is even)."""
    T_pos = t_end_under_constant_controls(model, V_h=1.0, V_n=0.0, V_c=6.0)
    T_neg = t_end_under_constant_controls(model, V_h=1.0, V_n=0.0, V_c=-6.0)
    assert abs(T_pos - T_neg) < 0.05, (
        f"V_c collapse not symmetric: V_c=+6 -> {T_pos:.3f}, "
        f"V_c=-6 -> {T_neg:.3f}"
    )

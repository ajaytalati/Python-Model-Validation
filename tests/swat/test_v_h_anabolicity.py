"""Test 1 — V_h is anabolic (T_end monotonically non-decreasing in V_h).

Sweep V_h ∈ {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0} at V_n=0.3, V_c=0.
Strong assertion: T_end at V_h=2.0 is at least 0.05 above T_end at V_h=0.5.

Pre-fix vendored snapshot: FAILS (T_end actually DECREASES with V_h).
Refined Option C: should PASS.
"""
import pytest
from model_validation.runner import t_end_under_constant_controls


V_H_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def test_v_h_monotonic_non_decreasing(model):
    # V_n=0: clean V_h-only signal (per user's clinical spec for healthy default)
    Ts = [t_end_under_constant_controls(model, V_h, V_n=0.0, V_c=0.0)
          for V_h in V_H_GRID]
    print(f"\n  V_h:    {V_H_GRID}")
    print(f"  T_end:  {[round(t, 3) for t in Ts]}")
    for i in range(1, len(Ts)):
        assert Ts[i] >= Ts[i - 1] - 0.01, (
            f"V_h is NOT anabolic: at V_h-step {i} "
            f"({V_H_GRID[i - 1]} -> {V_H_GRID[i]}), "
            f"T_end dropped {Ts[i - 1]:.3f} -> {Ts[i]:.3f}. "
            f"Full grid: V_h={V_H_GRID}, T={[round(t, 3) for t in Ts]}"
        )


def test_v_h_clinically_meaningful_response(model):
    """T_end at V_h=2.0 must exceed T_end at V_h=0.5 by at least 0.05."""
    T_low  = t_end_under_constant_controls(model, V_h=0.5, V_n=0.0, V_c=0.0)
    T_high = t_end_under_constant_controls(model, V_h=2.0, V_n=0.0, V_c=0.0)
    assert T_high >= T_low + 0.05, (
        f"V_h response too weak: T(V_h=2.0)={T_high:.3f}, "
        f"T(V_h=0.5)={T_low:.3f}, "
        f"difference {T_high - T_low:+.3f} below required +0.05"
    )

"""Test 2 — V_n is catabolic (T_end monotonically non-increasing in V_n).

Sweep V_n ∈ {0, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0} at V_h=1.0, V_c=0.

Pre-fix vendored snapshot: FAILS (T_end is non-monotonic in V_n).
Refined Option C: should PASS.
"""
import pytest
from model_validation.runner import t_end_under_constant_controls


V_N_GRID = [0.0, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0]


def test_v_n_monotonic_non_increasing(model):
    Ts = [t_end_under_constant_controls(model, V_h=1.0, V_n=V_n, V_c=0.0)
          for V_n in V_N_GRID]
    print(f"\n  V_n:    {V_N_GRID}")
    print(f"  T_end:  {[round(t, 3) for t in Ts]}")
    for i in range(1, len(Ts)):
        assert Ts[i] <= Ts[i - 1] + 0.01, (
            f"V_n is NOT catabolic: at V_n-step {i} "
            f"({V_N_GRID[i - 1]} -> {V_N_GRID[i]}), "
            f"T_end rose {Ts[i - 1]:.3f} -> {Ts[i]:.3f}. "
            f"Full grid: V_n={V_N_GRID}, T={[round(t, 3) for t in Ts]}"
        )


def test_v_n_clinically_meaningful_response(model):
    """T_end at V_n=0.3 must exceed T_end at V_n=3.5 by at least 0.10."""
    T_low_n  = t_end_under_constant_controls(model, V_h=1.0, V_n=0.3, V_c=0.0)
    T_high_n = t_end_under_constant_controls(model, V_h=1.0, V_n=3.5, V_c=0.0)
    assert T_low_n >= T_high_n + 0.10, (
        f"V_n response too weak: T(V_n=0.3)={T_low_n:.3f}, "
        f"T(V_n=3.5)={T_high_n:.3f}, "
        f"difference {T_low_n - T_high_n:+.3f} below required +0.10"
    )

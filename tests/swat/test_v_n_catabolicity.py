"""Test 2 — V_n is monotonically catabolic across the full V_n range.

Under Option C v4 (Option D) with multiplicative V_n dampener
damp(V_n) = exp(−V_n / V_n_scale), V_n is monotonically attenuating
on E_dyn — any V_n > 0 reduces T. Issue #5.

Pre-fix vendored snapshot: FAILS (T_end is non-monotonic in V_n).
Refined Option C v3 (no damp): partially passes (monotonic in [1, 5]
only, flat in [0, 1]).
Option C v4 / Option D (with damp): PASSES — monotonic across [0, 5].

Sweep V_n ∈ {0, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0} at V_h=1.0, V_c=0.
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
            f"V_n is NOT catabolic: at step {i} "
            f"({V_N_GRID[i - 1]} → {V_N_GRID[i]}), "
            f"T_end rose {Ts[i - 1]:.3f} → {Ts[i]:.3f}. "
            f"Full grid: V_n={V_N_GRID}, T={[round(t, 3) for t in Ts]}"
        )


def test_v_n_clinically_meaningful_low_range_response(model):
    """V_n in the low range should produce a measurable dampening.

    Under Option D, T(V_n=0) − T(V_n=0.5) should be > 0.05 (clinically
    meaningful). Pre-D, T was flat across V_n ∈ [0, 1].
    """
    T_zero = t_end_under_constant_controls(model, V_h=1.0, V_n=0.0, V_c=0.0)
    T_low  = t_end_under_constant_controls(model, V_h=1.0, V_n=0.5, V_c=0.0)
    assert T_zero >= T_low + 0.05, (
        f"V_n at low values has no measurable effect: "
        f"T(V_n=0)={T_zero:.3f}, T(V_n=0.5)={T_low:.3f}, "
        f"difference {T_zero - T_low:+.3f} below required +0.05. "
        f"Issue #5 — the V_n dampener may be too weak."
    )


def test_v_n_clinically_meaningful_high_load_response(model):
    """T_end at V_n=1.0 must exceed T_end at V_n=5.0 by at least 0.20."""
    T_mod  = t_end_under_constant_controls(model, V_h=1.0, V_n=1.0, V_c=0.0)
    T_high = t_end_under_constant_controls(model, V_h=1.0, V_n=5.0, V_c=0.0)
    assert T_mod >= T_high + 0.20, (
        f"V_n high-load response too weak: T(V_n=1.0)={T_mod:.3f}, "
        f"T(V_n=5.0)={T_high:.3f}, "
        f"difference {T_mod - T_high:+.3f} below required +0.20"
    )

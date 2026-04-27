"""Test 5 — healthy equilibrium reaches its claimed value.

Run scenario A (V_h=1, V_n=0.3, V_c=0, T_0=0.5) for D=30 days (15 tau_T)
and assert deterministic equilibrium T ∈ [0.50, 0.60], matching the
spec's T* ≈ 0.55 claim.

Pre-fix vendored snapshot: FAILS (deterministic equilibrium ≈ 0.46).
Refined Option C: should PASS (after any required parameter tuning).
"""
import pytest
from model_validation.runner import t_end_under_constant_controls


def test_healthy_equilibrium_in_band(model):
    T_eq = t_end_under_constant_controls(
        model, V_h=1.0, V_n=0.3, V_c=0.0, T_0=0.5, D=30.0
    )
    print(f"\n  Healthy deterministic equilibrium T_end (D=30): {T_eq:.4f}")
    assert 0.50 <= T_eq <= 0.60, (
        f"Healthy equilibrium {T_eq:.4f} outside [0.50, 0.60]. "
        f"Spec claims T* ≈ 0.55. Either the spec is stale, or the model "
        f"is wrong, or the parameters need re-tuning."
    )

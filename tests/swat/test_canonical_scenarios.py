"""Test 6 — canonical four scenarios produce expected outcomes.

Healthy / Insomnia / Recovery / Shift-work, per upstream PARAM_SET_A-D.
"""
import pytest
from model_validation.runner import t_end_under_constant_controls


@pytest.mark.parametrize("name, V_h, V_n, V_c, T_0, lo, hi", [
    ("A_healthy",   1.0, 0.3, 0.0, 0.5, 0.45, 0.65),
    ("B_insomnia",  0.2, 3.5, 0.0, 0.5, 0.00, 0.20),
    ("C_recovery",  1.0, 0.3, 0.0, 0.05, 0.30, 0.65),
    ("D_shift",     1.0, 0.3, 6.0, 0.5, 0.00, 0.20),
])
def test_canonical_scenario(model, name, V_h, V_n, V_c, T_0, lo, hi):
    T_end = t_end_under_constant_controls(
        model, V_h=V_h, V_n=V_n, V_c=V_c, T_0=T_0
    )
    print(f"\n  Scenario {name}: V_h={V_h}, V_n={V_n}, V_c={V_c}, "
          f"T_0={T_0} -> T_end={T_end:.4f}")
    assert lo <= T_end <= hi, (
        f"Scenario {name} T_end={T_end:.4f} outside [{lo}, {hi}]"
    )

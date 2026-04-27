"""Test 6 — canonical four scenarios produce expected outcomes.

Healthy / Insomnia / Recovery / Shift-work, per upstream PARAM_SET_A-D.
"""
import pytest
from model_validation.runner import t_end_under_constant_controls


@pytest.mark.parametrize("name, V_h, V_n, V_c, T_0, lo, hi", [
    # Healthy: V_n=0 (clean signal — focus on V_h's role). Under refined
    # Option C with calibrated λ_amp values, healthy V_h=1 should give
    # E ≈ 1 throughout, μ ≈ μ_0 + μ_E = 0.5, T*=√(0.5/0.5)=1.0.
    ("A_healthy",   1.0, 0.0, 0.0, 0.50, 0.50, 1.05),
    # Insomnia: pathological. Should collapse to noise floor.
    ("B_insomnia",  0.2, 3.5, 0.0, 0.50, 0.00, 0.20),
    # Recovery from sick T_0 at healthy V_h=1, V_n=0 (clean). With E ≈ 1
    # and μ ≈ 0.5, growth from T_0=0.05 over 14 days should be substantial.
    ("C_recovery",  1.0, 0.0, 0.0, 0.05, 0.50, 1.05),
    # Shift work: V_c=6 zeroes phase-quality → no entrainment → collapse.
    ("D_shift",     1.0, 0.0, 6.0, 0.50, 0.00, 0.20),
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

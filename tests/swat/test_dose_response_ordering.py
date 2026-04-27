"""Test 3 — clinical dose-response ordering.

T_end(robust) > T_end(healthy) > T_end(stressed) > T_end(insomnia).

  robust   = (V_h=2.0, V_n=0.1)  — high vitality, low load
  healthy  = (V_h=1.0, V_n=0.3)  — canonical Set A
  stressed = (V_h=0.5, V_n=1.0)  — low vitality, high load
  insomnia = (V_h=0.2, V_n=3.5)  — depleted vitality, severe load
"""
import pytest
from model_validation.runner import t_end_under_constant_controls


def test_dose_response_ordering(model):
    T_robust   = t_end_under_constant_controls(model, V_h=2.0, V_n=0.1, V_c=0.0)
    T_healthy  = t_end_under_constant_controls(model, V_h=1.0, V_n=0.3, V_c=0.0)
    T_stressed = t_end_under_constant_controls(model, V_h=0.5, V_n=1.0, V_c=0.0)
    T_insomnia = t_end_under_constant_controls(model, V_h=0.2, V_n=3.5, V_c=0.0)
    print(f"\n  robust   {T_robust:.3f}")
    print(f"  healthy  {T_healthy:.3f}")
    print(f"  stressed {T_stressed:.3f}")
    print(f"  insomnia {T_insomnia:.3f}")
    assert T_robust > T_healthy > T_stressed > T_insomnia, (
        f"Dose-response wrong: "
        f"robust={T_robust:.3f}, healthy={T_healthy:.3f}, "
        f"stressed={T_stressed:.3f}, insomnia={T_insomnia:.3f}"
    )

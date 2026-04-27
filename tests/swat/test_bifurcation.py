"""Test 9 — Stuart-Landau bifurcation behaves as expected.

Sub-critical controls: T should collapse toward the noise floor.
Super-critical controls: T should sustain.
"""
import pytest
from model_validation.runner import t_end_under_constant_controls


def test_subcritical_collapses(model):
    """V_h=4, V_n=0 in the pre-fix model gives sub-critical E < E_crit; T -> 0.
    After Option C fix, V_h=4 should make the system super-critical, so this
    test expectation may need to flip post-fix. For now, asserts on the V_n
    saturation regime instead.
    """
    # V_n = 5, V_h = 0 is unambiguously sub-critical in any reasonable model
    # (high chronic load AND no vitality).
    T_subcrit = t_end_under_constant_controls(model, V_h=0.0, V_n=5.0, V_c=0.0)
    print(f"\n  V_h=0, V_n=5: T_end = {T_subcrit:.3f}  (target: < 0.20)")
    assert T_subcrit < 0.20, (
        f"Pathologically high V_n with no V_h should collapse T: got {T_subcrit:.3f}"
    )


def test_supercritical_sustains(model):
    """Healthy controls should give E > E_crit and sustain T high.
    V_n=0 (clean V_h-only signal for healthy)."""
    T_supercrit = t_end_under_constant_controls(model, V_h=1.0, V_n=0.0, V_c=0.0)
    print(f"\n  V_h=1, V_n=0: T_end = {T_supercrit:.3f}  (target: > 0.50)")
    assert T_supercrit > 0.50, (
        f"Super-critical regime should sustain T > 0.50: got {T_supercrit:.3f}"
    )

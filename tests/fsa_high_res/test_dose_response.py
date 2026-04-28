"""Are the model's responses smooth in the prescription?

Clinician's question being tested
---------------------------------
"If I make a small adjustment to my patient's training prescription
— say, raise the fitness target by 10% — does the model predict a
correspondingly small change in outcome? Or does it lurch
discontinuously?"

A model whose predicted outcome jumps wildly from a small
prescription change is unsafe to optimise against: tiny tuning
adjustments — exactly what the optimal-control engine produces —
could move the patient between very different predicted regimes.
Smoothness in (T_B, Phi) is the model's promise that close
prescriptions give close outcomes.

This is distinct from the bifurcation tests: the model is allowed to
have a sharp boundary between healthy and overtrained at high Phi.
What it must NOT have is jumps in B or F that don't correspond to a
biological mechanism. Both fitness and strain pathways must respond
continuously in their healthy operating regime.

What the tests below assert
---------------------------
1. **Fitness responds smoothly to small T_B changes**
   (test_t_b_dose_response_smooth_in_healthy_regime).
   In the healthy training regime, halving the gap between two
   nearby T_B prescriptions must roughly halve the gap in terminal B.

2. **Strain responds smoothly to small Phi changes**
   (test_phi_dose_response_smooth_in_healthy_regime).
   Same property for the strain channel: small Phi differences
   produce proportionally small F differences.

If either fails, the model has a discontinuity in its healthy
operating range that the OT optimiser would happily exploit, leading
to an unstable prescription recommendation.
"""
from .conftest import terminal_state


def test_t_b_dose_response_smooth_in_healthy_regime(trajectory_solver):
    """Small T_B steps in the healthy range produce proportionally small B steps.

    Three prescriptions held constant for 60 days:
      A: T_B = 0.4, Phi = 0.05 — moderate target.
      B: T_B = 0.6, Phi = 0.05 — slightly higher.
      C: T_B = 0.8, Phi = 0.05 — higher still.

    For continuity, B(T_B=0.8) - B(T_B=0.4) should be roughly twice
    B(T_B=0.6) - B(T_B=0.4). Allow generous tolerance because the
    fitness ODE is mildly non-linear (alpha_A * A enters the
    rate); we just want "no order-of-magnitude jumps".
    """
    B_at_lo, _, _ = terminal_state(trajectory_solver, T_B=0.4, Phi=0.05)
    B_at_mid, _, _ = terminal_state(trajectory_solver, T_B=0.6, Phi=0.05)
    B_at_hi, _, _ = terminal_state(trajectory_solver, T_B=0.8, Phi=0.05)

    delta_lo_to_mid = B_at_mid - B_at_lo
    delta_lo_to_hi = B_at_hi - B_at_lo

    # Proportional response: 2x the input gap should give 1.6x to 2.4x
    # the response gap (40% slack on linearity).
    ratio = delta_lo_to_hi / max(delta_lo_to_mid, 1e-9)
    assert 1.6 < ratio < 2.4, (
        f"T_B response should be roughly proportional. "
        f"B(0.4) -> B(0.6): {delta_lo_to_mid:.4f}; "
        f"B(0.4) -> B(0.8): {delta_lo_to_hi:.4f}; "
        f"ratio = {ratio:.2f} (expected ~2)."
    )


def test_phi_dose_response_smooth_in_healthy_regime(trajectory_solver):
    """Small Phi steps in the healthy training range give proportional F steps.

    Three prescriptions held constant for 60 days, all under a
    moderate fitness target T_B = 0.5:
      A: Phi = 0.1.
      B: Phi = 0.2.
      C: Phi = 0.3.

    The strain response in this low range should be roughly linear in
    Phi (CIR drain dominates).
    """
    _, F_at_lo, _ = terminal_state(trajectory_solver, T_B=0.5, Phi=0.1)
    _, F_at_mid, _ = terminal_state(trajectory_solver, T_B=0.5, Phi=0.2)
    _, F_at_hi, _ = terminal_state(trajectory_solver, T_B=0.5, Phi=0.3)

    delta_lo_to_mid = F_at_mid - F_at_lo
    delta_lo_to_hi = F_at_hi - F_at_lo

    ratio = delta_lo_to_hi / max(delta_lo_to_mid, 1e-9)
    # Phi response: 2x input gap should produce ~2x response gap.
    # Wide tolerance (1.5 to 3.0) because:
    #   - F appears quadratically in mu(B, F), so larger F drags mu
    #     down, which drops A, which makes the strain-drain rate
    #     (which depends on lambda_A * A) slower, which raises F
    #     further at equilibrium. Mild super-linearity is the model's
    #     correct behaviour, not a discontinuity.
    #   - We catch order-of-magnitude jumps that would indicate a
    #     real discontinuity, not subtle non-linearity.
    assert 1.5 < ratio < 3.0, (
        f"Phi response should be roughly proportional in the healthy "
        f"low-strain range (no order-of-magnitude jump). "
        f"F(0.1) -> F(0.2): {delta_lo_to_mid:.4f}; "
        f"F(0.1) -> F(0.3): {delta_lo_to_hi:.4f}; "
        f"ratio = {ratio:.2f} (expected ~2, allowed 1.5–3.0)."
    )

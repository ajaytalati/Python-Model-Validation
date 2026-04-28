"""Are the simulator's responses to prescription changes free of discontinuities?

Question being tested
---------------------
The drift equations are smooth functions of (T_B, Phi). A correct
ODE solver should therefore produce a smooth response of the
terminal state to small prescription changes. These tests catch
solver / numerical issues that would introduce non-physical jumps
in the response within the healthy operating range.

The model is allowed (by design) to have sharp regime changes at
the bifurcation boundary (the overtraining cliff). The tests here
probe smoothness *within* the healthy regime, where the dynamics
are guaranteed-smooth by construction.

What these tests catch
----------------------
- Solver instability in the eps-regularised diffusion at low B
  or F (a buggy integrator could blow up or mis-step).
- An equation regression that introduces a non-smooth step
  function in the drift.
- Numerical issues that the FIM analysis would also be sensitive
  to, because both rely on smooth derivatives.

What these tests do NOT establish
---------------------------------
- That the response magnitude matches reality.
- That the design choice of where to place the smooth-vs-cliff
  boundary corresponds to the real overtraining boundary.

A note on tolerance: the linearity tolerance is wide (1.5–3.0×
ratio for a 2× input gap) because the drift is mildly non-linear
by design (F^2 in mu, A in tau-effective rates). What we're
catching is order-of-magnitude jumps, not subtle non-linearity.
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

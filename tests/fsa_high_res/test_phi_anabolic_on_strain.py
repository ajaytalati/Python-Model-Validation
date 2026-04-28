"""Is Phi designed-anabolic on strain, and does the deployed parameter set preserve that?

Question being tested
---------------------
The model's design intent is that raising Phi (training intensity)
drives F (simulated strain) up, and that Phi = 0 lets pre-existing
F drain back toward zero. The drift `dF/dt = Phi − (drain_rate) · F`
is structurally anabolic on F for any positive drain rate. The test
confirms the deployed parameter set preserves the designed sign on
both axes (strain accumulation under positive Phi, strain drain
under zero Phi).

What this test catches
----------------------
- A parameter regression that makes `tau_F` or one of its
  multiplicative coupling coefficients (`lambda_B`, `lambda_A`)
  produce a negative drain rate.
- An equation regression where Phi is dropped or sign-flipped.

What this test does NOT establish
---------------------------------
- That F as defined here corresponds to any real-world measure of
  accumulated training stress.
- That the magnitudes match real-world dose-response data.
"""
from .conftest import terminal_state


def test_phi_drives_strain_up(trajectory_solver):
    """Higher prescribed training intensity → more residual strain.

    Patient phenotype: moderately fit, low residual strain
    (B_0 = 0.5, F_0 = 0.1, A_0 = 0.5).
    Two prescriptions held constant over 60 days: Phi = 0.0 vs Phi = 1.5.
    Fitness target held the same (T_B = 0.5) so we isolate the Phi → F
    pathway.

    Assertion: after two months the high-intensity arm has carried at
    least 0.5 more strain than the rest arm (allowing the system to
    have reached its respective equilibria).
    """
    import jax.numpy as jnp
    patient = jnp.array([0.5, 0.1, 0.5])
    _, F_lo, _ = terminal_state(trajectory_solver, T_B=0.5, Phi=0.0,
                                 init_state=patient)
    _, F_hi, _ = terminal_state(trajectory_solver, T_B=0.5, Phi=1.5,
                                 init_state=patient)
    assert F_hi > F_lo + 0.5, (
        f"Higher Phi should raise residual strain. "
        f"Got F(Phi=0.0) = {F_lo:.3f}, F(Phi=1.5) = {F_hi:.3f}."
    )


def test_zero_phi_drains_strain(trajectory_solver):
    """Pure rest (Phi = 0) drains pre-existing strain over time.

    Patient phenotype: arrived overtrained — moderate fitness and
    high residual strain (B_0 = 0.4, F_0 = 0.8, A_0 = 0.05).
    Prescribed: Phi = 0 (full rest from training), T_B = 0.4 (hold
    fitness target at current level so we isolate the strain channel),
    held constant for 60 days.

    Assertion: terminal strain F(D) is at least 0.6 lower than the
    starting strain — the patient has shed most of the strain.
    """
    import jax.numpy as jnp
    overstrained = jnp.array([0.4, 0.8, 0.05])
    _, F_end, _ = terminal_state(trajectory_solver, T_B=0.4, Phi=0.0,
                                  init_state=overstrained)
    assert F_end < 0.8 - 0.6, (
        f"Resting patient should shed strain over 60 days. "
        f"Started F = 0.8, ended F = {F_end:.3f}."
    )

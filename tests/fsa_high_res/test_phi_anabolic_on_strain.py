"""Does prescribing higher training intensity (Phi) actually produce strain?

Clinician's question being tested
---------------------------------
"When I prescribe higher training intensity for my patient, does the
model say their accumulated strain will rise over time?"

Strain (F) is the model's representation of accumulated training load
not yet absorbed into fitness — the residue of hard sessions that the
body is still recovering from. Acute moderate strain is normal during
training; chronic high strain is the precursor to overtraining.
Clinicians need the model to say: "if I tell the patient to train
harder (raise Phi), the simulated strain rises". Otherwise the
model's notion of strain is decoupled from the prescribed intensity
and any overtraining-risk prediction is meaningless.

What the tests below assert
---------------------------
1. **Higher Phi raises terminal strain** (test_phi_drives_strain_up).
   The exact same patient under low-intensity vs high-intensity
   prescriptions must end with the high-intensity arm carrying more
   residual strain.

2. **Zero Phi lets strain drain** (test_zero_phi_drains_strain).
   A patient parked at zero training intensity must shed any
   pre-existing strain over time. Convalescence, in clinical terms.

If either fails, the strain channel is structurally disconnected from
the intensity prescription, which would break every overtraining-risk
calculation downstream.
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

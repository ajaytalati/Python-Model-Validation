"""Does prescribing a higher fitness target (T_B) actually build fitness?

Clinician's question being tested
---------------------------------
"When I prescribe a higher training-load target for my patient, does the
model say their fitness will climb over time?"

This is the most basic clinical sanity check on the fitness pathway. A
graded reconditioning programme — the canonical use case for the FSA
model — tells the patient to target progressively higher training
loads. If raising the target (T_B) didn't produce more fitness (B),
every reconditioning prescription the model justifies would be
backwards.

What the tests below assert
---------------------------
1. **Higher T_B builds more fitness** (test_t_b_drives_fitness_up).
   A deconditioned patient simulated for two months under "no training
   target" must end less fit than the same patient under "max
   training target".

2. **Zero T_B drains fitness** (test_zero_t_b_drains_fitness).
   A previously-fit patient parked at "no training target" must lose
   fitness over time. Sedentary deconditioning, in clinical terms.

If either fails, the model's first-order prescription pathway is
broken and downstream optimal-control recommendations cannot be
trusted on this axis.
"""
from .conftest import terminal_state


def test_t_b_drives_fitness_up(trajectory_solver):
    """Higher prescribed training-load target → higher achieved fitness.

    Patient phenotype: deconditioned (B_0 = 0.05, F_0 = 0.05, A_0 = 0.30).
    Two prescriptions held constant over 60 days: T_B = 0 vs T_B = 1.
    Strain prescription is held the same low value (Phi = 0.05) in both
    arms so we isolate the T_B → B pathway.

    Assertion: after two months the high-T_B arm has at least 0.4 more
    fitness than the no-training arm.
    """
    import jax.numpy as jnp
    deconditioned = jnp.array([0.05, 0.05, 0.30])
    B_lo, _, _ = terminal_state(trajectory_solver, T_B=0.0, Phi=0.05,
                                 init_state=deconditioned)
    B_hi, _, _ = terminal_state(trajectory_solver, T_B=1.0, Phi=0.05,
                                 init_state=deconditioned)
    assert B_hi > B_lo + 0.4, (
        f"Higher T_B should build more fitness over 60 days. "
        f"Got B(T_B=0) = {B_lo:.3f}, B(T_B=1) = {B_hi:.3f}."
    )


def test_zero_t_b_drains_fitness(trajectory_solver):
    """No training target → fitness drains over time (sedentary deconditioning).

    Patient phenotype: previously fit (B_0 = 0.7, F_0 = 0.05, A_0 = 0.7).
    Prescribed: T_B = 0 (no training-load target), Phi = 0 (no strain),
    held constant for 60 days.

    Assertion: terminal fitness B(D) is at least 0.5 lower than the
    starting fitness — the patient has clearly deconditioned.
    """
    import jax.numpy as jnp
    fit_patient = jnp.array([0.7, 0.05, 0.7])
    B_end, _, _ = terminal_state(trajectory_solver, T_B=0.0, Phi=0.0,
                                  init_state=fit_patient)
    assert B_end < 0.7 - 0.5, (
        f"Sedentary patient should lose fitness over 60 days. "
        f"Started B = 0.7, ended B = {B_end:.3f}."
    )

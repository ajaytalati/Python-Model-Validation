"""Is T_B designed-anabolic on fitness, and does the deployed parameter set preserve that?

Question being tested
---------------------
The model's design intent is that raising T_B (the prescribed
training-load target) drives B (simulated fitness) up over time.
This is encoded in the drift `dB/dt = (1 + alpha_A·A)/tau_B · (T_B − B)`,
which is anabolic on B for any positive `tau_B`. The test confirms
the deployed parameter set preserves the designed sign.

What this test catches
----------------------
- A parameter regression that flips `tau_B` negative (which would
  make T_B catabolic on B).
- An equation regression where T_B is dropped from the drift or
  enters with the wrong sign.

What this test does NOT establish
---------------------------------
- That `tau_B = 14 days` matches the real fitness-adaptation
  timescale of any actual patient population.
- That this designed pathway is what real training does to real
  fitness.
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

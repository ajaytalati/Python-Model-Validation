"""Do the parameter signs in mu(B, F) preserve the designed argmax structure?

Question being tested
---------------------
The model is designed so that `mu(B, F) = mu_0 + mu_B·B − mu_F·F − mu_FF·F^2`
is maximised when B is high and F is low. The terminal-amplitude
landscape over (T_B, Phi) prescriptions therefore has its argmax in
the high-T_B, low-Phi region (T_B drives B up, low Phi keeps F
small). The tests confirm the deployed parameter signs preserve this
designed argmax structure.

What these tests catch
----------------------
- A parameter regression that flips a sign (mu_B turns catabolic, or
  mu_F / mu_FF turn anabolic, or mu_0 dominates wrongly).
- An equation regression that drops one of the four mu terms.
- A solver regression that produces noisy terminal-A values
  reordering the (T_B, Phi) ranking.

What these tests do NOT establish
---------------------------------
- That "high training-load target with low strain" is the
  empirically optimal prescription for any real patient population.
- That the gap magnitudes (e.g. 0.7 difference between healthy and
  overtrained corners) correspond to clinically meaningful
  differences in real biomarker outcomes.
"""
import jax.numpy as jnp

from .conftest import terminal_amplitude


def test_argmax_lies_in_high_t_b_low_phi_region(trajectory_solver):
    """The (T_B, Phi) corner that maximises A is the clinically-correct one.

    Sweep a 5x5 grid of (T_B, Phi) values, run each prescription for
    60 days from the healthy reference state, record terminal A.
    Identify the argmax cell. Assert it sits in the
    (T_B in [0.5, 1.0], Phi in [0.0, 1.0]) region — the model's
    healthy training quadrant.
    """
    T_B_grid = jnp.linspace(0.0, 1.0, 5)
    Phi_grid = jnp.linspace(0.0, 2.0, 5)

    A_grid = []
    for T_B in T_B_grid:
        row = []
        for Phi in Phi_grid:
            row.append(terminal_amplitude(trajectory_solver,
                                            T_B=float(T_B), Phi=float(Phi)))
        A_grid.append(row)
    A_grid = jnp.asarray(A_grid)

    argmax_flat = int(jnp.argmax(A_grid))
    i_T_B, i_Phi = divmod(argmax_flat, len(Phi_grid))
    best_T_B = float(T_B_grid[i_T_B])
    best_Phi = float(Phi_grid[i_Phi])

    assert 0.5 <= best_T_B <= 1.0, (
        f"Argmax T_B = {best_T_B} should be in the high-fitness-target "
        f"half [0.5, 1.0]. A grid:\n{A_grid}"
    )
    assert 0.0 <= best_Phi <= 1.0, (
        f"Argmax Phi = {best_Phi} should be in the low-strain "
        f"half [0.0, 1.0]. A grid:\n{A_grid}"
    )


def test_healthy_reference_beats_doing_nothing(trajectory_solver):
    """Modest training (T_B=0.5, Phi=0.05) beats no training (T_B=0, Phi=0).

    Catches a model where the optimiser would happily recommend "do
    nothing" because the predicted endocrine outcome is just as good
    as healthy moderate training.
    """
    A_healthy_ref = terminal_amplitude(trajectory_solver, T_B=0.5, Phi=0.05)
    A_do_nothing = terminal_amplitude(trajectory_solver, T_B=0.0, Phi=0.0)

    assert A_healthy_ref > A_do_nothing + 0.3, (
        f"Modest healthy training should beat doing nothing by at least "
        f"0.3 in A(D). Got healthy ref = {A_healthy_ref:.3f}, "
        f"do-nothing = {A_do_nothing:.3f}."
    )


def test_overtrained_corner_is_much_worse_than_healthy(trajectory_solver):
    """Overtrained corner (T_B=1, Phi=2) is a much worse outcome than healthy.

    A model that fails to clearly discriminate between a sustainable
    training plan and an overtraining plan would let the optimiser
    pick whichever side has lower transport-cost — a clinical
    disaster.
    """
    A_healthy = terminal_amplitude(trajectory_solver, T_B=0.5, Phi=0.05)
    A_overtrained = terminal_amplitude(trajectory_solver, T_B=1.0, Phi=2.0)

    assert A_healthy > A_overtrained + 0.7, (
        f"Healthy regime should yield much higher A than the "
        f"overtrained corner. Got healthy = {A_healthy:.3f}, "
        f"overtrained = {A_overtrained:.3f}."
    )

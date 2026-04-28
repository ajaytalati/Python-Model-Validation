"""Does the F^2 term in mu(B, F) actually produce an amplitude cliff?

Question being tested
---------------------
The model's design encodes an overtraining cliff via the `−mu_FF·F^2`
term in `mu(B, F)` (the F^2 dominates at high F and pulls mu
negative, which collapses A through the Stuart-Landau dynamics).
The tests confirm two structural consequences of that design:

1. At max prescribed intensity (Phi = 2), the simulated amplitude
   collapses regardless of fitness target.
2. Holding T_B fixed, raising Phi only ever reduces A (no rise then
   fall on the Phi-only axis — the rise-then-fall intuition lives on
   the coupled (T_B, Phi) axis, which is not probed here).

What this test catches
----------------------
- A parameter regression that sets `mu_FF` to zero or wrong sign
  (the cliff disappears).
- An equation regression that drops the F^2 term.
- A solver regression that produces non-monotonic noise on a
  designed-monotonic axis.

What this test does NOT establish
---------------------------------
- That the cliff threshold corresponds to any real-world overtraining
  threshold.
- That the cliff shape (sharp vs gradual) matches real-world data.
"""
import jax.numpy as jnp

from .conftest import terminal_amplitude


def test_high_phi_collapses_amplitude(trajectory_solver):
    """Sustained max Phi produces near-zero terminal amplitude.

    Two prescriptions, both with Phi = 2 (the upper control bound) held
    constant for 60 days:
      - Patient A: target T_B = 0 (sedentary aiming).
      - Patient B: target T_B = 1 (athlete aiming).
    Both patients start at the healthy reference state.

    Assertion: regardless of T_B, A(D) collapses below 0.05 — the
    overtraining cliff is independent of fitness target.
    """
    A_path_low_target = terminal_amplitude(trajectory_solver, T_B=0.0, Phi=2.0)
    A_path_high_target = terminal_amplitude(trajectory_solver, T_B=1.0, Phi=2.0)
    assert A_path_low_target < 0.05, (
        f"At Phi = 2 with T_B = 0, terminal A should collapse below 0.05. "
        f"Got A(D) = {A_path_low_target:.4f}."
    )
    assert A_path_high_target < 0.05, (
        f"At Phi = 2 with T_B = 1 (overtrained athlete), terminal A "
        f"should still collapse below 0.05. Got A(D) = {A_path_high_target:.4f}."
    )


def test_phi_is_monotonically_catabolic_on_amplitude(trajectory_solver):
    """At fixed fitness target, more strain monotonically reduces amplitude.

    Sweep Phi at 9 points across [0, 2] with T_B = 0.5 (moderate
    target) held constant. The terminal amplitude A(D) must:
      - be highest at Phi = 0 (rest, where mu(B, F) is largest because
        F approaches zero),
      - decrease monotonically (or near-monotonically — small
        non-monotonic blips of < 0.05 are tolerated as numerical
        noise, but no rises larger than that),
      - collapse near A = 0 by Phi = 2 (the overtraining cliff).

    Clinical statement: "for a fixed fitness target, adding more
    training strain only makes endocrine outcome worse." This is the
    Phi-only catabolic axis; the broader "moderate training is best"
    statement requires T_B to rise too (different axis, not tested
    here).
    """
    phi_grid = jnp.linspace(0.0, 2.0, 9)
    A_terminal = [
        terminal_amplitude(trajectory_solver, T_B=0.5, Phi=float(p))
        for p in phi_grid
    ]
    A_arr = jnp.asarray(A_terminal)

    # Peak must be at Phi = 0 (rest) — anything Phi > 0 only adds strain.
    peak_idx = int(jnp.argmax(A_arr))
    assert peak_idx == 0, (
        f"Amplitude response peak should be at Phi = 0 when T_B is "
        f"held fixed. Got peak at index {peak_idx} (Phi = "
        f"{float(phi_grid[peak_idx]):.2f}). Sweep: "
        f"{[float(a) for a in A_arr]}."
    )

    # Monotonic decrease (with small numerical-noise tolerance).
    diffs = jnp.diff(A_arr)
    max_rise = float(jnp.max(diffs))
    assert max_rise < 0.05, (
        f"Amplitude response should be monotonically non-increasing in "
        f"Phi (allowing 0.05 numerical-noise tolerance for rises). "
        f"Largest rise observed: {max_rise:.4f}. Sweep: "
        f"{[float(a) for a in A_arr]}."
    )

    # Phi = 2 must have collapsed amplitude relative to Phi = 0.
    assert float(A_arr[-1]) < float(A_arr[0]) - 0.5, (
        f"Overtraining (Phi = 2) should collapse amplitude well below "
        f"the rest baseline. Phi=0: A = {float(A_arr[0]):.3f}, "
        f"Phi=2: A = {float(A_arr[-1]):.3f}."
    )

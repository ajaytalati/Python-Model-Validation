"""Does the healthy reference prescription drive (B, F, A) to a fixed point?

Question being tested
---------------------
The model is designed to have a stable fixed point under constant
super-critical controls (T_B = 0.5, Phi = 0.05 here): B converges
to T_B, F to its CIR equilibrium, A to the Stuart-Landau attractor
sqrt(mu/eta) at the resulting (B, F). These tests confirm three
structural properties under the deployed parameters:

1. The system settles (state at day 30 ≈ state at day 60).
2. The settled A value is substantially above zero (A > 0.5).
3. The basin of the settled fixed point covers the entire physical
   state space — four diverse starting phenotypes all converge to
   the same A value within 0.05.

What these tests catch
----------------------
- A parameter regression that produces a limit cycle instead of a
  fixed point (test 1 fails).
- A parameter regression that crosses mu sub-critical at the
  healthy reference (test 2 fails — A collapses to zero).
- A parameter regression that creates multiple attractors
  inside the physical state space (test 3 fails — different
  starting points end at different A values).

What these tests do NOT establish
---------------------------------
- That the deployed (T_B = 0.5, Phi = 0.05) point is "healthy" in
  any clinical sense — that's the model's design assertion.
- That the settled A value (~0.89 in current parameter set)
  corresponds to any real biomarker amplitude.
- That the basin of attraction covers the realistic patient
  parameter space — only that it covers the deployed (B, F, A)
  state space at the deployed parameter point.
"""
import jax.numpy as jnp

from .conftest import D_DAYS, INIT_STATE_HEALTHY, terminal_state


def test_healthy_state_converges_to_fixed_point(trajectory_solver):
    """Day-30 and day-60 state under healthy controls are essentially the same.

    If the model wandered or oscillated under a constant healthy
    prescription, the state at day 30 and at day 60 would be
    materially different. Under a stable fixed point they are not.

    Tolerance 0.02 (1-2% across each component): the slowest
    physiological timescale is tau_B = 14 days, so 30→60 days is
    ~2 more time-constants of refinement on the still-converging
    components (especially A, which tracks mu(B, F) and so settles
    after B has). 0.02 catches "still oscillating" or "drifting",
    accepts "asymptotically settling".
    """
    _, traj = trajectory_solver(0.5, 0.05, INIT_STATE_HEALTHY, D_DAYS)
    n_pts = traj.shape[0]
    half_idx = n_pts // 2

    state_at_30d = traj[half_idx]
    state_at_60d = traj[-1]
    drift = jnp.linalg.norm(state_at_30d - state_at_60d)

    assert float(drift) < 0.02, (
        f"Healthy state should be settled by day 30 to within 2%. "
        f"State at 30d: {state_at_30d}, at 60d: {state_at_60d}. "
        f"L2 drift = {float(drift):.4f}."
    )


def test_healthy_settled_state_has_high_amplitude(trajectory_solver):
    """Under healthy controls, terminal A should be substantially positive.

    If the model has the patient settling at A near zero under a
    healthy prescription, every "good clinical state" claim is empty.
    """
    _, _, A_end = terminal_state(trajectory_solver, T_B=0.5, Phi=0.05)
    assert A_end > 0.5, (
        f"Healthy reference should yield A(D) > 0.5. Got A(D) = {A_end:.3f}."
    )


def test_healthy_basin_covers_physical_state_space(trajectory_solver):
    """Four diverse starting states under healthy controls converge to the same A.

    Patient phenotypes:
      - sedentary unfit (B_0=0.05, F_0=0.05, A_0=0.10)
      - moderately fit (B_0=0.5, F_0=0.1, A_0=0.5)
      - athlete with residual strain (B_0=0.7, F_0=0.5, A_0=0.6)
      - depleted convalescent (B_0=0.2, F_0=0.0, A_0=0.05)

    All run under the healthy reference (T_B=0.5, Phi=0.05) for 60
    days. All must end with A within 0.05 of each other.
    """
    starts = [
        ("sedentary",    jnp.array([0.05, 0.05, 0.10])),
        ("fit",          jnp.array([0.50, 0.10, 0.50])),
        ("strained",     jnp.array([0.70, 0.50, 0.60])),
        ("convalescent", jnp.array([0.20, 0.00, 0.05])),
    ]
    A_ends = {}
    for name, init in starts:
        _, _, A = terminal_state(trajectory_solver, T_B=0.5, Phi=0.05,
                                  init_state=init)
        A_ends[name] = A

    A_values = list(A_ends.values())
    spread = max(A_values) - min(A_values)
    assert spread < 0.05, (
        f"All starting states should converge to the same healthy "
        f"attractor under the healthy prescription. "
        f"Got terminal A(D) spread {spread:.4f}: {A_ends}."
    )

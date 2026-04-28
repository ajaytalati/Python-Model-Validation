"""Test fixtures for the FSA-high-res structural gating suite.

Each test in this directory probes one designed structural property
of the model — a property the model's *equations + deployed parameters*
should produce by design. The pattern across all tests:

    1. Set the patient phenotype (initial state).
    2. Apply a constant training schedule (T_B, Phi) for D days.
    3. Simulate the model deterministically.
    4. Assert that the simulated outcome matches what the model's
       design intent says it should be.

What these tests catch
----------------------
- Parameter-sign errors (e.g. someone flips the sign of mu_B and
  fitness becomes catabolic on amplitude).
- Equation regressions (e.g. someone drops the F^2 term and the
  overtraining cliff disappears).
- Solver bugs (e.g. an integrator that fails on the eps-regularised
  diffusion silently produces NaN trajectories).

What these tests DO NOT catch
-----------------------------
- Whether the model agrees with reality. The tests check the model
  conforms to its own design intent under its deployed parameter
  values — they do *not* compare against real patient data, real
  training logs, published timescales, or expert clinical review.
  A model that internally passes every test in this suite can still
  be biophysically wrong in ways the suite cannot detect.
- Whether the magnitudes (timescales, coupling coefficients,
  thresholds) match real physiology.
- Behaviour outside the deployed parameter point.

This is the structural / mathematical layer of validation only.
True clinical validation is empirical and is out of scope for this
repo. See `how_to_add_a_new_validation_model/02_validation_contract.md`
for the canonical scope statement.

Convention: D = 60 days throughout. That's about 4× the slowest
model timescale (tau_B = 14 days), long enough for the deterministic
dynamics to settle into their fixed point under constant controls.

Note on the harness
-------------------
The package-level runner (`model_validation.runner.t_end_under_constant_controls`)
is shaped for SWAT's three controls (V_h, V_n, V_c). FSA's two
controls (T_B, Phi) need a parallel helper, defined here. Generalising
the runner to take an arbitrary control dictionary is a separate
refactor — flagged but out of scope for these tests.
"""
from __future__ import annotations

from typing import Tuple

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from model_validation.models.fsa_high_res import vendored_model
from model_validation.models.fsa_high_res.vendored_dynamics import (
    fsa_drift,
    _bifurcation_parameter,
)


# Healthy reference initial state — same point the OT-Control adapter
# uses to derive its model-derived target pool, and the same one the
# FIM and Lyapunov analyses run from. Means: a moderately fit patient
# with low residual strain and healthy endocrine pulsatility.
INIT_STATE_HEALTHY = jnp.array([0.3, 0.05, 0.4])

# Default horizon. Long enough for the slowest mode (B-equilibration
# on tau_B = 14 days) to have fully settled.
D_DAYS = 60.0


@pytest.fixture(scope="session")
def fsa_model():
    """The vendored FSA-high-res model — drift, diffusion, parameters."""
    return vendored_model()


@pytest.fixture(scope="session")
def trajectory_solver(fsa_model):
    """Deterministic simulation harness for FSA gating tests.

    Returns a callable:

        solve(T_B, Phi, init_state, D) -> (t_grid, trajectory)

    where the trajectory has shape (n_pts, 3) holding (B, F, A) at each
    point on the time grid. Solver is JIT-compiled once per session.
    """
    params = fsa_model.params
    n_pts = 600

    def _drift(t, y, args):
        T_B, Phi = args
        u = jnp.array([T_B, Phi])
        return fsa_drift(t, y, u, params)

    @jax.jit
    def solve(T_B, Phi, init_state, D):
        t_grid = jnp.linspace(0.0, D, n_pts)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_drift), diffrax.Tsit5(),
            t0=0.0, t1=D, dt0=0.01,
            y0=init_state, args=(T_B, Phi),
            stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
            saveat=diffrax.SaveAt(ts=t_grid),
            max_steps=200_000,
        )
        return t_grid, sol.ys

    return solve


# Convenience helpers used by the gating tests. These keep the test
# files focused on the clinical assertion, not on diffrax plumbing.

def terminal_state(solver, T_B: float, Phi: float,
                   init_state: jnp.ndarray = INIT_STATE_HEALTHY,
                   D: float = D_DAYS) -> Tuple[float, float, float]:
    """Run the model under constant controls for D days, return (B(D), F(D), A(D))."""
    _, traj = solver(T_B, Phi, init_state, D)
    end = np.asarray(traj[-1])
    return float(end[0]), float(end[1]), float(end[2])


def terminal_amplitude(solver, T_B: float, Phi: float,
                       init_state: jnp.ndarray = INIT_STATE_HEALTHY,
                       D: float = D_DAYS) -> float:
    """Just the terminal endocrine amplitude A(D) — the headline clinical readout."""
    _, _, A = terminal_state(solver, T_B, Phi, init_state, D)
    return A


def bifurcation_at_state(B: float, F: float, params: dict) -> float:
    """The bifurcation parameter mu(B, F) at a given (B, F) point.

    mu > 0 means the model predicts a healthy stable amplitude
    A* = sqrt(mu/eta); mu < 0 means amplitude collapses to zero.
    Reads from the vendored dynamics so anything used in the tests
    matches the deployed model exactly.
    """
    return float(_bifurcation_parameter(jnp.asarray(B), jnp.asarray(F), params))

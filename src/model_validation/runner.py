"""Central test fixture: t_end_under_constant_controls.

JAX-native, GPU-accelerated, vmap-friendly. Built on diffrax.

Two backends:
  - 'diffrax_ode': pure-JAX deterministic ODE (Tsit5 for non-stiff, Kvaerno5
    for stiff). Default. ~10-100ms per call after JIT, fully vmap-able.
  - 'diffrax_sde': Euler-Maruyama on the SDE for stochastic tests.

Both operate in DAYS as the time unit. Drift functions handed in must
respect that.

The whole pipeline stays inside JAX, so:
  - GPU-resident if `jax.default_backend() == 'gpu'`
  - vmap over (V_h, V_n, V_c) grids → batched parallel evaluation
  - JIT-compile-once, run-many for parameter sweeps
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

import diffrax

# x64 is essential for physiological models — float32 drift accumulates
# error over thousands of fast-oscillator integration steps.
jax.config.update("jax_enable_x64", True)


@dataclass
class ModelInterface:
    drift: Callable
    diffusion: Callable
    params: Dict[str, float]
    init_state: np.ndarray
    amplitude_index: int = 3
    state_clip: Optional[Callable] = None
    name: str = "unnamed"


# =========================================================================
# DETERMINISTIC ODE — diffrax Tsit5/Kvaerno5
# =========================================================================

def _make_ode_solve(model: ModelInterface, D: float):
    """Return a JIT-compiled function (V_h, V_n, V_c, T_0) -> T_end_mean.

    The function is closed over the model & D, with T_0 substituted into
    the init state at amplitude_index. The other state components come
    from model.init_state.

    Uses diffrax.Kvaerno5 (5th-order BDF-like implicit) which handles
    SWAT's stiffness from lambda=32 well. Kvaerno5 is also amenable to
    vmap and JIT.
    """
    drift = model.drift
    params = model.params
    init = jnp.asarray(model.init_state)
    amp_idx = model.amplitude_index

    # Diffrax wants the vector field as f(t, y, args) -> dy/dt.
    def vector_field(t, y, args):
        u = args                                  # (V_h, V_n, V_c)
        return drift(t, y, u, params)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
    # Save dense output near t=D so we can compute the last-day mean.
    saveat = diffrax.SaveAt(ts=jnp.linspace(D - 1.0, D, 32))

    def solve(V_h, V_n, V_c, T_0):
        u = jnp.array([V_h, V_n, V_c])
        y0 = init.at[amp_idx].set(T_0)
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=D, dt0=0.01,
            y0=y0, args=u,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=50_000,
        )
        # sol.ys: (32, dim_state). Take mean of T over the last day.
        return jnp.mean(sol.ys[:, amp_idx])

    # JIT once; vmap-friendly.
    return jax.jit(solve)


# =========================================================================
# STOCHASTIC SDE — diffrax Euler-Maruyama
# =========================================================================

def _make_sde_solve(model: ModelInterface, D: float, dt: float = 0.005):
    """Return a JIT-compiled function (V_h, V_n, V_c, T_0, rng) -> traj.

    Uses diffrax.Euler with a VirtualBrownianTree for the stochastic
    increments. Same vmap/JIT story.
    """
    drift = model.drift
    diffusion = model.diffusion
    params = model.params
    init = jnp.asarray(model.init_state)
    amp_idx = model.amplitude_index
    dim_state = init.shape[0]

    def drift_fn(t, y, args):
        u = args
        return drift(t, y, u, params)

    # Diagonal noise: returns the same vector regardless of state.
    def diffusion_fn(t, y, args):
        return jnp.diag(diffusion(y, params))     # (d, d) diagonal

    def solve(V_h, V_n, V_c, T_0, rng):
        u = jnp.array([V_h, V_n, V_c])
        y0 = init.at[amp_idx].set(T_0)
        bm = diffrax.VirtualBrownianTree(
            t0=0.0, t1=D, tol=dt / 2.0, shape=(dim_state,), key=rng
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift_fn),
            diffrax.ControlTerm(diffusion_fn, bm),
        )
        solver = diffrax.Euler()
        n_pts = max(int(round(D / dt)) + 1, 32)
        saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, D, n_pts))
        sol = diffrax.diffeqsolve(
            terms, solver, t0=0.0, t1=D, dt0=dt,
            y0=y0, args=u, saveat=saveat,
            max_steps=int(D / dt) + 100,
        )
        return sol.ys                              # (n_pts, dim_state)

    return jax.jit(solve)


# =========================================================================
# THE GATING-TEST FIXTURE
# =========================================================================

# Tiny module-level cache so the same (model, D) pair JITs only once
# across many calls (e.g. inside a parametrised pytest test).
_ODE_CACHE: dict = {}


def _get_ode_solver(model: ModelInterface, D: float):
    key = (id(model), D)
    if key not in _ODE_CACHE:
        _ODE_CACHE[key] = _make_ode_solve(model, D)
    return _ODE_CACHE[key]


def t_end_under_constant_controls(
    model: ModelInterface,
    V_h: float, V_n: float, V_c: float,
    T_0: float = 0.5,
    D: float = 14.0,
    backend: str = "diffrax_ode",
) -> float:
    """Last-day mean of T under constant controls.

    Default backend is `diffrax_ode` (deterministic, JIT'd). For
    stochastic tests use `sleep_fraction_under_controls` directly.
    """
    if backend != "diffrax_ode":
        raise ValueError(
            f"Unknown backend {backend!r}. The default 'diffrax_ode' is "
            "what every gating test should use; stochastic tests should "
            "use sleep_fraction_under_controls directly."
        )
    solver = _get_ode_solver(model, D)
    return float(solver(V_h, V_n, V_c, T_0))


# =========================================================================
# CONVENIENCE: noise-off variant
# =========================================================================

def with_noise_off(model: ModelInterface) -> ModelInterface:
    """Return a copy of `model` with all diffusion temperatures forced to 0."""
    new_params = dict(model.params)
    for key in list(new_params.keys()):
        if key.startswith("T_") and isinstance(new_params[key], (int, float)):
            new_params[key] = 0.0
    return ModelInterface(
        drift=model.drift,
        diffusion=model.diffusion,
        params=new_params,
        init_state=model.init_state.copy(),
        amplitude_index=model.amplitude_index,
        state_clip=model.state_clip,
        name=f"{model.name}__noise_off",
    )


# =========================================================================
# VMAP'D GRID EVAL — for sweeps and heatmaps
# =========================================================================

def vmap_grid_eval(
    model: ModelInterface, V_h_vals: np.ndarray, V_n_vals: np.ndarray,
    V_c: float = 0.0, T_0: float = 0.5, D: float = 14.0,
) -> np.ndarray:
    """Evaluate T_end on a (V_h_vals × V_n_vals) grid via vmap.

    Returns a 2-D array of shape (len(V_h_vals), len(V_n_vals)).
    """
    solver = _get_ode_solver(model, D)
    # Build mesh-grid of V_h, V_n.
    V_h_mesh, V_n_mesh = jnp.meshgrid(jnp.asarray(V_h_vals),
                                        jnp.asarray(V_n_vals),
                                        indexing="ij")
    flat_V_h = V_h_mesh.flatten()
    flat_V_n = V_n_mesh.flatten()
    flat_V_c = jnp.full_like(flat_V_h, V_c)
    flat_T_0 = jnp.full_like(flat_V_h, T_0)

    grid_solver = jax.vmap(solver)
    flat_T = grid_solver(flat_V_h, flat_V_n, flat_V_c, flat_T_0)
    return np.asarray(flat_T).reshape(V_h_mesh.shape)


# =========================================================================
# CONVENIENCE: sleep fraction (uses SDE backend)
# =========================================================================

def sleep_fraction_under_controls(
    model: ModelInterface, V_h: float, V_n: float, V_c: float,
    z_index: int = 1, c_tilde_key: str = "c_tilde",
    T_0: float = 0.5, D: float = 14.0,
    *, dt: float = 0.005, n_particles: int = 64, rng_seed: int = 42,
) -> float:
    """Fraction of trajectory time-points with Z >= c_tilde."""
    sde = _make_sde_solve(model, D, dt=dt)
    rng = jax.random.PRNGKey(rng_seed)
    rngs = jax.random.split(rng, n_particles)
    # vmap over particles
    batched = jax.vmap(sde, in_axes=(None, None, None, None, 0))
    traj = batched(V_h, V_n, V_c, T_0, rngs)        # (n_particles, n_pts, dim)
    c_tilde = float(model.params[c_tilde_key])
    return float((np.asarray(traj[..., z_index]) >= c_tilde).mean())

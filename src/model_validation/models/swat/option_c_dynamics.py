"""Option C — V_h modulates entrainment quality, NOT the drift.

This is a structural revision of the original Option C draft (which had
V_h scale λ inside the drift directly). The original draft killed the
W↔Z flip-flop because dropping λ from 32 → 4 in the drift weakened
circadian forcing below the noise floor.

The fix: split λ into two roles.

  - λ (in the drift): stays at the spec value (32). Preserves the
    physical circadian forcing strength on W → clean flip-flop, sleep
    architecture intact, sleep fraction at ~33% with c_tilde=2.5.

  - λ_amp_W, λ_amp_Z (NEW, in entrainment_quality only): replace the
    forcing-amplitude term in the amp_W / amp_Z formulas. These are
    *analytical* knobs that govern how strongly V_h modulates E_dyn —
    they have no effect on the actual W or Z trajectories.

V_h is also removed from u_W's drift (matching the original draft).
This decouples vitality from sleep-wake architecture, giving V_h a
single, monotone, interpretable effect: stronger V_h → larger amp_W
and amp_Z analytical terms → larger E_dyn → larger μ(E) → larger T*.

Drift is otherwise unchanged from the spec. Two new parameters total:
λ_amp_W and λ_amp_Z.
"""
from __future__ import annotations
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .vendored_parameters import default_swat_parameters
from . import vendored_dynamics


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def _circadian(t_days, V_c_hours, phi_0):
    return jnp.sin(2.0 * jnp.pi * (t_days - V_c_hours / 24.0) + phi_0)


def entrainment_quality_option_c(
    W: jnp.ndarray, Z: jnp.ndarray, a: jnp.ndarray, T: jnp.ndarray,
    V_h: jnp.ndarray, V_n: jnp.ndarray, V_c: jnp.ndarray,
    params: Dict[str, float]
) -> jnp.ndarray:
    """E_dyn under refined Option C.

    amp_W = sigmoid(B_W + A_W) - sigmoid(B_W - A_W)  with
        A_W = λ_amp_W · (1 + V_h)
        B_W = V_n - a + α_T · T

    Same form for amp_Z. λ_amp_W and λ_amp_Z are entrainment-formula
    parameters and NOT used in the drift.
    """
    alpha_T = params['alpha_T']
    beta_Z = params['beta_Z']
    lam_amp_W = params['lambda_amp_W']
    lam_amp_Z = params['lambda_amp_Z']

    A_W = lam_amp_W * (1.0 + V_h)
    A_Z = lam_amp_Z * (1.0 + V_h)
    B_W = V_n - a + alpha_T * T
    B_Z = -V_n + beta_Z * a

    amp_W = _sigmoid(B_W + A_W) - _sigmoid(B_W - A_W)
    amp_Z = _sigmoid(B_Z + A_Z) - _sigmoid(B_Z - A_Z)
    phase = jnp.maximum(jnp.cos(2.0 * jnp.pi * V_c / 24.0), 0.0)
    return amp_W * amp_Z * phase


def swat_drift_option_c(t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray,
                          params: Dict[str, float]) -> jnp.ndarray:
    """Drift under refined Option C.

    KEY: identical to the spec drift EXCEPT V_h is removed from u_W.
    The spec λ stays at 32 (in 'lmbda' params key) — circadian forcing
    on W is unchanged, so the daily flip-flop is preserved.

    V_h is *not used at all* in the drift. Its effect on T flows
    entirely through entrainment_quality_option_c.
    """
    W, Z, a, T = x[0], x[1], x[2], x[3]
    V_h, V_n, V_c = u[0], u[1], u[2]

    lam = params['lmbda']           # SPEC VALUE — keeps W flip-flop
    kappa = params['kappa']
    gamma_3 = params['gamma_3']
    beta_Z = params['beta_Z']
    A_scale = params['A_scale']
    phi_0 = params['phi_0']
    tau_W = params['tau_W']
    tau_Z = params['tau_Z']
    tau_a = params['tau_a']
    tau_T = params['tau_T']
    mu_0 = params['mu_0']
    mu_E = params['mu_E']
    eta = params['eta']
    alpha_T = params['alpha_T']

    # Spec drift, with V_h REMOVED from u_W.
    C_eff = _circadian(t, V_c, phi_0)
    u_W = lam * C_eff + V_n - a - kappa * Z + alpha_T * T   # V_h removed
    u_Z = -gamma_3 * W - V_n + beta_Z * a                   # unchanged

    dW = (_sigmoid(u_W) - W) / tau_W
    dZ = (A_scale * _sigmoid(u_Z) - Z) / tau_Z
    da = (W - a) / tau_a

    E_dyn = entrainment_quality_option_c(W, Z, a, T, V_h, V_n, V_c, params)
    mu = mu_0 + mu_E * E_dyn
    dT = (mu * T - eta * T ** 3) / tau_T

    return jnp.stack([dW, dZ, da, dT])


def option_c_parameters(
    lambda_amp_W: Optional[float] = None,
    lambda_amp_Z: Optional[float] = None,
    c_tilde: Optional[float] = None,
) -> Dict[str, float]:
    """Build a parameter dict for refined Option C.

    Calibrated defaults:
      lmbda           = 32.0  (UNCHANGED from spec — preserves W↔Z flip-flop)
      lambda_amp_W    = 4.0   (NEW — entrainment-formula forcing scale on W;
                                gives V_h sensitivity in [0, 4])
      lambda_amp_Z    = 1.0   (NEW — entrainment-formula forcing scale on Z;
                                ratio matches spec's γ_3:λ ≈ 1:4)
      c_tilde         = 2.5   (UNCHANGED from OT-Control vendored — works at
                                spec λ=32)
    """
    p = default_swat_parameters()
    # Spec lambda is kept (default_swat_parameters has lmbda=32 already).
    p['lambda_amp_W'] = 4.0 if lambda_amp_W is None else float(lambda_amp_W)
    p['lambda_amp_Z'] = 1.0 if lambda_amp_Z is None else float(lambda_amp_Z)
    if c_tilde is not None:
        p['c_tilde'] = float(c_tilde)
    return p


def option_c_model(
    lambda_amp_W: Optional[float] = None,
    lambda_amp_Z: Optional[float] = None,
    c_tilde: Optional[float] = None,
    *,
    # Backward-compat: older code may pass lambda_base / lambda_Z_base.
    # Map them to the new lambda_amp_* slot, IGNORING the request to
    # change the drift-side λ (since the structural fix requires keeping
    # spec λ in the drift).
    lambda_base: Optional[float] = None,
    lambda_Z_base: Optional[float] = None,
):
    """Return a ModelInterface configured for refined Option C."""
    from model_validation.runner import ModelInterface

    if lambda_amp_W is None and lambda_base is not None:
        lambda_amp_W = lambda_base
    if lambda_amp_Z is None and lambda_Z_base is not None:
        lambda_amp_Z = lambda_Z_base

    _INIT_STATE_A = np.array([0.5, 3.5, 0.5, 0.5])
    p = option_c_parameters(lambda_amp_W, lambda_amp_Z, c_tilde)
    return ModelInterface(
        drift=swat_drift_option_c,
        diffusion=vendored_dynamics.swat_diffusion,
        params=p,
        init_state=_INIT_STATE_A.copy(),
        amplitude_index=3,
        state_clip=vendored_dynamics.swat_state_clip,
        name=f"option_c_swat(λ_amp_W={p['lambda_amp_W']}, "
             f"λ_amp_Z={p['lambda_amp_Z']}, c_tilde={p['c_tilde']})",
    )

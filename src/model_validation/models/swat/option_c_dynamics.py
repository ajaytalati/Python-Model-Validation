"""Option C v4 / "Option D" — refined entrainment formulation.

Two structural fixes layered on top of the spec:

(1) V_h modulates the analytical entrainment quality, NOT the physical
    drift. λ in the drift stays at the spec value (32) so the W↔Z
    daily flip-flop is preserved. V_h enters via NEW parameters
    λ_amp_W and λ_amp_Z which scale the amplitude-formula forcing
    terms inside `entrainment_quality_option_c`. With A = λ_amp · V_h
    (no +1 offset), V_h=0 gives A=0 → amp=0 → no entrainment
    (clinically right: depleted vitality = no rhythm).

(2) V_n acts as a multiplicative DAMPENER on E_dyn (rather than as a
    "balance-point tuner" via the bell-shaped amp formulas). One new
    parameter V_n_scale. damp(V_n) = exp(−V_n / V_n_scale) is
    monotonically decreasing in V_n, so any chronic load > 0
    monotonically attenuates the rhythm — clinically correct.

Drift is otherwise unchanged from the spec. Three new parameters
total: λ_amp_W, λ_amp_Z, V_n_scale.

Issue refs:
  - #4 — V_h structural inversion (fixed by (1))
  - #5 — V_n non-monotonic catabolicity (fixed by (2))
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
    """E_dyn under Option C v4 / Option D.

    E_dyn = damp(V_n) · amp_W · amp_Z · phase(V_c)

    where
        amp_W = σ(B_W + A_W) − σ(B_W − A_W)
        amp_Z = σ(B_Z + A_Z) − σ(B_Z − A_Z)
        A_W   = λ_amp_W · V_h        (V_h is anabolic via forcing scale)
        A_Z   = λ_amp_Z · V_h
        B_W   = V_n − a + α_T · T
        B_Z   = −V_n + β_Z · a
        damp  = exp(−V_n / V_n_scale)         # NEW (issue #5)
        phase = max(cos(2π V_c / 24), 0)
    """
    alpha_T = params['alpha_T']
    beta_Z = params['beta_Z']
    lam_amp_W = params['lambda_amp_W']
    lam_amp_Z = params['lambda_amp_Z']
    V_n_scale = params['V_n_scale']

    A_W = lam_amp_W * V_h
    A_Z = lam_amp_Z * V_h
    B_W = V_n - a + alpha_T * T
    B_Z = -V_n + beta_Z * a

    amp_W = _sigmoid(B_W + A_W) - _sigmoid(B_W - A_W)
    amp_Z = _sigmoid(B_Z + A_Z) - _sigmoid(B_Z - A_Z)

    # Multiplicative V_n dampener — addresses issue #5. Any V_n > 0
    # monotonically attenuates E. damp(0)=1, damp(V_n_scale·ln 2)=0.5,
    # damp(∞)=0.
    damp = jnp.exp(-V_n / V_n_scale)

    phase = jnp.maximum(jnp.cos(2.0 * jnp.pi * V_c / 24.0), 0.0)
    return damp * amp_W * amp_Z * phase


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
    V_n_scale: Optional[float] = None,
    c_tilde: Optional[float] = None,
) -> Dict[str, float]:
    """Build a parameter dict for Option C v4 / Option D.

    Calibrated defaults:
      lmbda           = 32.0  (UNCHANGED from spec — preserves W↔Z flip-flop)
      lambda_amp_W    = 5.0   (NEW — entrainment-formula forcing scale on W.
                                With A_W = λ_amp_W · V_h, gives amp_W ≈ 1
                                across daily B_W cycle at healthy V_h=1.)
      lambda_amp_Z    = 8.0   (NEW — Z-side scale. Larger because β_Z·a can
                                reach ~4, so A_Z must dominate. Gives
                                amp_Z ≈ 1 across daily a-cycle at V_h=1.)
      V_n_scale       = 2.0   (NEW — V_n dampener time-scale (issue #5).
                                damp = exp(−V_n / V_n_scale).
                                damp(0) = 1.00
                                damp(0.3) = 0.86
                                damp(1.0) = 0.61
                                damp(2.0) = 0.37
                                damp(3.5) = 0.17  (sub-critical)
                                damp(5.0) = 0.08)
      c_tilde         = 3.0   (Matches upstream PARAM_SET_A. At V_n=0 healthy
                                default, gives sleep fraction ~35%, in
                                target window. The OT-Control vendored bump
                                to 2.5 was for the V_n=0.3 healthy regime.)
    """
    p = default_swat_parameters()
    # Spec lambda is kept (default_swat_parameters has lmbda=32 already).
    p['lambda_amp_W'] = 5.0 if lambda_amp_W is None else float(lambda_amp_W)
    p['lambda_amp_Z'] = 8.0 if lambda_amp_Z is None else float(lambda_amp_Z)
    p['V_n_scale']    = 2.0 if V_n_scale is None else float(V_n_scale)
    p['c_tilde']      = 3.0 if c_tilde is None else float(c_tilde)
    return p


def option_c_model(
    lambda_amp_W: Optional[float] = None,
    lambda_amp_Z: Optional[float] = None,
    V_n_scale: Optional[float] = None,
    c_tilde: Optional[float] = None,
    *,
    # Backward-compat aliases for older callers (test fixtures, scripts).
    lambda_base: Optional[float] = None,
    lambda_Z_base: Optional[float] = None,
):
    """Return a ModelInterface configured for Option C v4 / Option D."""
    from model_validation.runner import ModelInterface

    if lambda_amp_W is None and lambda_base is not None:
        lambda_amp_W = lambda_base
    if lambda_amp_Z is None and lambda_Z_base is not None:
        lambda_amp_Z = lambda_Z_base

    _INIT_STATE_A = np.array([0.5, 3.5, 0.5, 0.5])
    p = option_c_parameters(lambda_amp_W, lambda_amp_Z, V_n_scale, c_tilde)
    return ModelInterface(
        drift=swat_drift_option_c,
        diffusion=vendored_dynamics.swat_diffusion,
        params=p,
        init_state=_INIT_STATE_A.copy(),
        amplitude_index=3,
        state_clip=vendored_dynamics.swat_state_clip,
        name=f"option_c_v4_swat(λ_amp_W={p['lambda_amp_W']}, "
             f"λ_amp_Z={p['lambda_amp_Z']}, V_n_scale={p['V_n_scale']}, "
             f"c_tilde={p['c_tilde']})",
    )

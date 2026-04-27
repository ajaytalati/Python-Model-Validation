"""Option C — refined SWAT drift with redefined amp_W and amp_Z.

This is the candidate fix for the V_h structural inversion (issue #4).
Key differences from the vendored spec:

1. **V_h removed from u_W's slow drive.** V_h scales the circadian forcing
   amplitude on W instead.

2. **gamma_3 replaced by lambda_Z.** New parameter for the forcing amplitude
   on Z. Symmetric with lambda's role on W.

3. **amp_W and amp_Z redefined** to use the actual oscillation amplitude:

       amp_W = sigmoid(B_W + A_W) - sigmoid(B_W - A_W)
       amp_Z = sigmoid(B_Z + A_Z) - sigmoid(B_Z - A_Z)

   where A_W = lambda * (1 + V_h) and A_Z = lambda_Z * (1 + V_h) are the
   forcing amplitudes (V_h-controlled), and B_W = V_n - a + alpha_T*T,
   B_Z = -V_n + beta_Z*a are the slow drive offsets.

The drift formulas for the state SDEs (W, Z, a, T) are unchanged in form;
only the entrainment_quality formula changes.

Configurable parameters:
  - lambda (existing, default 32 from spec)
  - lambda_Z (new, default proposed: same as gamma_3 = 8 to match
    the original Z forcing strength)
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
    """Refined E_dyn using actual oscillation amplitude."""
    alpha_T = params['alpha_T']
    beta_Z = params['beta_Z']
    lam = params['lmbda']
    lam_Z = params['lambda_Z']

    A_W = lam * (1.0 + V_h)
    A_Z = lam_Z * (1.0 + V_h)
    B_W = V_n - a + alpha_T * T
    B_Z = -V_n + beta_Z * a

    amp_W = _sigmoid(B_W + A_W) - _sigmoid(B_W - A_W)
    amp_Z = _sigmoid(B_Z + A_Z) - _sigmoid(B_Z - A_Z)
    phase = jnp.maximum(jnp.cos(2.0 * jnp.pi * V_c / 24.0), 0.0)
    return amp_W * amp_Z * phase


def swat_drift_option_c(t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray,
                          params: Dict[str, float]) -> jnp.ndarray:
    """Drift function under Option C.

    The W and Z SDEs use the new forcing amplitudes (V_h-modulated).
    u_W's slow drive no longer contains V_h.
    u_Z replaces -gamma_3 * W with -lambda_Z * (1 + V_h) * something —
    but to keep the W -> Z coupling in the drift consistent with the
    spec's structural intent, we instead keep the spec's u_Z formula
    and let lambda_Z affect entrainment_quality only via amp_Z's
    forcing-amplitude term. (This is the cleanest interpretation: the
    PHYSICAL drift formulas are unchanged; only the analytical
    amplitude-quality factor uses lambda_Z.)
    """
    W, Z, a, T = x[0], x[1], x[2], x[3]
    V_h, V_n, V_c = u[0], u[1], u[2]

    lam = params['lmbda']
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

    # Refined u_W: V_h modulates forcing strength, V_h removed from slow drive.
    C_eff = _circadian(t, V_c, phi_0)
    u_W = lam * (1.0 + V_h) * C_eff + V_n - a - kappa * Z + alpha_T * T
    u_Z = -gamma_3 * W - V_n + beta_Z * a

    dW = (_sigmoid(u_W) - W) / tau_W
    dZ = (A_scale * _sigmoid(u_Z) - Z) / tau_Z
    da = (W - a) / tau_a

    E_dyn = entrainment_quality_option_c(W, Z, a, T, V_h, V_n, V_c, params)
    mu = mu_0 + mu_E * E_dyn
    dT = (mu * T - eta * T ** 3) / tau_T

    return jnp.stack([dW, dZ, da, dT])


def option_c_parameters(
    lambda_base: Optional[float] = None,
    lambda_Z_base: Optional[float] = None,
    c_tilde: Optional[float] = None,
) -> Dict[str, float]:
    """Build a parameter dict for Option C.

    Calibrated defaults (per validation library's sleep-fraction tuning):
      lambda    = 4.0   (down from spec's 32; required for V_h sensitivity
                          under the new amp_W = sigmoid(B+A) - sigmoid(B-A) form)
      lambda_Z  = 1.0   (Z forcing baseline; symmetric with lambda)
      c_tilde   = 3.0   (matches upstream PARAM_SET_A; gives ~35% sleep
                          fraction at lambda=4. The vendored bump to 2.5
                          was for the spec's lambda=32 regime.)

    Each parameter is overridable; pass None to keep the calibrated default.
    """
    p = default_swat_parameters()
    p['lmbda'] = 4.0 if lambda_base is None else float(lambda_base)
    p['lambda_Z'] = 1.0 if lambda_Z_base is None else float(lambda_Z_base)
    p['c_tilde'] = 3.0 if c_tilde is None else float(c_tilde)
    return p


def option_c_model(
    lambda_base: Optional[float] = None,
    lambda_Z_base: Optional[float] = None,
):
    """Return a ModelInterface configured for Option C."""
    from model_validation.runner import ModelInterface

    _INIT_STATE_A = np.array([0.5, 3.5, 0.5, 0.5])
    p = option_c_parameters(lambda_base, lambda_Z_base)
    return ModelInterface(
        drift=swat_drift_option_c,
        diffusion=vendored_dynamics.swat_diffusion,
        params=p,
        init_state=_INIT_STATE_A.copy(),
        amplitude_index=3,
        state_clip=vendored_dynamics.swat_state_clip,
        name=f"option_c_swat(lambda={p['lmbda']}, lambda_Z={p['lambda_Z']})",
    )

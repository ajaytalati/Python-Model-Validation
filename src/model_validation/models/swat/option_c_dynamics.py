"""Backward-compat shim — Option C / Option D is now just SWAT.

The "Option C v4 / Option D" refinement was the validation work that
identified the V_h-anabolic structural fix.  After upstream PR #11 in
Python-Model-Development-Simulation landed, that fix is the canonical
SWAT model — there is no separate "variant" any more.

This file remains as a backward-compatibility shim so existing test
fixtures, scripts, and CI workflows that import the Option-C names
continue to work without immediate rewriting.  All names re-export
the corresponding entries from `vendored_dynamics` /
`vendored_parameters`.

Plan: delete this shim once all callers have been moved to the
canonical names (`swat_drift`, `entrainment_quality`,
`default_swat_parameters`, `vendored_model`).
"""
from __future__ import annotations
from typing import Optional

from . import vendored_dynamics
from .vendored_dynamics import (
    swat_drift as swat_drift_option_c,
    entrainment_quality as entrainment_quality_option_c,
)
from .vendored_parameters import default_swat_parameters


def option_c_parameters(
    lambda_amp_W: Optional[float] = None,
    lambda_amp_Z: Optional[float] = None,
    V_n_scale: Optional[float] = None,
    V_c_max: Optional[float] = None,
    c_tilde: Optional[float] = None,
):
    """Return a SWAT parameter dict, optionally overriding the four
    structural-fix scalars.

    The defaults from `default_swat_parameters` already include the
    calibrated values (lambda_amp_W=5, lambda_amp_Z=8, V_n_scale=2,
    V_c_max=3); overrides here are for tests and scripts that want
    to perturb individual values.
    """
    p = default_swat_parameters()
    if lambda_amp_W is not None:
        p['lambda_amp_W'] = float(lambda_amp_W)
    if lambda_amp_Z is not None:
        p['lambda_amp_Z'] = float(lambda_amp_Z)
    if V_n_scale is not None:
        p['V_n_scale'] = float(V_n_scale)
    if V_c_max is not None:
        p['V_c_max'] = float(V_c_max)
    if c_tilde is not None:
        p['c_tilde'] = float(c_tilde)
    return p


def option_c_model(
    lambda_amp_W: Optional[float] = None,
    lambda_amp_Z: Optional[float] = None,
    V_n_scale: Optional[float] = None,
    V_c_max: Optional[float] = None,
    c_tilde: Optional[float] = None,
    *,
    # Backward-compat aliases for older callers.
    lambda_base: Optional[float] = None,
    lambda_Z_base: Optional[float] = None,
):
    """Return a `ModelInterface` configured for SWAT (formerly Option C)."""
    import numpy as np
    from model_validation.runner import ModelInterface

    if lambda_amp_W is None and lambda_base is not None:
        lambda_amp_W = lambda_base
    if lambda_amp_Z is None and lambda_Z_base is not None:
        lambda_amp_Z = lambda_Z_base

    _INIT_STATE_A = np.array([0.5, 3.5, 0.5, 0.5])
    p = option_c_parameters(lambda_amp_W, lambda_amp_Z, V_n_scale,
                              V_c_max, c_tilde)
    return ModelInterface(
        drift=vendored_dynamics.swat_drift,
        diffusion=vendored_dynamics.swat_diffusion,
        params=p,
        init_state=_INIT_STATE_A.copy(),
        amplitude_index=3,
        state_clip=vendored_dynamics.swat_state_clip,
        name=f"swat(λ_amp_W={p['lambda_amp_W']}, "
             f"λ_amp_Z={p['lambda_amp_Z']}, V_n_scale={p['V_n_scale']}, "
             f"c_tilde={p['c_tilde']})",
    )

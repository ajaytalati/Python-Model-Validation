"""Sim-est consistency: drift() (numpy/scipy) must equal drift_jax() (JAX).

Pattern from postmortem Bug 1 (sign-flip via `mu_0_abs` reparameterisation):
https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md

The dev SWAT model exposes two drift implementations:
- `drift(t, y, params, aux)` — numpy, used by the scipy SDE solver path.
- `drift_jax(t, y, args)` — JAX, used by the Diffrax path.

Both MUST produce identical values at any (t, y, params), or the
JAX-path simulator will silently disagree with the numpy-path
identifiability/SDE tools that consume PARAM_SET_A. This test pins a
fixed state at the truth params and asserts bit-equality (modulo
tiny float rounding).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)


def _ensure_dev_model_on_path():
    if not os.path.isdir(PUBLIC_DEV_V1):
        pytest.skip(f"Dev model repo not found at {PUBLIC_DEV_V1}")
    if PUBLIC_DEV_V1 not in sys.path:
        sys.path.insert(0, PUBLIC_DEV_V1)


def test_drift_numpy_matches_drift_jax():
    _ensure_dev_model_on_path()
    import jax.numpy as jnp
    from models.swat.simulation import (
        PARAM_SET_A, INIT_STATE_A, drift, drift_jax,
    )

    p = dict(PARAM_SET_A)

    # Build a 7-state vector (W, Zt, a, T, C, Vh, Vn) at the prior mean.
    y_np = np.array([
        INIT_STATE_A['W_0'], INIT_STATE_A['Zt_0'], INIT_STATE_A['a_0'],
        INIT_STATE_A['T_0'],
        # C(t=0) per the analytical pacemaker:
        np.sin(0.0 + (-np.pi / 3.0)),
        INIT_STATE_A['Vh'], INIT_STATE_A['Vn'],
    ], dtype=np.float64)
    y_jax = jnp.asarray(y_np)

    t = 6.0          # mid-day so circadian forcing is non-trivial

    dy_np = drift(t, y_np, p, aux=None)
    dy_jax = np.asarray(drift_jax(t, y_jax, args=(p,)))

    print(f"\n  drift     = {np.array2string(dy_np, precision=6)}")
    print(f"  drift_jax = {np.array2string(dy_jax, precision=6)}")

    np.testing.assert_allclose(
        dy_np, dy_jax, atol=1e-9, rtol=1e-7,
        err_msg=("drift (numpy) and drift_jax (JAX) disagree at "
                  "PARAM_SET_A truth. Check parameter sign / "
                  "reparameterisation parity per the FSA postmortem "
                  "Bug 1 template."),
    )


if __name__ == "__main__":
    test_drift_numpy_matches_drift_jax()
    print("PASS: drift / drift_jax parity holds at PARAM_SET_A truth")

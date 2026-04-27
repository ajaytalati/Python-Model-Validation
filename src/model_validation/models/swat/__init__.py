"""SWAT model — vendored snapshot + ModelInterface for the gating tests."""
from __future__ import annotations
import numpy as np

from model_validation.runner import ModelInterface

from . import vendored_dynamics
from .vendored_parameters import default_swat_parameters


# Canonical "scenario A" healthy initial state from the upstream spec.
# (W_0=0.5, Zt_0=3.5, a_0=0.5, T_0=0.5)
_INIT_STATE_A = np.array([0.5, 3.5, 0.5, 0.5])


def vendored_model() -> ModelInterface:
    """Return the vendored SWAT model (current snapshot)."""
    return ModelInterface(
        drift=vendored_dynamics.swat_drift,
        diffusion=vendored_dynamics.swat_diffusion,
        params=default_swat_parameters(),
        init_state=_INIT_STATE_A.copy(),
        amplitude_index=3,                         # T is state[3]
        state_clip=vendored_dynamics.swat_state_clip,
        name="vendored_swat",
    )

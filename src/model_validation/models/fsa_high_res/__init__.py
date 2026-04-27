"""FSA-high-res model — vendored snapshot + ModelInterface for validation."""
from __future__ import annotations
import numpy as np

from model_validation.runner import ModelInterface

from . import vendored_dynamics
from .vendored_parameters import default_fsa_parameters


# Canonical "moderately healthy" initial state used by the FSA adapter
# in OT-Control to derive the empirical target pool. (B_0=0.3, F_0=0.05,
# A_0=0.4) — sits in the super-critical region of mu(B,F).
_INIT_STATE_HEALTHY = np.array([0.3, 0.05, 0.4])


def vendored_model() -> ModelInterface:
    """Return the vendored FSA-high-res model (current snapshot).

    Note: the validation runner's `t_end_under_constant_controls` is
    SWAT-shaped (V_h, V_n, V_c) controls. FSA's controls are (T_B, Phi)
    so callers building FSA pipelines should construct their own
    diffrax solver — see identifiability/fsa_high_res/compute_fim.py
    and stability/fsa_high_res/corner_case_sweep.py for templates.
    The ModelInterface returned here is for callers that want the
    drift / diffusion / params / init_state in one bundle.
    """
    return ModelInterface(
        drift=vendored_dynamics.fsa_drift,
        diffusion=vendored_dynamics.fsa_diffusion,
        params=default_fsa_parameters(),
        init_state=_INIT_STATE_HEALTHY.copy(),
        amplitude_index=2,                         # A is state[2] in (B, F, A)
        state_clip=vendored_dynamics.fsa_state_clip,
        name="vendored_fsa_high_res",
    )

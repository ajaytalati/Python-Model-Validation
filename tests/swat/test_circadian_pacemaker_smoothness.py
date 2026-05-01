"""Sim-est consistency: circadian pacemaker C(t) must be smooth.

The dev SWAT model declares C as a 5th deterministic state with
analytical dynamics dC/dt = (2π/24)·cos(2πt/24 + φ_0). This test
forward-simulates the model and asserts |dC/dt| stays bounded by the
analytic maximum (2π/24 + numerical slack), which catches any future
regression that accidentally folds V_c into the pacemaker (instead of
keeping V_c as a SUBJECT-SHIFTED drive in u_W).

Pattern from the FSA postmortem (Bug 3, C-phase mis-alignment):
https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md
"""
from __future__ import annotations

import math
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


def test_pacemaker_is_smooth_under_set_A():
    _ensure_dev_model_on_path()

    from psim.scenarios.presets.swat_set_A_healthy import truth_params_and_init
    from psim.pipelines import synthesise_scenario
    from models.swat.simulation import SWAT_MODEL

    truth_params, init_state = truth_params_and_init()

    n_days = 3
    dt_hours = 5.0 / 60.0
    bins_per_day = int(round(24.0 / dt_hours))
    n_bins_total = n_days * bins_per_day

    sim_run = synthesise_scenario(
        SWAT_MODEL,
        truth_params=truth_params, init_state=init_state,
        exogenous_arrays={},
        n_bins_total=n_bins_total, dt_days=dt_hours,
        bins_per_day=bins_per_day, n_substeps=4, seed=42,
        obs_channel_names=(),
    )

    traj = np.asarray(sim_run.trajectory)
    # State order per SWAT_MODEL: W, Zt, a, T, C, Vh, Vn → C is column 4.
    C = traj[:, 4]

    dC_per_h = np.abs(np.diff(C)) / dt_hours
    analytic_max = 2.0 * math.pi / 24.0           # ≈ 0.2618
    tolerance = 1e-3                              # numerical slack

    max_observed = float(dC_per_h.max())
    print(f"\n  max |dC/dt| = {max_observed:.5f} /h  "
          f"(analytic cap {analytic_max:.5f}, tol +{tolerance})")

    assert max_observed <= analytic_max + tolerance, (
        f"Pacemaker derivative {max_observed:.5f}/h exceeds analytic "
        f"max {analytic_max:.5f}/h. Something is folding V_c (or other "
        f"control inputs) into the pacemaker C state — V_c must only "
        f"shift the SUBJECT'S effective drive C_eff in u_W, not the "
        f"pacemaker itself."
    )


if __name__ == "__main__":
    test_pacemaker_is_smooth_under_set_A()
    print("PASS: pacemaker C(t) is smooth")

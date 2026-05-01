"""Sim-est consistency: W, Zt, a must stay in [0, 1].

Phase 3.6 rescaled Zt and a from their old domains ([0, A_SCALE=6] for
Zt, [0, ∞) for a) into [0, 1] with Jacobi diffusion `g(x) = sqrt(x(1-x))`
that vanishes at the boundaries. This test runs each canonical scenario
(Set A/B/C/D) and asserts the bound holds. Catches regressions where
someone re-introduces the A_SCALE multiplier in the Zt drift, or
swaps Jacobi back to constant-noise that pushes the SDE outside the
domain.

Pattern from
https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md
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


# Allow a small slack for Euler-Maruyama overshoot at the boundaries
# (the IMEX clipper catches it but a single substep may briefly cross).
SLACK = 0.05


SCENARIOS = [
    ("set_A_healthy",        "swat_set_A_healthy"),
    ("set_B_amplitude",      "swat_set_B_amplitude"),
    ("set_C_recovery",       "swat_set_C_recovery"),
    ("set_D_phase_shift",    "swat_set_D_phase_shift"),
]


@pytest.mark.parametrize("nice_name, preset_module", SCENARIOS)
def test_latents_in_unit_interval(nice_name, preset_module):
    _ensure_dev_model_on_path()
    import importlib
    from psim.pipelines import synthesise_scenario
    from models.swat.simulation import SWAT_MODEL

    preset = importlib.import_module(f"psim.scenarios.presets.{preset_module}")
    truth_params, init_state = preset.truth_params_and_init()

    n_days = 14
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
    W, Zt, a = traj[:, 0], traj[:, 1], traj[:, 2]

    print(f"\n  {nice_name}:  W=[{W.min():.3f},{W.max():.3f}]  "
          f"Zt=[{Zt.min():.3f},{Zt.max():.3f}]  "
          f"a=[{a.min():.3f},{a.max():.3f}]")

    assert -SLACK <= W.min() and W.max() <= 1.0 + SLACK, (
        f"W outside [0, 1] in {nice_name}: range [{W.min():.4f}, {W.max():.4f}]")
    assert -SLACK <= Zt.min() and Zt.max() <= 1.0 + SLACK, (
        f"Zt outside [0, 1] in {nice_name}: range [{Zt.min():.4f}, {Zt.max():.4f}]. "
        f"If max is ~5-6, the A_SCALE multiplier in dZt drift may have been "
        f"re-introduced.")
    assert -SLACK <= a.min() and a.max() <= 1.0 + SLACK, (
        f"a outside [0, 1] in {nice_name}: range [{a.min():.4f}, {a.max():.4f}]")


if __name__ == "__main__":
    for nice_name, preset_module in SCENARIOS:
        test_latents_in_unit_interval(nice_name, preset_module)
    print("\nPASS: W, Zt, a stay in [0, 1] across all 4 SWAT scenarios")

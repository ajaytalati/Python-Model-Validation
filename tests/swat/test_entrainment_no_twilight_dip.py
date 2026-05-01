"""Regression guard: E_dyn must NOT oscillate with the daily circadian.

The V_h-anabolic formula (per swat_entrainment_docs/) depends only on
slow states (a, T) and controls (V_h, V_n, V_c). It MUST NOT read
instantaneous W, Zt, or C(t). The signature would be: a daily
oscillation in E_dyn with twilight dips at the C zero-crossings.

This test runs psim Set A 14d, computes E_dyn(t) at every bin, and
asserts the relative variation between adjacent 1-hour windows stays
below a tight threshold (modulo slow drift in a/T over hours).

If a future regression reintroduces an instantaneous-state dependency
into the formula, this test fails immediately with a clear diagnostic.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)


def _ensure_path():
    if not os.path.isdir(PUBLIC_DEV_V1):
        pytest.skip(f"Dev model repo not found at {PUBLIC_DEV_V1}")
    if PUBLIC_DEV_V1 not in sys.path:
        sys.path.insert(0, PUBLIC_DEV_V1)


# Bound on relative variation between adjacent 1h windows. The slow (a, T)
# drift over hours is allowed; daily oscillation from C is NOT allowed.
# 8% is generous (the V_h-anabolic formula gives ~1-2% noise from a/T).
MAX_HOURLY_RELATIVE_VARIATION = 0.08


def test_no_twilight_dip_set_A():
    _ensure_path()
    from psim.scenarios.presets.swat_set_A_healthy import truth_params_and_init
    from psim.pipelines import synthesise_scenario
    from models.swat.simulation import SWAT_MODEL, entrainment_quality

    truth_params, init_state = truth_params_and_init()

    n_days = 14
    dt_hours = 5.0 / 60.0
    bins_per_day = int(round(24.0 / dt_hours))   # 288
    bins_per_hour = int(round(1.0 / dt_hours))   # 12
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

    # Compute E_dyn at every bin via the dev model's helper
    E_per_bin = np.array([
        entrainment_quality(0.0, 0.0, traj[k, 2], traj[k, 3],
                             traj[k, 5], traj[k, 6], truth_params)
        for k in range(n_bins_total)
    ])

    # Average per 1-hour window
    n_hours = n_bins_total // bins_per_hour
    E_per_hour = E_per_bin[:n_hours * bins_per_hour].reshape(
        n_hours, bins_per_hour).mean(axis=1)

    # Relative variation between consecutive hourly windows
    rel_var = np.abs(np.diff(E_per_hour)) / np.maximum(
        np.abs(E_per_hour[:-1]), 1e-3)
    max_rel = float(rel_var.max())

    print(f"\n  E_dyn per-hour: mean={E_per_hour.mean():.4f}, "
          f"std={E_per_hour.std():.4f}, max hourly relvar={max_rel:.2%}  "
          f"(cap: {MAX_HOURLY_RELATIVE_VARIATION:.0%})")

    assert max_rel < MAX_HOURLY_RELATIVE_VARIATION, (
        f"E_dyn varies {max_rel:.2%} between consecutive 1h windows "
        f"in Set A (cap {MAX_HOURLY_RELATIVE_VARIATION:.0%}). This "
        f"indicates the entrainment formula has acquired an "
        f"instantaneous-W or instantaneous-C dependency — see "
        f"swat_entrainment_docs/01_formula.md for the correct shape."
    )


if __name__ == "__main__":
    test_no_twilight_dip_set_A()
    print("PASS: E_dyn is structurally non-oscillating in Set A")

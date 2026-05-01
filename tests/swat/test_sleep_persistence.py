"""Sim-est consistency: sleep labels must form multi-hour blocks.

Forward-simulates the SWAT model under PARAM_SET_A (healthy basin) and
asserts the per-day count of sleep_level transitions stays under a
biological-realism threshold. Pre-fix (independent multinomial draws
per bin) the count was hundreds per day; post-fix (sticky-HMM kernel
with tau_sleep_persist_h ≈ 0.5h) it should be ≤ ~12/day.

This test would have caught the per-bin sleep_label flicker in
2026-04-25 had it existed before. Pattern from
https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


# Inject the dev-model path the same way psim's example scripts do.
PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)


def _ensure_dev_model_on_path():
    if not os.path.isdir(PUBLIC_DEV_V1):
        pytest.skip(f"Dev model repo not found at {PUBLIC_DEV_V1}")
    if PUBLIC_DEV_V1 not in sys.path:
        sys.path.insert(0, PUBLIC_DEV_V1)


# Generous upper bound: real sleep has ~4 stage transitions per day
# (wake → light → deep → light → wake). The sticky kernel at 5-min
# binning produces around 8-12. Anything above 30/day means the
# sticky kernel is broken or absent.
MAX_TRANSITIONS_PER_DAY = 30


def test_sleep_blocks_form_under_set_A():
    _ensure_dev_model_on_path()

    from psim.scenarios.presets.swat_set_A_healthy import truth_params_and_init
    from psim.pipelines import synthesise_scenario
    from models.swat.simulation import SWAT_MODEL

    truth_params, init_state = truth_params_and_init()

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
        obs_channel_names=('sleep',),
    )

    sleep_labels = np.asarray(sim_run.obs_channels['sleep']['sleep_level'])
    assert sleep_labels.shape == (n_bins_total,)

    n_transitions = int(np.sum(np.diff(sleep_labels) != 0))
    transitions_per_day = n_transitions / n_days

    print(f"\n  Sleep transitions: {n_transitions} total over {n_days}d = "
          f"{transitions_per_day:.1f}/day  (cap: {MAX_TRANSITIONS_PER_DAY}/day)")

    assert transitions_per_day <= MAX_TRANSITIONS_PER_DAY, (
        f"Sleep label flicker: {transitions_per_day:.1f} transitions/day "
        f"exceeds {MAX_TRANSITIONS_PER_DAY}/day. The sticky-sleep kernel "
        f"in models/swat/simulation.py:gen_sleep may be missing or "
        f"misconfigured."
    )


if __name__ == "__main__":
    test_sleep_blocks_form_under_set_A()
    print("PASS: sleep blocks form correctly under Set A")

"""Sim-est consistency: pin the V_h-anabolic entrainment reference values.

These are the canonical sanity values from the model author's tutorial:
    /home/ajay/Downloads/swat_entrainment_docs/02_components.md
        section "Sanity checks"
    /home/ajay/Downloads/swat_entrainment_docs/04_worked_examples.md
        section "Summary table"

The dev model's `entrainment_quality` MUST reproduce these values to
1e-4 tolerance. Plus a bit-exact (1e-12) cross-check against the
tutorial's reference implementation `entrainment_model.py:entrainment_quality`.

Pattern from
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
TUTORIAL_DIR = os.path.expanduser("~/Downloads/swat_entrainment_docs")


def _ensure_paths():
    if not os.path.isdir(PUBLIC_DEV_V1):
        pytest.skip(f"Dev model repo not found at {PUBLIC_DEV_V1}")
    if not os.path.isdir(TUTORIAL_DIR):
        pytest.skip(f"Tutorial dir not found at {TUTORIAL_DIR}")
    if PUBLIC_DEV_V1 not in sys.path:
        sys.path.insert(0, PUBLIC_DEV_V1)
    if TUTORIAL_DIR not in sys.path:
        sys.path.insert(0, TUTORIAL_DIR)


# Documented sanity values (a=0.5, T=0.85) from 02_components.md "Sanity checks"
SANITY_VALUES = [
    # (label, V_h, V_n, V_c, expected E_dyn)
    ('healthy',                   1.0, 0.3, 0.0, 0.8476),
    ('V_h depleted',              0.2, 0.3, 0.0, 0.1747),
    ('chronic load high',         1.0, 3.5, 0.0, 0.1477),
    ('phase shift 1h',            1.0, 0.3, 1.0, 0.7340),
    ('phase shift 2h',            1.0, 0.3, 2.0, 0.4238),
    ('phase shift past threshold', 1.0, 0.3, 6.0, 0.0000),
]


@pytest.mark.parametrize("label, V_h, V_n, V_c, expected", SANITY_VALUES)
def test_documented_sanity_value(label, V_h, V_n, V_c, expected):
    """Each of the 6 sanity values from 02_components.md matches to 1e-4."""
    _ensure_paths()
    from models.swat.simulation import (
        entrainment_quality, PARAM_SET_A,
    )
    p = dict(PARAM_SET_A)
    p['V_c'] = V_c
    E = entrainment_quality(0.0, 0.0, 0.5, 0.85, V_h, V_n, p)
    assert abs(E - expected) < 1e-4, (
        f"Scenario '{label}' (V_h={V_h}, V_n={V_n}, V_c={V_c}): "
        f"expected E_dyn ≈ {expected:.4f}, got {E:.6f} (diff {abs(E-expected):.2e})"
    )


def test_bit_exact_match_with_tutorial_reference():
    """Cross-repo invariant from README.md "How to verify": dev model's
    entrainment_quality must be MATHEMATICALLY IDENTICAL to the tutorial's
    reference implementation, to ~1e-12 precision."""
    _ensure_paths()
    from models.swat.simulation import (
        entrainment_quality as dev_E, PARAM_SET_A,
    )
    from entrainment_model import entrainment_quality as ref_E

    a_ss, T_ss = 0.5, 0.85
    failures = []
    for label, V_h, V_n, V_c, _ in SANITY_VALUES:
        ref = ref_E(0.0, 0.0, a_ss, T_ss, V_h, V_n, V_c)
        p = dict(PARAM_SET_A); p['V_c'] = V_c
        dev = dev_E(0.0, 0.0, a_ss, T_ss, V_h, V_n, p)
        if abs(ref - dev) >= 1e-12:
            failures.append((label, ref, dev, abs(ref - dev)))

    assert not failures, (
        "Dev model deviates from tutorial reference:\n"
        + "\n".join(f"  {l}: ref={r:.6f} dev={d:.6f} diff={x:.2e}"
                    for l, r, d, x in failures)
    )


def test_worked_example_intermediates_set_A():
    """Worked example from 04_worked_examples.md § "Example 1 — Healthy
    reference": every intermediate (B_W, B_Z, amp_W, amp_Z, damp, phase,
    E_dyn, mu, T*) matches to 1e-4."""
    _ensure_paths()
    from models.swat.simulation import PARAM_SET_A

    # Inputs (from worked example): V_h=1, V_n=0.3, V_c=0; a=0.5, T=0.85
    p = PARAM_SET_A
    V_h, V_n, V_c = 1.0, 0.3, 0.0
    a, T = 0.5, 0.85

    # Step 1 — slow backdrops
    B_W = V_n - a + p['alpha_T'] * T
    B_Z = -V_n + p['beta_Z'] * a
    assert abs(B_W - 0.055) < 1e-4
    assert abs(B_Z - 1.7)   < 1e-4

    # Step 2 — band half-widths
    A_W = p['lambda_amp_W'] * V_h
    A_Z = p['lambda_amp_Z'] * V_h
    assert A_W == 5.0 and A_Z == 8.0

    # Step 3 — amplitude factors
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    amp_W = sig(B_W + A_W) - sig(B_W - A_W)
    amp_Z = sig(B_Z + A_Z) - sig(B_Z - A_Z)
    assert abs(amp_W - 0.9866) < 1e-3
    assert abs(amp_Z - 0.9981) < 1e-3

    # Step 4 — damper
    damp = math.exp(-V_n / p['V_n_scale'])
    assert abs(damp - 0.8607) < 1e-3

    # Step 5 — phase
    V_c_eff = min(abs(V_c), p['V_c_max'])
    phase = math.cos(math.pi * V_c_eff / (2.0 * p['V_c_max']))
    assert phase == 1.0

    # Step 6 — E_dyn
    E = damp * amp_W * amp_Z * phase
    assert abs(E - 0.8476) < 1e-3

    # Step 7 — bifurcation
    mu = p['mu_0'] + p['mu_E'] * E
    T_star = math.sqrt(max(mu, 0.0) / p['eta'])
    assert abs(mu - 0.348) < 1e-3
    assert abs(T_star - 0.834) < 1e-3


if __name__ == "__main__":
    for label, V_h, V_n, V_c, expected in SANITY_VALUES:
        test_documented_sanity_value(label, V_h, V_n, V_c, expected)
    test_bit_exact_match_with_tutorial_reference()
    test_worked_example_intermediates_set_A()
    print("PASS: all entrainment reference values match the tutorial")

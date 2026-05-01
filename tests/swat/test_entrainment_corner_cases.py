"""Corner-case assertions for the V_h-anabolic entrainment formula.

Each test corresponds to a case from
    /home/ajay/Downloads/swat_entrainment_docs/03_corner_cases.md
"""
from __future__ import annotations

import math
import os
import sys

import pytest


PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)


def _ensure_path():
    if not os.path.isdir(PUBLIC_DEV_V1):
        pytest.skip(f"Dev model repo not found at {PUBLIC_DEV_V1}")
    if PUBLIC_DEV_V1 not in sys.path:
        sys.path.insert(0, PUBLIC_DEV_V1)


def _E(V_h, V_n, V_c):
    from models.swat.simulation import entrainment_quality, PARAM_SET_A
    p = dict(PARAM_SET_A); p['V_c'] = V_c
    return entrainment_quality(0.0, 0.0, 0.5, 0.85, V_h, V_n, p)


def test_corner_3_1_V_h_zero_hard_zero():
    """§3.1 V_h = 0 ⇒ E_dyn = 0 EXACTLY (hard zero through amp_*)."""
    _ensure_path()
    assert _E(V_h=0.0, V_n=0.3, V_c=0.0) == 0.0


def test_corner_3_2_high_V_n_damper_dominates():
    """§3.2 V_n = 5 (the bound) ⇒ damp ≈ 0.082, E essentially zero."""
    _ensure_path()
    E = _E(V_h=1.0, V_n=5.0, V_c=0.0)
    # damp alone = 0.0821; combined with the off-centre amp_* this
    # gives a small E. Tutorial's table for V_n=5 isn't pinned but
    # it's certainly < damp = 0.0821 (since amp_W < 1).
    assert E < 0.082, f"E at V_n=5 should be ≤ damp(5) = 0.082, got {E:.4f}"


def test_corner_3_3_phase_clamp_hard_zero():
    """§3.3 |V_c| > V_c_max ⇒ phase = 0 (machine-epsilon hard zero;
    cos(π/2) is fp ≈ 6e-17 not exactly 0)."""
    _ensure_path()
    for V_c in (3.5, 4.0, 6.0, 12.0):
        E = _E(V_h=1.0, V_n=0.3, V_c=V_c)
        assert abs(E) < 1e-15, (
            f"E at V_c={V_c}h should be 0 (phase clamp), got {E:.3e}"
        )


def test_corner_3_4_phase_symmetry():
    """§3.4 phase(V_c) = phase(-V_c)."""
    _ensure_path()
    for V_c in (1.0, 2.0, 3.0, 6.0):
        E_pos = _E(V_h=1.0, V_n=0.3, V_c=+V_c)
        E_neg = _E(V_h=1.0, V_n=0.3, V_c=-V_c)
        assert abs(E_pos - E_neg) < 1e-12, (
            f"E({V_c}) = {E_pos:.6f} != E({-V_c}) = {E_neg:.6f}"
        )


def test_corner_3_5_boundary_V_h_zero_V_n_zero():
    """§3.5 V_h = V_n = 0 ⇒ E = 0 (zero through amp_*)."""
    _ensure_path()
    assert _E(V_h=0.0, V_n=0.0, V_c=0.0) == 0.0


if __name__ == "__main__":
    test_corner_3_1_V_h_zero_hard_zero()
    test_corner_3_2_high_V_n_damper_dominates()
    test_corner_3_3_phase_clamp_hard_zero()
    test_corner_3_4_phase_symmetry()
    test_corner_3_5_boundary_V_h_zero_V_n_zero()
    print("PASS: all 5 corner cases hold")

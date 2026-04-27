"""Test 4 — no backwards optimum on the clinical box.

Smoke detector: argmax of T_end over a 5x5 (V_h, V_n) grid must NOT be at
V_h=0 (the lower bound). If it is, V_h is structurally inverted.
"""
import pytest
import numpy as np
from model_validation.runner import t_end_under_constant_controls


V_H_VALS = [0.0, 0.5, 1.0, 2.0, 4.0]
V_N_VALS = [0.0, 0.3, 1.0, 2.0, 4.0]


def test_no_backwards_optimum(model):
    grid = np.zeros((len(V_H_VALS), len(V_N_VALS)))
    for i, V_h in enumerate(V_H_VALS):
        for j, V_n in enumerate(V_N_VALS):
            grid[i, j] = t_end_under_constant_controls(
                model, V_h=V_h, V_n=V_n, V_c=0.0
            )
    print(f"\n  T_end grid (rows=V_h, cols=V_n):")
    print(f"        V_n=" + "  ".join(f"{v:5.2f}" for v in V_N_VALS))
    for i, V_h in enumerate(V_H_VALS):
        print(f"  V_h={V_h:4.2f}  " + "  ".join(f"{grid[i, j]:5.3f}" for j in range(len(V_N_VALS))))

    argmax_i, argmax_j = np.unravel_index(grid.argmax(), grid.shape)
    print(f"  argmax at V_h={V_H_VALS[argmax_i]}, V_n={V_N_VALS[argmax_j]}, "
          f"T_end={grid[argmax_i, argmax_j]:.3f}")

    assert argmax_i > 0, (
        f"Backwards optimum: argmax T_end is at V_h={V_H_VALS[argmax_i]} "
        f"(the lower bound). V_h is structurally anti-anabolic."
    )

"""Step 3a — λ-sensitivity sweep for Option C.

Saturation diagnostic: at each (λ, λ_Z) baseline, run the V_h sweep and
check whether T_end actually moves with V_h. The smallest baseline at
which T_end has a Δ > 0.1 spread across V_h ∈ [0, 4] is the regime where
Option C produces the desired clinical behaviour.

Output:
  runs/option_c_lambda_sweep/sweep.csv        — full results
  runs/option_c_lambda_sweep/sweep.png        — T_end vs V_h, one curve per baseline
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_validation.runner import t_end_under_constant_controls, with_noise_off
from model_validation.models.swat.option_c_dynamics import option_c_model

OUT = Path(__file__).resolve().parents[1] / "runs" / "option_c_lambda_sweep"
OUT.mkdir(parents=True, exist_ok=True)

LAMBDA_PAIRS = [
    (32.0, 8.0),     # spec default
    (8.0, 2.0),
    (4.0, 1.0),
    (2.0, 0.5),
    (1.0, 0.25),
    (0.5, 0.125),
]
V_H_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def main():
    rows = []
    fig, ax = plt.subplots(figsize=(9, 6))
    for lam, lam_Z in LAMBDA_PAIRS:
        m = with_noise_off(option_c_model(lam, lam_Z))
        Ts = []
        for V_h in V_H_GRID:
            T = t_end_under_constant_controls(m, V_h, V_n=0.3, V_c=0.0,
                                                T_0=0.5, D=14.0)
            Ts.append(T)
            rows.append({
                "lambda": lam, "lambda_Z": lam_Z, "V_h": V_h, "T_end": T,
            })
        spread = max(Ts) - min(Ts)
        argmax_v_h = V_H_GRID[int(np.argmax(Ts))]
        print(f"λ={lam:5.2f}, λ_Z={lam_Z:5.2f}  V_h sweep T_end: "
              f"{[f'{t:.3f}' for t in Ts]}  "
              f"Δ={spread:.3f}  argmax_V_h={argmax_v_h:.2f}")
        ax.plot(V_H_GRID, Ts, "o-", label=f"λ={lam}, λ_Z={lam_Z} (Δ={spread:.2f})")

    ax.set_xlabel("V_h")
    ax.set_ylabel("T_end (deterministic, last-day mean, D=14d, V_n=0.3, V_c=0)")
    ax.set_title("Option C: λ-sensitivity sweep")
    ax.axhline(0.55, color="green", ls=":", alpha=0.6, label="spec target T*=0.55")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sweep.png", dpi=130)

    with (OUT / "sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote: {OUT / 'sweep.png'}")
    print(f"Wrote: {OUT / 'sweep.csv'}")


if __name__ == "__main__":
    main()

"""V_h × V_n heatmap under Option C — proves the structural fix.

Compares:
  (a) Pre-fix vendored model — argmax at V_h=0 (the inversion).
  (b) Refined Option C at λ=4, λ_Z=1 — argmax at V_h=4 (the corner).

Output: runs/option_c_heatmap/heatmap_compare.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_validation.runner import vmap_grid_eval, with_noise_off
from model_validation.models.swat import vendored_model
from model_validation.models.swat.option_c_dynamics import option_c_model

OUT = Path(__file__).resolve().parents[1] / "runs" / "option_c_heatmap"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    V_h_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    V_n_vals = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0])

    # (a) Pre-fix
    print("Pre-fix vendored:")
    m_pre = with_noise_off(vendored_model())
    grid_pre = vmap_grid_eval(m_pre, V_h_vals, V_n_vals)
    i, j = np.unravel_index(grid_pre.argmax(), grid_pre.shape)
    print(f"  argmax at V_h={V_h_vals[i]}, V_n={V_n_vals[j]}, T_end={grid_pre[i, j]:.3f}")

    # (b) Option C at λ=4, λ_Z=1
    print("Option C (λ=4, λ_Z=1):")
    m_oc = with_noise_off(option_c_model(lambda_base=4.0, lambda_Z_base=1.0))
    grid_oc = vmap_grid_eval(m_oc, V_h_vals, V_n_vals)
    i, j = np.unravel_index(grid_oc.argmax(), grid_oc.shape)
    print(f"  argmax at V_h={V_h_vals[i]}, V_n={V_n_vals[j]}, T_end={grid_oc[i, j]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, grid, title in [
        (axes[0], grid_pre, "Pre-fix vendored — argmax at V_h=0 (inverted)"),
        (axes[1], grid_oc,  "Option C, λ=4, λ_Z=1 — argmax at V_h=4 (clinical)"),
    ]:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                          vmin=0, vmax=1.0)
        ax.set_xticks(range(len(V_n_vals)))
        ax.set_xticklabels([f"{v:.1f}" for v in V_n_vals], rotation=45)
        ax.set_yticks(range(len(V_h_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in V_h_vals])
        ax.set_xlabel("V_n")
        ax.set_ylabel("V_h")
        ax.set_title(title)
        for ii in range(grid.shape[0]):
            for jj in range(grid.shape[1]):
                ax.text(jj, ii, f"{grid[ii, jj]:.2f}",
                          ha="center", va="center",
                          color="white" if grid[ii, jj] < 0.5 else "black",
                          fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("V_h × V_n grid — pre-fix vs Option C (deterministic, D=14d, T_0=0.5)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_compare.png", dpi=130)
    plt.close(fig)
    print(f"\nWrote: {OUT / 'heatmap_compare.png'}")


if __name__ == "__main__":
    main()

"""Generate clinician-facing diagnostic plots for the four canonical scenarios.

For each of the scenarios A/B/C/D, produces under both:
  - pre-fix vendored snapshot (current upstream)
  - Option C at calibrated defaults (lambda=4, lambda_Z=1, c_tilde=3)

three plots:
  - latents.png          (W, Z̃, a, T, C_ext)
  - observations.png     (HR, sleep stages, steps, stress)
  - entrainment.png      (E_dyn, μ(E_dyn), T(t) vs T*)

Output tree: docs/clinician_views/<variant>/<scenario>/

D=14 days by default. n_realisations=8 stochastic trajectories per panel
(8 traces shown, mean overlaid).
"""
from __future__ import annotations
from pathlib import Path

from model_validation.models.swat import vendored_model
from model_validation.models.swat.option_c_dynamics import option_c_model
from model_validation.clinician_plots import plot_all_for_scenario

OUT = Path(__file__).resolve().parents[1] / "docs" / "clinician_views"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    # name, V_h, V_n, V_c, T_0, description
    # Healthy default uses V_n=0 — clean V_h-only signal per user's spec.
    ("A_healthy",   1.0, 0.0, 0.0, 0.50, "Healthy: V_h=1, V_n=0 (clean signal)"),
    ("B_insomnia",  0.2, 3.5, 0.0, 0.50, "Insomnia: low V_h, severe V_n"),
    ("C_recovery",  1.0, 0.0, 0.0, 0.05, "Recovery: healthy controls, sick T_0"),
    ("D_shift_work", 1.0, 0.0, 6.0, 0.50, "Shift-work: 6h phase shift, V_n=0"),
]
D = 14.0
N_REAL = 8


def main():
    for variant_name, model_factory in [
        ("pre_fix_vendored", vendored_model),
        ("option_c_calibrated", option_c_model),
    ]:
        print(f"\n=== {variant_name} ===")
        m = model_factory()
        for label, V_h, V_n, V_c, T_0, descr in SCENARIOS:
            save_dir = OUT / variant_name / label
            print(f"  {label}: {descr}")
            plot_all_for_scenario(
                m, label, V_h, V_n, V_c, T_0,
                save_dir=save_dir, D=D,
                variant=("option-c" if "option_c" in variant_name else "vendored"),
                n_realisations=N_REAL,
            )
            print(f"      -> {save_dir.relative_to(OUT.parent.parent)}/")


if __name__ == "__main__":
    main()

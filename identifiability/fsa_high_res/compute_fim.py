"""Fisher Information Matrix analysis of the FSA-high-res model.

Method
------
Sensitivity-based FIM proxy via JAX autograd through diffrax. For an
SDE the rigorous Fisher information uses particle-filter likelihood;
the standard practical proxy is the Jacobian of the deterministic
predicted observation trajectory:

    J[k, j] = ∂y_pred[k] / ∂θ_j

The FIM is then F = J^T · Σ_obs⁻¹ · J. Rank, condition number, and
per-parameter scores are read off this matrix.

Scope
-----
We analyse the 10 DETERMINISTIC drift parameters of the FSA-high-res
SDE under the deployed defaults:

  Fitness block:        tau_B, alpha_A
  Strain block:         tau_F, lambda_B, lambda_A
  Amplitude block:      mu_0, mu_B, mu_F, mu_FF, eta

NOT analysed here:
  Diffusion temperatures (3): sigma_B, sigma_F, sigma_A — these enter
  only the noise level, so deterministic-trajectory sensitivity gives
  zero. They need a separate residual-variance analysis (deferred).

Observation model
-----------------
FSA's vendoring deliberately omits the observation model (the upstream
HR / sleep / stress / steps channels live with the upstream filtering
pipeline, not in OT-Control's vendored copy). So the "observation" for
this FIM proxy is the latent state itself — the cleanest case, where
all three components (B, F, A) are taken as directly observed with
fixed Gaussian noise levels. This corresponds to a best-case
identifiability bound; if the latent state is observed only via noisy
proxies the achievable identifiability can only get worse.

Multi-operating-point design
----------------------------
The 10 drift parameters are not all visible at a single operating
point. tau_F, lambda_B, lambda_A are invisible when Phi=0 (F stays
near zero). mu_FF (the F^2 term) is invisible at low F. alpha_A is
invisible when A is near zero. To excite all parameters we compute J
at four (T_B, Phi) operating points and stack:

  (0.0, 0.0)   unfit_recovery — slow regime, mu_0 + tau_B visible
  (0.5, 0.05)  healthy reference — full mu(B,F) coupling visible
  (0.7, 0.5)   moderate overtraining — mu_FF visible (high F)
  (1.0, 2.0)   extreme overtraining — large dynamic range

Output
------
- results/sensitivity_heatmap.png      |J[k, j]| matrix per (op, channel)
- results/eigenvalue_plot.png          eigenvalue spectrum (log scale)
- results/per_parameter_identifiability.png  per-param info bar chart
- results/fim_summary.json             rank, condition number, table
- results/per_parameter_table.csv      human-readable parameter breakdown
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# x64 is non-negotiable for FIM on a stiff ODE
jax.config.update("jax_enable_x64", True)

import diffrax

from model_validation.models.fsa_high_res.vendored_parameters import (
    default_fsa_parameters,
)
from model_validation.models.fsa_high_res.vendored_dynamics import (
    EPS_A_FROZEN, EPS_B_FROZEN, EPS_F_FROZEN,
)


OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


# =========================================================================
# Parameter packing — fixed order, deterministic-drift parameters only
# =========================================================================

PARAM_NAMES = [
    "tau_B", "alpha_A",                   # fitness block
    "tau_F", "lambda_B", "lambda_A",      # strain block
    "mu_0", "mu_B", "mu_F", "mu_FF",      # bifurcation parameter mu(B, F)
    "eta",                                # Landau cubic
]
N_PARAMS = len(PARAM_NAMES)


# Observation noise levels (fixed). The latent state is treated as
# observed with these Gaussian-noise stds — same role as Σ_obs in
# the SWAT analysis. These are reasonable scale-by-dynamic-range
# choices for the (B, F, A) state in the deployed regime.
SIGMA_B = 0.05      # B in [0, 1], 5% of range
SIGMA_F = 0.05      # F typically O(0.1)–O(1), 5% of typical
SIGMA_A = 0.05      # A typically O(0.5)–O(1), 5% of typical


# =========================================================================
# Differentiable predicted state-trajectory
# =========================================================================

def _bifurcation_parameter(B, F, p):
    return p["mu_0"] + p["mu_B"] * B - p["mu_F"] * F - p["mu_FF"] * F ** 2


def _fsa_drift(t, x, T_B, Phi, p):
    """Drift f(t, x, u, params). FSA-3-state, controls in u = (T_B, Phi)."""
    del t                                       # autonomous
    B, F, A = x[0], x[1], x[2]
    inv_tau_B_eff = (1.0 + p["alpha_A"] * A) / p["tau_B"]
    dB = inv_tau_B_eff * (T_B - B)
    inv_tau_F_eff = (1.0 + p["lambda_B"] * B + p["lambda_A"] * A) / p["tau_F"]
    dF = Phi - inv_tau_F_eff * F
    mu = _bifurcation_parameter(B, F, p)
    dA = mu * A - p["eta"] * (A ** 3)
    return jnp.stack([dB, dF, dA])


def _vec_to_params(theta_vec):
    """Build params dict from theta vector. Diffusion temperatures stay
    at default (irrelevant for deterministic prediction)."""
    p = {name: theta_vec[i] for i, name in enumerate(PARAM_NAMES)}
    # Pin diffusion temperatures at defaults — they don't affect the
    # deterministic mean, but we keep them in the dict in case any
    # downstream helper reads them.
    defaults = default_fsa_parameters()
    for k in ("sigma_B", "sigma_F", "sigma_A"):
        p[k] = defaults[k]
    return p


def predict_state_trajectory(theta_vec, T_B, Phi, init_state, t_grid):
    """Deterministic state trajectory under (T_B, Phi); flat 1-D output.

    Returns a length (n_pts × 3) array: B(t_grid), F(t_grid), A(t_grid)
    concatenated in that order.
    """
    p = _vec_to_params(theta_vec)

    def vf(t, y, args):
        return _fsa_drift(t, y, T_B, Phi, p)

    D = float(t_grid[-1])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Tsit5(),
        t0=0.0, t1=D, dt0=0.01,
        y0=init_state,
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
        saveat=diffrax.SaveAt(ts=t_grid),
        max_steps=200_000,
    )
    traj = sol.ys                                   # (n_pts, 3)
    B = traj[:, 0]; F = traj[:, 1]; A = traj[:, 2]
    return jnp.concatenate([B, F, A])


# =========================================================================
# Reference parameter point + operating-point design
# =========================================================================

def reference_theta() -> jnp.ndarray:
    p = default_fsa_parameters()
    return jnp.array([float(p[name]) for name in PARAM_NAMES])


# Initial state at the FSA adapter's "moderately healthy" reference:
# (B_0=0.3, F_0=0.05, A_0=0.4) — same point used to derive the model-
# derived target pool in OT-Control's adapter.
INIT_STATE = jnp.array([0.3, 0.05, 0.4])


def operating_points():
    """Four (T_B, Phi) op-points spanning the parameter-visibility space."""
    return [
        (0.0, 0.0),    # unfit_recovery — mu_0, tau_B, eta visible
        (0.5, 0.05),   # healthy reference — full mu(B,F) coupling
        (0.7, 0.5),    # moderate overtraining — mu_FF visible (high F)
        (1.0, 2.0),    # extreme overtraining — large dynamic range
    ]


# =========================================================================
# Stacked Jacobian
# =========================================================================

def compute_stacked_jacobian(theta_0, ops, init_state,
                              D_days=14.0, n_per_day=24):
    n_pts = int(D_days * n_per_day) + 1
    t_grid = jnp.linspace(0.0, D_days, n_pts)

    rows = []
    op_labels = []
    for T_B, Phi in ops:
        def predict_at_op(theta, _T_B=T_B, _Phi=Phi):
            return predict_state_trajectory(theta, _T_B, _Phi,
                                              init_state, t_grid)
        J = jax.jacrev(predict_at_op)(theta_0)        # (3 * n_pts, N_PARAMS)
        rows.append(np.asarray(J))
        op_labels.append(f"T_B={T_B}, Phi={Phi}")
        print(f"  Computed J at {op_labels[-1]}: shape {J.shape}, "
              f"|J|_F = {float(jnp.linalg.norm(J)):.3e}", flush=True)
    J_stack = np.vstack(rows)
    return J_stack, op_labels, n_pts


# =========================================================================
# Observation noise weighting
# =========================================================================

def build_sigma_inv_diagonal(n_pts):
    """Per-observation 1/σ² weights, in the order [B(t_grid), F, A]."""
    inv_B = np.full(n_pts, 1.0 / SIGMA_B ** 2)
    inv_F = np.full(n_pts, 1.0 / SIGMA_F ** 2)
    inv_A = np.full(n_pts, 1.0 / SIGMA_A ** 2)
    return np.concatenate([inv_B, inv_F, inv_A])


# =========================================================================
# Main FIM analysis
# =========================================================================

def main():
    print("=" * 70, flush=True)
    print("FIM analysis — FSA-high-res model (deployed v1.2.1)", flush=True)
    print("=" * 70, flush=True)
    print(f"Parameters analysed: {N_PARAMS}", flush=True)
    print(f"Parameter names: {PARAM_NAMES}", flush=True)
    print(flush=True)

    theta_0 = reference_theta()
    ops = operating_points()
    print(f"Reference theta_0 = {dict(zip(PARAM_NAMES, [float(x) for x in theta_0]))}", flush=True)
    print(f"Initial state: B_0={INIT_STATE[0]}, F_0={INIT_STATE[1]}, A_0={INIT_STATE[2]}", flush=True)
    print(f"Operating points: {ops}", flush=True)
    print(flush=True)

    print("Computing stacked Jacobian (autograd through diffrax) ...", flush=True)
    J_stack, op_labels, n_pts = compute_stacked_jacobian(
        theta_0, ops, INIT_STATE, D_days=14.0, n_per_day=24
    )
    print(f"Total stacked J shape: {J_stack.shape}", flush=True)
    print(flush=True)

    sigma_inv_one_op = build_sigma_inv_diagonal(n_pts)
    sigma_inv_full = np.tile(sigma_inv_one_op, len(ops))
    assert sigma_inv_full.shape[0] == J_stack.shape[0]

    # Form the FIM
    print("Forming FIM = J^T · Σ_obs⁻¹ · J ...", flush=True)
    Wd = sigma_inv_full[:, None]
    F = J_stack.T @ (Wd * J_stack)
    print(f"FIM shape: {F.shape}", flush=True)
    print(flush=True)

    # SVD of weighted J
    print("SVD of weighted J ...", flush=True)
    sqrt_w = np.sqrt(sigma_inv_full[:, None])
    J_w = sqrt_w * J_stack
    U, sigma, Vh = np.linalg.svd(J_w, full_matrices=False)
    print(f"Singular values (largest 3): {sigma[:3]}", flush=True)
    print(f"Singular values (smallest 3): {sigma[-3:]}", flush=True)
    print(flush=True)

    # FIM eigendecomposition
    print("FIM eigendecomposition ...", flush=True)
    eigvals, eigvecs = np.linalg.eigh(F)
    print(f"FIM eigenvalues (largest 3): {eigvals[-1:-4:-1]}", flush=True)
    print(f"FIM eigenvalues (smallest 3): {eigvals[:3]}", flush=True)
    cond = float(eigvals[-1] / max(eigvals[0], 1e-30))
    print(f"FIM condition number: {cond:.3e}", flush=True)
    rank_eps = max(eigvals) * 1e-10
    rank = int(np.sum(eigvals > rank_eps))
    print(f"Numerical rank (eigval > {rank_eps:.2e}): {rank} / {N_PARAMS}", flush=True)
    print(flush=True)

    # Per-parameter scoring
    print("Per-parameter identifiability analysis ...", flush=True)
    F_diag = np.diag(F).copy()
    try:
        F_inv = np.linalg.inv(F + 1e-12 * np.eye(N_PARAMS) * eigvals.max())
    except np.linalg.LinAlgError:
        F_inv = np.linalg.pinv(F)
    crb_diag = np.diag(F_inv)
    identifiable_individual = 1.0 / np.sqrt(np.maximum(crb_diag, 1e-30))

    def correlation_to_others(j):
        denoms = np.sqrt(F_diag[j] * F_diag) + 1e-30
        cors = np.abs(F[j, :]) / denoms
        cors[j] = 0.0
        k = int(np.argmax(cors))
        return k, float(cors[k])

    table = []
    for j, name in enumerate(PARAM_NAMES):
        partner_idx, partner_corr = correlation_to_others(j)
        table.append({
            "param": name,
            "value": float(theta_0[j]),
            "F_jj": float(F_diag[j]),
            "self_info_sqrt_F_jj": float(np.sqrt(F_diag[j])),
            "marginal_std_lower_bound": float(np.sqrt(max(crb_diag[j], 0.0))),
            "individual_id_score": float(identifiable_individual[j]),
            "most_collinear_partner": PARAM_NAMES[partner_idx],
            "partner_correlation": partner_corr,
        })

    # Plots
    print("\nGenerating plots ...", flush=True)

    # Sensitivity heatmap
    mean_abs_J = np.abs(J_stack).reshape(
        len(ops), 3, n_pts, N_PARAMS
    ).mean(axis=2)
    heatmap = mean_abs_J.reshape(-1, N_PARAMS)
    row_labels = [f"{op_labels[o]} | {ch}"
                   for o in range(len(ops))
                   for ch in ["B", "F", "A"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    log_h = np.log10(heatmap + 1e-12)
    im = ax.imshow(log_h, aspect="auto", cmap="viridis")
    ax.set_xticks(range(N_PARAMS))
    ax.set_xticklabels(PARAM_NAMES, rotation=60, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label=r"$\log_{10}\,|\partial y_{\rm mean}/\partial\theta|$")
    ax.set_title("Sensitivity heatmap — mean |J| per (op-point, channel) row")
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_heatmap.png", dpi=130)
    plt.close(fig)

    # Eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(range(1, N_PARAMS + 1), eigvals[::-1], "o-", color="darkblue")
    ax.set_xlabel("Eigenvalue index (largest first)")
    ax.set_ylabel("FIM eigenvalue (log scale)")
    ax.set_title(f"FIM eigenvalue spectrum — N={N_PARAMS} params, "
                  f"rank={rank}, κ={cond:.2e}")
    ax.grid(alpha=0.3, which="both")
    ax.axhline(eigvals.max() * 1e-10, color="red", linestyle="--",
                alpha=0.5, label="rank threshold (1e-10 of max)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "eigenvalue_plot.png", dpi=130)
    plt.close(fig)

    # Per-param info bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    scores = [t["individual_id_score"] for t in table]
    colors = []
    for name in PARAM_NAMES:
        if name in ("mu_0", "mu_B", "mu_F", "mu_FF", "eta"):
            colors.append("crimson")             # bifurcation block
        elif name in ("tau_B", "alpha_A"):
            colors.append("forestgreen")         # fitness block
        else:
            colors.append("steelblue")           # strain block
    ax.bar(range(N_PARAMS), scores, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(N_PARAMS))
    ax.set_xticklabels(PARAM_NAMES, rotation=60, fontsize=9)
    ax.set_ylabel("Individual identifiability score (1/CRB std, log)")
    ax.set_title("Per-parameter identifiability — FSA-high-res "
                  "(red = bifurcation block, green = fitness, blue = strain)")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(OUT / "per_parameter_identifiability.png", dpi=130)
    plt.close(fig)

    # Save summaries
    summary = {
        "n_params": N_PARAMS,
        "rank": rank,
        "condition_number_FIM": cond,
        "eigvals_largest_3": eigvals[-1:-4:-1].tolist(),
        "eigvals_smallest_3": eigvals[:3].tolist(),
        "operating_points": op_labels,
        "param_table": table,
    }
    (OUT / "fim_summary.json").write_text(json.dumps(summary, indent=2))

    with (OUT / "per_parameter_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    print(f"\nWrote: {OUT / 'sensitivity_heatmap.png'}", flush=True)
    print(f"Wrote: {OUT / 'eigenvalue_plot.png'}", flush=True)
    print(f"Wrote: {OUT / 'per_parameter_identifiability.png'}", flush=True)
    print(f"Wrote: {OUT / 'fim_summary.json'}", flush=True)
    print(f"Wrote: {OUT / 'per_parameter_table.csv'}", flush=True)
    print(flush=True)
    print("=== HEADLINE ===", flush=True)
    print(f"  Parameters: {N_PARAMS}", flush=True)
    print(f"  FIM rank: {rank} {'OK' if rank == N_PARAMS else 'FAIL'}", flush=True)
    print(f"  Condition number: {cond:.2e} {'OK' if cond < 1e10 else 'FAIL'}", flush=True)
    if rank < N_PARAMS:
        print(flush=True)
        print("UNIDENTIFIABLE DIRECTIONS:", flush=True)
        for k in range(N_PARAMS - rank):
            v = eigvecs[:, k]
            top = np.argsort(np.abs(v))[::-1][:3]
            comps = ", ".join(f"{PARAM_NAMES[i]}({v[i]:+.3f})" for i in top)
            print(f"  λ={eigvals[k]:.3e}  -> {comps}", flush=True)

    rank_ok = rank == N_PARAMS
    cond_ok = cond < 1e10
    if rank_ok and cond_ok:
        print(flush=True)
        print("ACCEPTANCE: PASS  (full rank + cond < 1e10)", flush=True)
        return 0
    print(flush=True)
    print(f"ACCEPTANCE: FAIL  (rank_ok={rank_ok}, cond_ok={cond_ok})", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

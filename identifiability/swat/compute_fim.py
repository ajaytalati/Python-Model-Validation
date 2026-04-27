"""Fisher Information Matrix analysis of the Option D SWAT model.

Method
------
Sensitivity-based FIM proxy. For a state-space SDE the rigorous Fisher
information uses particle-filter likelihood; the standard practical proxy
is the Jacobian of the deterministic predicted observation trajectory:

    J[k, j] = ∂y_pred[k] / ∂θ_j

The FIM is then F = J^T · Σ_obs⁻¹ · J. Rank, condition number, and
per-parameter scores are read off this matrix.

Scope
-----
We analyse parameters that affect the DETERMINISTIC observation MEAN
trajectory. These are:

  Latent dynamics (14):  κ, λ, γ_3, β_Z, A_scale, φ_0, τ_W, τ_Z, τ_a, τ_T,
                          μ_0, μ_E, η, α_T
  New entrainment   (3): λ_amp_W, λ_amp_Z, V_n_scale
  Observation means (9): HR_base, α_HR, c_tilde, δ_c,
                          λ_base, λ_step, W_thresh,
                          s_base, α_s, β_s

Total: 26 parameters.

NOT analysed here (separate variance-based analysis would be needed):
  Latent diffusion (4): T_W, T_Z, T_a, T_T
  Observation noise (2): σ_HR, σ_s

These six are identifiable through residual VARIANCE, not through mean-
trajectory sensitivity. Their omission is documented in the markdown
write-up.

Multi-operating-point design
----------------------------
V_n_scale is invisible at V_n=0 (damp(0)=1). To excite it, we compute J
at three operating points and stack:
    (V_h, V_n, V_c) ∈ {(1, 0, 0), (1, 1, 0), (1, 3, 0)}

Output
------
- results/sensitivity_heatmap.png : |J[k, j]| matrix
- results/eigenvalue_plot.png     : eigenvalue spectrum
- results/condition_number_vs_subset.png : how sensitivity grows with operating points
- results/fim_summary.json        : rank, condition number, per-param scores
- results/per_parameter_table.csv : parameter-by-parameter breakdown
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force x64 — non-negotiable for FIM on a stiff system
jax.config.update("jax_enable_x64", True)

import diffrax

from model_validation.models.swat.option_c_dynamics import option_c_parameters

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

PHI_MORNING = -np.pi / 3.0
A_SCALE_DEFAULT = 6.0


# =========================================================================
# Parameter packing — fixed order, mean-trajectory parameters only
# =========================================================================

# Two parameter-set choices: with or without tau_T as free parameter.
# tau_T is a well-known degeneracy partner with (mu_0, mu_E, eta) due to the
# rate-vs-time scaling symmetry in scalar Stuart-Landau dynamics. Standard
# practice: pin to physiological value (48h = 2 days) and infer the rate
# parameters relative to it. Set INCLUDE_TAU_T=False for the "fixed tau_T"
# analysis.
INCLUDE_TAU_T = False
# λ_amp_Z and λ_amp_W are structurally degenerate (only their product
# affects E_dyn). Pin λ_amp_Z to its calibrated value and infer λ_amp_W only.
INCLUDE_LAMBDA_AMP_Z = False

PARAM_NAMES_FULL = [
    # Latent drift (excluding diffusion temperatures)
    "kappa", "lmbda", "gamma_3", "beta_Z", "A_scale", "phi_0",
    "tau_W", "tau_Z", "tau_a", "tau_T",
    "mu_0", "mu_E", "eta", "alpha_T",
    # Option D entrainment formula
    "lambda_amp_W", "lambda_amp_Z", "V_n_scale",
    # Observation channel MEANS
    "HR_base", "alpha_HR",
    "c_tilde", "delta_c",
    "lambda_base", "lambda_step", "W_thresh",
    "s_base", "alpha_s", "beta_s",
]
_excluded = []
if not INCLUDE_TAU_T:
    _excluded.append("tau_T")
if not INCLUDE_LAMBDA_AMP_Z:
    _excluded.append("lambda_amp_Z")
PARAM_NAMES = [p for p in PARAM_NAMES_FULL if p not in _excluded]
N_PARAMS = len(PARAM_NAMES)

# Observation noise stds (for FIM weighting). These are the parameters
# we are NOT analysing in J; they enter only as the Σ_obs weighting.
SIGMA_HR = 8.0
SIGMA_S = 15.0
# Sleep level variance — bounded variance of a 3-level ordinal in [0,2]
SLEEP_VAR_PROXY = 0.5
# Steps Poisson — variance ≈ mean ≈ peak rate × bin_hours; use a typical mean
STEPS_VAR_PROXY = 12.5


# =========================================================================
# Build the differentiable predicted observation trajectory
# =========================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def _circadian(t_days, V_c_hours, phi_0):
    return jnp.sin(2.0 * jnp.pi * (t_days - V_c_hours / 24.0) + phi_0)


def _entrainment_quality(W, Z, a, T, V_h, V_n, V_c, p_dict):
    A_W = p_dict["lambda_amp_W"] * V_h
    A_Z = p_dict["lambda_amp_Z"] * V_h
    B_W = V_n - a + p_dict["alpha_T"] * T
    B_Z = -V_n + p_dict["beta_Z"] * a
    amp_W = _sigmoid(B_W + A_W) - _sigmoid(B_W - A_W)
    amp_Z = _sigmoid(B_Z + A_Z) - _sigmoid(B_Z - A_Z)
    damp = jnp.exp(-V_n / p_dict["V_n_scale"])
    V_c_max = p_dict["V_c_max"]
    V_c_eff = jnp.minimum(jnp.abs(V_c), V_c_max)
    phase = jnp.cos(jnp.pi * V_c_eff / (2.0 * V_c_max))
    return damp * amp_W * amp_Z * phase


def _swat_drift(t, x, V_h, V_n, V_c, p_dict):
    W, Z, a, T = x[0], x[1], x[2], x[3]
    lam = p_dict["lmbda"]
    kappa = p_dict["kappa"]
    gamma_3 = p_dict["gamma_3"]
    beta_Z = p_dict["beta_Z"]
    A_scale = p_dict["A_scale"]
    phi_0 = p_dict["phi_0"]
    tau_W = p_dict["tau_W"]
    tau_Z = p_dict["tau_Z"]
    tau_a = p_dict["tau_a"]
    tau_T = p_dict["tau_T"]
    mu_0 = p_dict["mu_0"]
    mu_E = p_dict["mu_E"]
    eta = p_dict["eta"]
    alpha_T = p_dict["alpha_T"]

    C_eff = _circadian(t, V_c, phi_0)
    u_W = lam * C_eff + V_n - a - kappa * Z + alpha_T * T
    u_Z = -gamma_3 * W - V_n + beta_Z * a

    dW = (_sigmoid(u_W) - W) / tau_W
    dZ = (A_scale * _sigmoid(u_Z) - Z) / tau_Z
    da = (W - a) / tau_a

    E = _entrainment_quality(W, Z, a, T, V_h, V_n, V_c, p_dict)
    mu = mu_0 + mu_E * E
    dT = (mu * T - eta * T ** 3) / tau_T

    return jnp.stack([dW, dZ, da, dT])


def _vec_to_params(theta_vec):
    """Build params dict from theta vector. Pin excluded params to defaults."""
    p = {name: theta_vec[i] for i, name in enumerate(PARAM_NAMES)}
    if "tau_T" not in PARAM_NAMES:
        p["tau_T"] = 2.0       # spec value, in days (= 48 hours)
    if "lambda_amp_Z" not in PARAM_NAMES:
        p["lambda_amp_Z"] = 8.0  # calibrated default
    # V_c_max is always pinned at 3.0 (clinical interpretation: any phase
    # shift > 3h is pathology). Visible only at V_c > 0 operating points,
    # so identifiability would need V_c-varying data — pin it instead.
    p["V_c_max"] = 3.0
    return p


def predict_observation_means(theta_vec, V_h, V_n, V_c, init_state, t_grid):
    """Run the ODE under given controls and return the predicted observation
    mean trajectory — a flat 1-D array of length (n_pts × 4_channels).

    Channel order: [HR_mean(t), sleep_expected_level(t), step_rate(t), stress_mean(t)]
    where each is a length-n_pts array.
    """
    p = _vec_to_params(theta_vec)

    def vf(t, y, args):
        return _swat_drift(t, y, V_h, V_n, V_c, p)

    D = float(t_grid[-1])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Kvaerno5(),
        t0=0.0, t1=D, dt0=0.01,
        y0=init_state,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=t_grid),
        max_steps=200_000,
    )
    traj = sol.ys                                 # (n_pts, 4)
    W = traj[:, 0]; Zt = traj[:, 1]; T_amp = traj[:, 3]

    # Observation channel means (deterministic part).
    hr_mean = p["HR_base"] + p["alpha_HR"] * W
    # Sleep level expected value: 0·P(wake) + 1·P(light) + 2·P(deep)
    #   = (1 - P(wake)) + P(deep) = σ(Z - c1) + σ(Z - c2)
    c1 = p["c_tilde"]
    c2 = c1 + p["delta_c"]
    sleep_mean = _sigmoid(Zt - c1) + _sigmoid(Zt - c2)
    # Step rate (per 15-min bin): r(W) · 0.25
    step_rate = (p["lambda_base"] + p["lambda_step"]
                 * _sigmoid(10.0 * (W - p["W_thresh"]))) * 0.25
    # Stress mean
    stress_mean = p["s_base"] + p["alpha_s"] * W + p["beta_s"] * V_n

    return jnp.concatenate([hr_mean, sleep_mean, step_rate, stress_mean])


# =========================================================================
# Reference parameter point and operating-point design
# =========================================================================

def reference_theta() -> jnp.ndarray:
    """Reference parameter vector at the calibrated Option D defaults."""
    p = option_c_parameters()
    # add observation-channel defaults (from clinician_plots._OBS_DEFAULTS)
    obs_defaults = {
        "HR_base": 50.0, "alpha_HR": 25.0,
        "c_tilde": p["c_tilde"], "delta_c": 1.5,
        "lambda_base": 0.5, "lambda_step": 200.0, "W_thresh": 0.6,
        "s_base": 30.0, "alpha_s": 40.0, "beta_s": 10.0,
    }
    p = {**p, **obs_defaults}
    return jnp.array([float(p[name]) for name in PARAM_NAMES])


# =========================================================================
# Compute J at multiple operating points and stack
# =========================================================================

def compute_stacked_jacobian(theta_0, operating_points, init_state,
                              D_days=14.0, n_per_day=24):
    """Compute J at each operating point and stack vertically.

    Returns:
        J_stack: shape (n_ops × n_pts × 4_channels, N_PARAMS)
        op_labels: list of strings describing each operating point
    """
    n_pts = int(D_days * n_per_day)
    t_grid = jnp.linspace(0.0, D_days, n_pts + 1)

    rows = []
    op_labels = []
    for V_h, V_n, V_c in operating_points:
        # Build a closure of theta -> predicted obs at this op point.
        def predict_at_op(theta, _V_h=V_h, _V_n=V_n, _V_c=V_c):
            return predict_observation_means(
                theta, _V_h, _V_n, _V_c, init_state, t_grid
            )
        # JAX autodiff
        J = jax.jacrev(predict_at_op)(theta_0)            # shape (4*n_pts, N_PARAMS)
        rows.append(np.asarray(J))
        op_labels.append(f"V_h={V_h}, V_n={V_n}, V_c={V_c}")
        print(f"  Computed J at {op_labels[-1]}: shape {J.shape}, "
              f"|J|_F = {float(jnp.linalg.norm(J)):.3e}")
    J_stack = np.vstack(rows)
    return J_stack, op_labels, n_pts


# =========================================================================
# Build observation noise weighting
# =========================================================================

def build_sigma_inv_diagonal(n_pts):
    """Per-observation inverse variance (diagonal of Σ_obs⁻¹).

    Stack: 4 channels × n_pts samples each.
    """
    inv_hr = np.full(n_pts, 1.0 / SIGMA_HR ** 2)
    inv_sleep = np.full(n_pts, 1.0 / SLEEP_VAR_PROXY)
    inv_steps = np.full(n_pts, 1.0 / STEPS_VAR_PROXY)
    inv_stress = np.full(n_pts, 1.0 / SIGMA_S ** 2)
    return np.concatenate([inv_hr, inv_sleep, inv_steps, inv_stress])


# =========================================================================
# Main FIM analysis
# =========================================================================

def main():
    print("=" * 70)
    print("FIM analysis — Option D model")
    print("=" * 70)
    print(f"Parameters analysed: {N_PARAMS}")
    print(f"Parameter names: {PARAM_NAMES}")
    print()

    theta_0 = reference_theta()
    init_state = jnp.array([0.5, 3.5, 0.5, 0.5])

    # Operating-point design — span both V_h (to break λ_amp saturation
    # at healthy values) and V_n (so V_n_scale becomes visible via damp).
    operating_points = [
        (0.3, 0.0, 0.0),      # low V_h — A_W, A_Z unsaturated → λ_amp_* visible
        (0.7, 0.0, 0.0),      # mid V_h
        (1.0, 0.0, 0.0),      # healthy
        (1.0, 1.0, 0.0),      # moderate V_n — V_n_scale visible
        (1.0, 3.0, 0.0),      # strong V_n
    ]
    print(f"Operating points: {operating_points}")
    print()

    print("Computing stacked Jacobian (autograd through diffrax)...")
    J_stack, op_labels, n_pts = compute_stacked_jacobian(
        theta_0, operating_points, init_state, D_days=14.0, n_per_day=24
    )
    print(f"Total stacked J shape: {J_stack.shape}\n")

    # Build the per-observation inverse variance, replicated across operating points
    sigma_inv_one_op = build_sigma_inv_diagonal(n_pts + 1)        # (4 × (n_pts+1),)
    sigma_inv_full = np.tile(sigma_inv_one_op, len(operating_points))
    assert sigma_inv_full.shape[0] == J_stack.shape[0], \
        f"sigma_inv {sigma_inv_full.shape} vs J {J_stack.shape}"

    # Form the FIM. Use symmetric weighted form: F = J^T · diag(σ_inv) · J.
    # This is the Cramer-Rao-bound matrix.
    print("Forming FIM = J^T · Σ_obs⁻¹ · J ...")
    W = sigma_inv_full[:, None]          # (n_obs, 1)
    F = J_stack.T @ (W * J_stack)         # (N_PARAMS, N_PARAMS), symmetric
    print(f"FIM shape: {F.shape}\n")

    # ====== SVD of J for rank/conditioning ======
    print("SVD of J ...")
    # Weighted J: rows scaled by sqrt(σ_inv) so SVD reflects weighted FIM
    sqrt_w = np.sqrt(sigma_inv_full[:, None])
    J_weighted = sqrt_w * J_stack
    U, sigma, Vh = np.linalg.svd(J_weighted, full_matrices=False)
    print(f"Singular values (largest 5): {sigma[:5]}")
    print(f"Singular values (smallest 5): {sigma[-5:]}")
    print()

    # FIM eigendecomposition
    print("FIM eigendecomposition ...")
    eigvals, eigvecs = np.linalg.eigh(F)        # ascending order
    # FIM eigvals = sigma**2 of weighted J (up to numerics)
    print(f"FIM eigenvalues (largest 5): {eigvals[-1:-6:-1]}")
    print(f"FIM eigenvalues (smallest 5): {eigvals[:5]}")
    cond = float(eigvals[-1] / max(eigvals[0], 1e-30))
    print(f"FIM condition number: {cond:.3e}")
    rank_eps = max(eigvals) * 1e-10
    rank = int(np.sum(eigvals > rank_eps))
    print(f"Numerical rank (eigval > {rank_eps:.2e}): {rank} / {N_PARAMS}")
    print()

    # ====== Per-parameter analysis ======
    print("Per-parameter identifiability analysis ...")
    # Diagonal FIM entry F_jj = self-information about parameter j
    F_diag = np.diag(F).copy()
    # Marginal Cramer-Rao bound: F⁻¹_jj = (1 / individual identifiability score)
    # If F is singular, use pseudo-inverse and mark non-identifiable params
    try:
        F_inv = np.linalg.inv(F + 1e-12 * np.eye(N_PARAMS) * eigvals.max())
        crb_diag = np.diag(F_inv)
        identifiable_individual = (1.0 / np.sqrt(np.maximum(crb_diag, 1e-30)))
    except np.linalg.LinAlgError:
        F_inv = np.linalg.pinv(F)
        crb_diag = np.diag(F_inv)
        identifiable_individual = (1.0 / np.sqrt(np.maximum(crb_diag, 1e-30)))

    # Most-collinear partner: param k that has largest |F_jk| / sqrt(F_jj F_kk)
    def correlation_to_others(j):
        denoms = np.sqrt(F_diag[j] * F_diag) + 1e-30
        cors = np.abs(F[j, :]) / denoms
        cors[j] = 0.0
        k = int(np.argmax(cors))
        return k, float(cors[k])

    # Build per-parameter table
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

    # ====== Plots ======
    print("\nGenerating plots ...")

    # Sensitivity heatmap — log10 |J|, columns = parameters, rows aggregated
    # by channel (mean over time per channel × operating point).
    # J_stack rows: (n_ops × n_pts × 4) blocks. Average within each (op, channel).
    mean_abs_J = np.abs(J_stack).reshape(
        len(operating_points), 4, n_pts + 1, N_PARAMS
    ).mean(axis=2)              # (n_ops, 4_channels, N_PARAMS)
    # flatten (n_ops, 4) into rows
    heatmap = mean_abs_J.reshape(-1, N_PARAMS)
    row_labels = [f"{op_labels[o]} | {ch}"
                   for o in range(len(operating_points))
                   for ch in ["HR", "sleep", "steps", "stress"]]
    fig, ax = plt.subplots(figsize=(14, 6))
    log_h = np.log10(heatmap + 1e-12)
    im = ax.imshow(log_h, aspect="auto", cmap="viridis")
    ax.set_xticks(range(N_PARAMS))
    ax.set_xticklabels(PARAM_NAMES, rotation=75, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    plt.colorbar(im, ax=ax, label="log₁₀ |∂y_mean/∂θ|")
    ax.set_title("Sensitivity heatmap — mean |J| per (op-point, channel) row")
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_heatmap.png", dpi=130)
    plt.close(fig)

    # Eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(11, 5))
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

    # Per-parameter individual-info bar chart
    fig, ax = plt.subplots(figsize=(13, 6))
    scores = [t["individual_id_score"] for t in table]
    colors = ["steelblue"] * N_PARAMS
    # highlight the 3 new entrainment-formula params
    for j, name in enumerate(PARAM_NAMES):
        if name in ("lambda_amp_W", "lambda_amp_Z", "V_n_scale"):
            colors[j] = "crimson"
    ax.bar(range(N_PARAMS), scores, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(N_PARAMS))
    ax.set_xticklabels(PARAM_NAMES, rotation=75, fontsize=8)
    ax.set_ylabel("Individual identifiability score (1/CRB std, log scale)")
    ax.set_title("Per-parameter identifiability — Option D model "
                  "(red = new entrainment params)")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(OUT / "per_parameter_identifiability.png", dpi=130)
    plt.close(fig)

    # ====== Save summaries ======
    summary = {
        "n_params": N_PARAMS,
        "rank": rank,
        "condition_number_FIM": cond,
        "eigvals_largest_5": eigvals[-1:-6:-1].tolist(),
        "eigvals_smallest_5": eigvals[:5].tolist(),
        "operating_points": op_labels,
        "param_table": table,
    }
    (OUT / "fim_summary.json").write_text(json.dumps(summary, indent=2))

    # CSV for human reading
    import csv
    with (OUT / "per_parameter_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    print(f"\nWrote: {OUT / 'sensitivity_heatmap.png'}")
    print(f"Wrote: {OUT / 'eigenvalue_plot.png'}")
    print(f"Wrote: {OUT / 'per_parameter_identifiability.png'}")
    print(f"Wrote: {OUT / 'fim_summary.json'}")
    print(f"Wrote: {OUT / 'per_parameter_table.csv'}")
    print()
    print("=== HEADLINE ===")
    print(f"  Parameters: {N_PARAMS}")
    print(f"  FIM rank: {rank} {'✓' if rank == N_PARAMS else '✗'}")
    print(f"  Condition number: {cond:.2e} {'✓' if cond < 1e10 else '✗'}")
    if rank < N_PARAMS:
        print()
        print("UNIDENTIFIABLE DIRECTIONS (eigenvectors of small eigvals):")
        for k in range(N_PARAMS - rank):
            v = eigvecs[:, k]                       # ascending eigvals
            top_components = np.argsort(np.abs(v))[::-1][:5]
            comps = ", ".join(
                f"{PARAM_NAMES[i]}({v[i]:+.3f})" for i in top_components
            )
            print(f"  lambda={eigvals[k]:.3e}  -> {comps}")

    # Acceptance / CI exit code
    rank_ok = rank == N_PARAMS
    cond_ok = cond < 1e10
    if rank_ok and cond_ok:
        print()
        print("ACCEPTANCE: PASS  (full rank + cond < 1e10)")
        return 0
    else:
        print()
        print(f"ACCEPTANCE: FAIL  (rank_ok={rank_ok}, cond_ok={cond_ok})")
        return 1


if __name__ == "__main__":
    sys.exit(main())

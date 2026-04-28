# Fisher-Information Identifiability — FSA-high-res Model

**Date:** 2026-04-27
**Model version:** Vendored from OT-Control PR #7 (`feature/fsa-adapter`, v1.2.1) — `mu_0 = +0.02` (deployed) not `mu_0 = -0.3` (v4.1 spec).
**Companion analysis:** [`stability/fsa_high_res/lyapunov_stability_analysis.md`](../../stability/fsa_high_res/lyapunov_stability_analysis.md)

## TL;DR

**The FSA-high-res model is well-identifiable.** All 10 deterministic drift parameters are jointly identifiable from the latent state trajectory under a multi-operating-point design:

| Metric | Result |
|:---|:---|
| Parameters analysed | 10 |
| FIM rank | **10 / 10** ✓ |
| Condition number | **1.69 × 10⁵** ✓ (well below the 1e10 acceptance threshold) |
| Largest eigenvalue | 2.39 × 10⁶ |
| Smallest eigenvalue | 1.41 × 10¹ |

The FIM is **four orders of magnitude better-conditioned than SWAT's** (κ = 4.77 × 10⁹). Two structural reasons: (i) FSA has half as many free parameters, no observation-channel coupling, no entrainment-amplitude formula degeneracies; (ii) the Stuart-Landau scaling symmetry that bites SWAT's `(τ_T, μ_0, μ_E, η)` quartet is broken in FSA because `μ(B, F)` is *state-dependent* — `μ_0` and `η` cease to be a clean rate-vs-amplitude pair when the system traverses regions of different μ.

## Method

Sensitivity-based linearised FIM via JAX autograd through a `diffrax` Tsit5 deterministic solve. Following the standard practical proxy:

```
J[k, j] = ∂y_pred[k] / ∂θ_j
F = Jᵀ Σ⁻¹ J
```

Implementation: [`compute_fim.py`](compute_fim.py).

### Scope

The 10 parameters analysed are the deterministic-drift block:

| Block | Parameters |
|:---|:---|
| Fitness     | `tau_B, alpha_A` |
| Strain      | `tau_F, lambda_B, lambda_A` |
| Bifurcation | `mu_0, mu_B, mu_F, mu_FF` |
| Landau      | `eta` |

**Excluded:** the three diffusion temperatures `sigma_B, sigma_F, sigma_A`. These enter only the noise level — deterministic-trajectory sensitivity is zero. Identifiability of these three needs a separate variance-residual analysis (deferred; same status as SWAT's diffusion temperatures).

### Observation model

FSA's vendoring deliberately omits the upstream observation model (HR / sleep / stress / steps live with the upstream filtering pipeline, not with OT-Control's vendored copy). For this FIM proxy the "observation" is the latent state itself — `(B(t), F(t), A(t))` taken as directly observed with fixed Gaussian noise:

```
σ_B = σ_F = σ_A = 0.05
```

This is the **best-case identifiability bound**. If in production filter the latent state is observed only via noisy proxies, achievable identifiability can only be worse.

### Multi-operating-point design

The 10 parameters are not all visible at a single (T_B, Φ) pair:

- `tau_F`, `lambda_B`, `lambda_A` are invisible at Φ = 0 (F stays near zero).
- `mu_FF` (the F² term) is invisible at low F.
- `alpha_A` is dampened when A is small.
- `mu_B`, `mu_F`, `mu_FF` depend on the (B, F) state varying.

So J is computed at **four operating points** and stacked:

| (T_B, Φ) | Why |
|:---|:---|
| (0.0, 0.0)   | unfit-recovery — `mu_0`, `tau_B`, `eta` visible at low (B, F) |
| (0.5, 0.05)  | healthy reference — full μ(B, F) coupling at moderate state |
| (0.7, 0.5)   | moderate overtraining — `mu_FF` visible (high F²) |
| (1.0, 2.0)   | extreme overtraining — large dynamic range |

At each, the deterministic ODE is integrated for D = 14 days (the upstream POC horizon) at 24 points/day. Stacked Jacobian: shape `(4 × 14×24+1) × 3 channels × 10 params) = 4044 × 10`.

## Results

### Eigenvalue spectrum

![eigenvalue spectrum](https://raw.githubusercontent.com/ajaytalati/Python-Model-Validation/main/identifiability/fsa_high_res/results/eigenvalue_plot.png)

Three orders of magnitude between largest (2.4 × 10⁶, dominated by `mu_0`) and smallest (1.4 × 10¹, the constrained tail). All ten eigenvalues sit well above the rank threshold.

### Sensitivity heatmap

![sensitivity heatmap](https://raw.githubusercontent.com/ajaytalati/Python-Model-Validation/main/identifiability/fsa_high_res/results/sensitivity_heatmap.png)

Reads as expected: `tau_F`, `lambda_B`, `lambda_A` light up at `Phi > 0` rows (when F is non-zero); `mu_FF` lights up only at the (T_B=0.7, Φ=0.5) and (T_B=1.0, Φ=2.0) rows; `alpha_A` is dampest at the (T_B=0, Φ=0) row where A drifts toward zero.

### Per-parameter identifiability table

| Param | Value | Self-info √F_jj | Marginal CRB std | Individual id score | Most collinear partner | Pair corr |
|:---|---:|---:|---:|---:|:---|---:|
| `tau_B`     | 14.000  | 13.5  | 0.218     | 4.58   | `lambda_B` | 0.948 |
| `alpha_A`   |  1.000  | 19.5  | 0.091     | 10.97  | `tau_B`    | 0.824 |
| `tau_F`     |  7.000  | 99.0  | 0.160     | 6.25   | `lambda_B` | **0.998** |
| `lambda_B`  |  3.000  | 153.7 | 0.111     | 8.97   | `tau_F`    | **0.998** |
| `lambda_A`  |  1.500  | 14.6  | 0.089     | 11.20  | `mu_FF`    | 0.631 |
| `mu_0`      |  0.020  | 1158.7 | 3.5e-3   | 287.45 | `eta`      | 0.968 |
| `mu_B`      |  0.300  | 187.3 | 0.0214    | 46.80  | `eta`      | 0.967 |
| `mu_F`      |  0.100  | 39.6  | 0.0448    | 22.32  | `mu_FF`    | 0.807 |
| `mu_FF`     |  0.400  | 39.7  | 0.0448    | 22.33  | `mu_F`     | 0.807 |
| `eta`       |  0.200  | 405.4 | 0.0148    | 67.39  | `mu_0`     | 0.968 |

(`individual_id_score = 1 / sqrt(CRB_jj)` — higher is better.)

### Pairwise correlations to flag

Three pairs sit above 95% correlation. They are jointly identifiable (the FIM is full rank), but joint-posterior estimation will have high variance along each.

**`tau_F ↔ lambda_B` (corr 0.998).** The effective strain-recovery rate is `(1 + lambda_B B + lambda_A A) / tau_F`. At B ≈ const and A ≈ const, only the combination matters. The four operating points span B ∈ {0, 0.5, 0.7, 1} but B equilibrates at the operating point on a τ_B = 14 day timescale, comparable to the 14-day window — so within the FIM window B doesn't traverse its full range. Recommendation: **fix `tau_F` to its calibrated value and infer `lambda_B` only**, or extend the FIM window beyond 14 days. Filing this as a structural concern; not blocking acceptance.

**`mu_0 ↔ eta` (corr 0.968).** Classical Stuart-Landau scaling: at fixed μ, the equilibrium amplitude is A* = √(μ/η), so multiplying both `mu_0` and `eta` by the same constant leaves A* unchanged. The (B, F)-dependent piece of μ(B, F) breaks this for off-fixed-point trajectories — that's why the FIM is full rank — but the trajectory is dominated by the fixed-point value over most of the 14-day window, so the bound on the (`mu_0`, `eta`) joint posterior is loose. Recommendation: **prior on η based on physiological constraints** (typical Landau saturation is η ~ 1/A_max² for amplitude bounds A_max).

**`mu_B ↔ eta` (corr 0.967).** Same scaling argument: `mu_B B` enters μ additively, and at the four operating points B equilibrates fast enough that `mu_B B` looks like a constant offset to `mu_0`. Same recommendation: prior on η.

### Bifurcation block summary

The four bifurcation parameters `(mu_0, mu_B, mu_F, mu_FF)` plus η have the strongest individual identifiability scores (top half of the table). Of these, `mu_0` is the most strongly identifiable (id_score = 287) — makes sense, it sets the baseline μ at the (T_B=0, Φ=0) operating point where mu_B B and mu_F F + mu_FF F² are both near zero. The unfit-recovery operating point is critical: it's where `mu_0` is most cleanly visible without contamination from the other bifurcation parameters.

## Inference recommendations

For downstream filtering (particle filter / SMC²) on FSA:

1. **Pin `tau_F`** at its calibrated 7 days. The 0.998 correlation with `lambda_B` over a 14-day window means there is no informative data to distinguish them. (If running over D > 30 days the correlation drops.)
2. **Tight LogNormal prior on `eta`** based on physiological amplitude bounds: e.g. `eta ~ LogNormal(log(0.2), 0.3)`. The 0.968 correlation with `mu_0` and `mu_B` is the rate-vs-amplitude scaling symmetry; only an external constraint (a prior) can break it strongly.
3. **Hierarchical SMC² assignments:**
   - "Population-level" parameters (shared across patients): the bifurcation block `(mu_0, mu_B, mu_F, mu_FF)` and `eta`. These define the model's qualitative behaviour.
   - "Subject-level" parameters (per-patient): the timescale block `(tau_B, alpha_A, lambda_A)` and the strain coupling `lambda_B`. These vary by physiology.
4. **No diffusion-temperature inference from the deterministic FIM.** `(sigma_B, sigma_F, sigma_A)` need a separate residual-variance analysis (deferred; same status as SWAT's `(T_W, T_Z, T_a, T_T)`).

## Acceptance

| Criterion | Threshold | Result | Status |
|:---|:---|:---|:---:|
| FIM rank | = N (= 10) | 10 | ✓ |
| Condition number | < 1e10 | 1.69e+05 | ✓ |
| Per-parameter individual id score | > 1 (i.e. CRB std < parameter value) | min 4.58 (tau_B) | ✓ |

**ACCEPTANCE: PASS.**

The script `compute_fim.py` exits 0 on pass / non-zero on fail; the CI workflow uses this as the gating signal.

## Reproducibility

```bash
cd Python-Model-Validation
JAX_PLATFORMS=cpu python identifiability/fsa_high_res/compute_fim.py
```

~5 minutes on CPU (Tsit5 explicit RK + JAX autograd). All five output files (heatmap, eigenvalue plot, per-param info bar, JSON summary, CSV table) are reproduced bit-stable across runs (seed-free; the FIM is deterministic).

## What's still open

- **Diffusion-temperature identifiability** (`sigma_B, sigma_F, sigma_A`). Need a residual-variance analysis, separate from this deterministic-FIM study.
- **Joint identifiability with the upstream observation model.** This proxy assumes the latent state is observed directly; the realised observation channels (HR / sleep / stress / steps) couple through additional parameters not in scope here.
- **Long-window study.** Extending the FIM window beyond 14 days would partially break the `tau_F ↔ lambda_B` near-degeneracy, since B traverses more of its dynamic range. Worth doing if the downstream filtering pipeline operates over multi-month windows.

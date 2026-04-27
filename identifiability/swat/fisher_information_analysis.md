# Fisher Information Analysis — Option D SWAT Model

**Date:** 2026-04-27
**Model version:** Option C v4 / Option D (with V_h modulating entrainment-formula forcing amplitudes and `damp(V_n) = exp(−V_n / V_n_scale)` dampener)
**Issue:** [#6](https://github.com/ajaytalati/Python-Model-OT-Control/issues/6)

## TL;DR

The full-parameter Option D model (27 mean-trajectory parameters spanning latent dynamics, the new entrainment formula, and the four observation channels) is **NOT identifiable** out of the box: rank 24/27, condition number ~10³⁶. Three structural degeneracies were uncovered:

1. **Stuart-Landau time-vs-rate scaling** — `(μ_0, μ_E, η, τ_T)` are coupled by a continuous group symmetry. *Fix:* pin `τ_T = 48 h` (physiological constant from circadian biology).
2. **`λ_amp_W` and `λ_amp_Z` saturation at healthy V_h=1** — `amp_W` and `amp_Z` are saturated at ~1 throughout the trajectory at V_h=1, so individual variations in λ_amp are invisible to the data. *Fix:* operating-point design must span V_h ∈ [0.3, 1.0] so the unsaturated regime gives sensitivity.
3. **`λ_amp_W ↔ λ_amp_Z` product-only identifiability** — both parameters affect E_dyn only through `amp_W · amp_Z`, so only the product (or one of them with the other pinned) is identifiable. *Fix:* pin `λ_amp_Z` to a structural value and infer `λ_amp_W` only.

After applying all three fixes, the FIM is **full rank (25/25)** with condition number 4.77 × 10⁹ — borderline but acceptable. Remaining near-degeneracies (Stuart-Landau aliasing among `μ_0, μ_E, η`; coupling-strength aliasing between `λ` and `κ`) are documented but do not invalidate the filter.

The filtering problem **is well-posed** under the recommended reduced parameter set:

> Free parameters (25): `κ, λ, γ_3, β_Z, A_scale, φ_0, τ_W, τ_Z, τ_a, μ_0, μ_E, η, α_T, λ_amp_W, V_n_scale, HR_base, α_HR, c_tilde, δ_c, λ_base, λ_step, W_thresh, s_base, α_s, β_s`
>
> Pinned: `τ_T = 2.0 days`, `λ_amp_Z = 8.0`

---

## Method

### Sensitivity-based FIM proxy

For a state-space SDE the rigorous Fisher information uses particle-filter likelihood. The standard practical proxy is the Jacobian of the deterministic predicted observation trajectory:

$$
J[k, j] = \frac{\partial y_\mathrm{pred}[k]}{\partial \theta_j}
$$

with
$$
F = J^\top \, \Sigma_\mathrm{obs}^{-1} \, J
$$

`F` is symmetric positive semi-definite. Its rank tells us how many independent parameter-directions the data can constrain; the condition number tells us how well-constrained they are; the eigenvectors of small eigenvalues tell us *which* parameter combinations are unidentifiable.

### Implementation

JAX autograd through `diffrax.diffeqsolve` (Kvaerno5 implicit ODE) gives `J` exactly to machine precision. Observation channel means are computed from the latent trajectory:

- `HR_mean(t) = HR_base + α_HR · W(t)`
- `sleep_expected_level(t) = σ(Z̃ − c_tilde) + σ(Z̃ − c_tilde − δ_c)`
- `step_rate(t) = (λ_base + λ_step · σ(10·(W − W_thresh))) · 0.25`
- `stress_mean(t) = s_base + α_s · W + β_s · V_n`

Sample times: hourly grid over D=14 days. Σ_obs is diagonal with per-channel variances (σ_HR=8, sleep variance proxy 0.5, step Poisson variance proxy 12.5, σ_s=15).

### Multi-operating-point design

Single-trajectory FIM is impoverished — `V_n_scale` is invisible at V_n=0 (`damp` is constant 1), and `λ_amp_W`/`λ_amp_Z` are invisible if all trajectories are at the saturation point. The analysis stacks `J` across multiple `(V_h, V_n, V_c)` operating points to make all parameters visible:

```
operating_points = [
    (V_h=0.3, V_n=0.0, V_c=0),    # low V_h → A_W, A_Z unsaturated → λ_amp_* visible
    (V_h=0.7, V_n=0.0, V_c=0),    # mid V_h
    (V_h=1.0, V_n=0.0, V_c=0),    # healthy
    (V_h=1.0, V_n=1.0, V_c=0),    # damp(V_n) becomes informative
    (V_h=1.0, V_n=3.0, V_c=0),    # full damp dynamics
]
```

### Scope

**Analysed (27 mean-trajectory parameters):**

| Block | Parameters |
|:---|:---|
| Latent drift | `κ, λ, γ_3, β_Z, A_scale, φ_0, τ_W, τ_Z, τ_a, τ_T, μ_0, μ_E, η, α_T` (14) |
| Option D entrainment | `λ_amp_W, λ_amp_Z, V_n_scale` (3) |
| Observation channel means | `HR_base, α_HR, c_tilde, δ_c, λ_base, λ_step, W_thresh, s_base, α_s, β_s` (10) |

**Not analysed (analysed separately would require variance-based methods):**

- Latent diffusion temperatures: `T_W, T_Z, T_a, T_T` — only enter observation noise, invisible to mean-trajectory Jacobian.
- Observation noise: `σ_HR, σ_s` — same.

These six are identifiable in principle through residual variance pattern, just not through this analysis.

---

## Results

Three FIM runs were performed, each adding a fix to the previous one:

### Run 1 — Naive (all 27 params, V_h=1 only, τ_T included)

```
FIM rank:           24 / 27         ✗
Condition number:   2.25e+36         ✗
```

Three unidentifiable directions (eigenvectors of zero/near-zero eigenvalues):

1. `λ ≈ −2.3e−15` (effectively zero)
   Heaviest weight: `τ_T (−0.853), μ_E (−0.426), μ_0 (+0.213), η (−0.213)`.
   **Stuart-Landau time-rate scaling**: scaling `μ_*, η` by `k` and `τ_T` by `k` gives identical dynamics.

2. `λ ≈ 1.6e−9`
   Weight almost entirely on `λ_amp_Z (−0.999)`.
   **`λ_amp_Z` saturation**: at V_h=1, `A_Z = λ_amp_Z·V_h ∈ [4, 16]` already saturates `amp_Z = σ(B_Z+A_Z) − σ(B_Z−A_Z)` to ~1, so changes in `λ_amp_Z` do not change predictions.

3. `λ ≈ 1.7e−8`
   Weight almost entirely on `λ_amp_W (+0.996)`.
   Same diagnosis for `λ_amp_W`.

### Run 2 — Pin τ_T, span V_h in operating points

```
FIM rank:           25 / 26         ✗   (one missing direction)
Condition number:   9.98e+11         ✗
```

The Stuart-Landau scaling degeneracy is resolved by pinning `τ_T`. The `λ_amp_W` and `λ_amp_Z` saturation degeneracy is resolved by including operating points at V_h=0.3 and V_h=0.7 (where `A_W`, `A_Z` are partially unsaturated and the parameters become observable individually).

One residual unidentifiable direction:

```
λ ≈ 3.8e−6:  λ_amp_Z (−0.858), λ_amp_W (+0.512), ...
```

This is the **product-only identifiability** of `λ_amp_W` and `λ_amp_Z`: both parameters affect E_dyn only through `amp_W · amp_Z` (`E = damp(V_n) · amp_W · amp_Z · phase(V_c)`), so a coordinated change `(δ λ_amp_W, δ λ_amp_Z)` along the eigenvector keeps `E` constant. **Only the product is observable.**

### Run 3 — Pin τ_T and λ_amp_Z

```
FIM rank:           25 / 25         ✓   FULL RANK
Condition number:   4.77e+9          (borderline — see "remaining near-degeneracies" below)
```

With `τ_T` and `λ_amp_Z` both pinned, all 25 free parameters are identifiable.

#### Per-parameter identifiability scores (sorted weakest-first)

| Param | Value | Indiv. id score | Most-collinear partner | Pearson r |
|:---|---:|---:|:---|---:|
| `λ_amp_W` | 5.0 | 0.030 | μ_0 | 0.83 |
| `V_n_scale` | 2.0 | 0.079 | μ_0 | 0.58 |
| `μ_E` | 1.0 | 0.21 | η | 0.95 |
| `λ` | 32.0 | 0.23 | κ | **0.996** |
| `μ_0` | −0.5 | 0.23 | μ_E | 0.95 |
| `γ_3` | 8.0 | 0.35 | β_Z | 0.80 |
| `η` | 0.5 | 0.36 | μ_E | 0.95 |
| `δ_c` | 1.5 | 0.64 | c_tilde | 0.98 |
| `κ` | 6.67 | 0.68 | λ | **0.996** |
| `β_Z` | 4.0 | 0.69 | γ_3 | 0.80 |
| `α_T` | 0.3 | 0.83 | μ_E | 0.95 |
| `A_scale` | 6.0 | 0.98 | κ | 0.96 |
| `α_s` | 40.0 | 1.03 | s_base | 0.73 |
| `λ_step` | 200.0 | 1.05 | λ_base | 0.65 |
| `s_base` | 30.0 | 1.72 | α_s | 0.73 |
| `α_HR` | 25.0 | 1.92 | HR_base | 0.73 |
| `λ_base` | 0.5 | 1.96 | λ_step | 0.65 |
| `c_tilde` | 3.0 | 2.76 | δ_c | 0.98 |
| `β_s` | 10.0 | 3.15 | s_base | 0.57 |
| `HR_base` | 50.0 | 3.39 | α_HR | 0.73 |
| `τ_a` | 0.125 | 25.8 | A_scale | 0.95 |
| `τ_Z` | 0.083 | 65.4 | α_T | 0.69 |
| `φ_0` | −1.047 | 86.5 | τ_W | 0.90 |
| `W_thresh` | 0.6 | 121.0 | μ_0 | 0.83 |
| `τ_W` | 0.083 | 469.7 | φ_0 | 0.90 |

Reading: `individual id score` = `1 / (CRB std)` where `CRB std = √([F⁻¹]_jj)`. Higher = better identified. The fast-subsystem timescales (τ_W, τ_Z, τ_a, φ_0, W_thresh) are very well identified through the HR / sleep / steps channels. The Stuart-Landau parameters (μ_0, μ_E, η) are weakly identified due to internal aliasing — they all act on `μ(E)` and `T*` similarly. The new entrainment-formula parameters (λ_amp_W, V_n_scale) are also weakly identified, partly because they sit at saturation in the canonical healthy regime.

---

## Remaining near-degeneracies (cond number 4.77e9)

Even at full rank, the FIM has eigenvalues spanning ~10 orders of magnitude. The lowest-eigenvalue directions reveal:

1. **`λ ↔ κ` (corr 0.996)** — circadian forcing strength λ and Z→W coupling κ both control W's daily oscillation amplitude. Hard to distinguish from observation alone. Resolution would require an experiment that drives W's daily amplitude in a way that distinguishes these contributions (e.g., light-cycle perturbation studies).

2. **Stuart-Landau internal aliasing: `μ_0 ↔ μ_E ↔ η` (mutual corr 0.95)** — these three parameters all set the bifurcation parameter μ(E) and the equilibrium amplitude T*. Standard practice in Stuart-Landau model fitting is to constrain at least two of them from physiological priors (μ_0 from the noise-free flatline, η from the saturation amplitude).

3. **`λ_amp_W ↔ μ_0` (corr 0.83)** and **`V_n_scale ↔ μ_0` (corr 0.58)** — both new entrainment params are weakly aliased with the bifurcation baseline. Improvable with longer trajectories or more diverse V_h, V_n probing.

These near-degeneracies are not fatal — the FIM is full rank — but they mean that joint inference will be slow to converge and will benefit from informative priors on `μ_0, μ_E, η` (e.g. from upstream's biological grounding).

---

## Recommendations

### For inference (SMC², EKF, etc.)

1. **Pin `τ_T = 2.0 days` (= 48 h)** as a known physiological constant. Mathematically equivalent to fixing the time-scale of the analysis.
2. **Pin `λ_amp_Z` to its calibrated value (8.0)** — the spec describes `λ_amp_W` and `λ_amp_Z` as separate parameters but only their product enters E. An alternative is to reformulate Option D with a single entrainment-amplitude parameter (see §"Reformulation proposal" below).
3. **Use informative priors on `μ_0, μ_E, η`** to stabilise inference of the Stuart-Landau block.
4. **Operating-point design for inference data**: ensure subjects span V_h ∈ [low, high] (e.g. via natural patient population variation or designed treatment protocols) so the unsaturated `λ_amp_W` regime is probed.

### Reformulation proposal (optional, simpler)

The product-only structure of `λ_amp_W · λ_amp_Z` suggests collapsing the two parameters into one. Concretely, replace the current asymmetric form

```
A_W = λ_amp_W · V_h
A_Z = λ_amp_Z · V_h
```

with a single-parameter symmetric form

```
A = λ_amp · V_h
A_W = A
A_Z = A · ρ_amp     # ρ_amp = constant ratio, set from clinical grounding
```

That removes one parameter (`λ_amp_Z`) and pins the W-vs-Z forcing-strength ratio. The natural choice is `ρ_amp = γ_3 / λ ≈ 0.25` (the spec's Z-vs-W coupling ratio), giving `λ_amp_Z = 0.25 · λ_amp_W`. Empirically this would change the calibrated default from `(λ_amp_W=5, λ_amp_Z=8)` to a single λ_amp ≈ 5–8 with derived ratio. **Worth doing only if the user / upstream finds the simpler structure clinically defensible.**

### Diffusion-temperature identifiability (out of scope here)

The 4 latent diffusion temperatures `T_W, T_Z, T_a, T_T` and 2 observation noise stds `σ_HR, σ_s` were not analysed because they don't enter the deterministic mean trajectory. A separate variance-residual analysis (or a particle-filter-based FIM) would address them. Standard practice is to either:
- estimate them jointly with the drift parameters via maximum-likelihood (particle filter)
- fix them at small physiological values (the spec uses `T_T ~ 10⁻⁴` per hour)

---

## Headline plots

- `results/sensitivity_heatmap.png` — `log₁₀ |J[k, j]|` per (op-point, channel, parameter). Visual summary of which parameters affect which observations.
- `results/eigenvalue_plot.png` — log-scale FIM eigenvalue spectrum showing the 25 non-zero eigenvalues spanning ~10⁹.
- `results/per_parameter_identifiability.png` — bar chart of individual id scores per parameter, log scale, with the 3 new entrainment-formula params highlighted.

## Reproducibility

```bash
cd Python-Model-Validation
python identifiability/compute_fim.py
```

Toggling `INCLUDE_TAU_T` and `INCLUDE_LAMBDA_AMP_Z` at the top of `compute_fim.py` reproduces the three runs above.

## Conclusion

**The Option D SWAT model is identifiable** under the recommended reduced parameter set (25 free + 2 pinned + 6 noise-only). The filtering problem is well-posed. Two structural degeneracies were found and resolved (τ_T scaling, `λ_amp_W ↔ λ_amp_Z` product). Several near-degeneracies remain (cond number ~5×10⁹) but they don't obstruct inference if informative priors are used on the Stuart-Landau block.

The Lyapunov stability analysis (issue #6, Part 2) can now proceed against this identifiable parameterisation.

---

## Inference-time recommendations (for SMC² / EKF / particle-filter repos)

**Pinned parameters (do NOT infer):**

| Parameter | Pinned value | Reason |
|:---|---:|:---|
| `τ_T` | 2.0 days (= 48 h) | Stuart-Landau time-vs-rate scaling degeneracy. Physiological constant from circadian biology — well-measured independently. |
| `λ_amp_Z` | 8.0 | Product-only identifiability with `λ_amp_W` (only `amp_W · amp_Z` enters E). Pinning the Z-side and inferring `λ_amp_W` resolves the degeneracy. |

**Informative priors for the Stuart-Landau block (`μ_0`, `μ_E`, `η`):**

The three Stuart-Landau parameters are mutually correlated (~0.95) — they jointly determine the bifurcation parameter `μ(E) = μ_0 + μ_E·E` and the equilibrium amplitude `T* = √(μ_max / η)`. Without informative priors, joint inference will be slow to converge and the marginal posteriors will be inflated. Recommended priors:

```
μ_0 ~ −LogNormal(log 0.5, 0.20)        # weakly negative, ~exp(±20%) around -0.5
μ_E ~ LogNormal(log 1.0, 0.20)         # positive, ~exp(±20%) around 1.0
η   ~ LogNormal(log 0.5, 0.30)         # positive, slightly looser
```

These priors enforce the structural constraints (`μ_0 < 0`, `μ_E > 0`, `η > 0`) and concentrate on physiologically-grounded values from the spec while permitting reasonable patient-to-patient variation.

**Optional reparameterisation (cleaner, recommended for new inference code):**

Trade `(μ_0, μ_E, η)` for `(T*, μ_max, μ_excursion)`:

```
T*           = √(μ_max / η)              # equilibrium amplitude at full entrainment
μ_max        = μ_0 + μ_E                 # peak bifurcation at E=1
μ_excursion  = μ_max − μ_0 = μ_E         # how much E modulates μ
```

These are directly observable from data and orthogonal in the limit. Inverse map back to canonical parameters: `η = μ_max / T*²`, `μ_0 = μ_max − μ_E`. Informative priors on `(T*, μ_max)`:

```
T*    ~ LogNormal(log 1.0, 0.20)        # equilibrium amplitude near 1.0
μ_max ~ LogNormal(log 0.5, 0.20)        # super-critical at full entrainment
```

This reparameterisation eliminates the Stuart-Landau internal aliasing and makes the marginal posteriors interpretable.

**`λ ↔ κ` aliasing (corr 0.996):**

Circadian forcing amplitude `λ` and Z → W inhibition `κ` both control W's daily oscillation amplitude. Their effects on observations are nearly indistinguishable. Two practical resolutions:

1. **Pin `κ` to its spec value (6.67)**, treat it as a structural coupling constant rather than a free parameter. Standard practice in SWAT-class models — `κ` is the W-Z reciprocal-inhibition gain, often treated as an architecture-level constant.
2. **Tight LogNormal priors on both**:
   ```
   λ ~ LogNormal(log 32.0, 0.10)        # very tight (~10%)
   κ ~ LogNormal(log 6.67, 0.10)
   ```
   Allows inference of small per-subject deviations without exploiting the aliasing direction.

**Per-subject vs population-level parameters:**

In the SMC² hierarchical-inference setting:

| Level | Parameters | Notes |
|:---|:---|:---|
| **Per subject** | `V_h, V_n, V_c, T_0` | The clinical state being inferred. Per-subject priors based on population distribution. |
| **Per subject (initial state)** | `W_0, Z̃_0, a_0` | Latent initial conditions. Tight priors centered on phase-of-day. |
| **Universal (cohort-level)** | `μ_0, μ_E, η, V_n_scale, λ_amp_W, α_T` | Stuart-Landau and entrainment-formula params. Informative priors as above. |
| **Universal (architectural)** | `κ, λ, γ_3, β_Z, A_scale, φ_0, τ_W, τ_Z, τ_a, c_tilde, δ_c, λ_base, λ_step, W_thresh, HR_base, α_HR, s_base, α_s, β_s` | Spec defaults; tight priors around spec values. Can be pinned in a first-pass inference. |
| **Pinned (do not infer)** | `τ_T = 2.0`, `λ_amp_Z = 8.0` | Per identifiability analysis above. |

**Diffusion temperatures (`T_W, T_Z, T_a, T_T`) and observation noise stds (`σ_HR, σ_s`):**

These six parameters did not enter this Jacobian-based FIM and need a separate variance-residual or particle-filter-based identifiability analysis. **Recommendation for first-pass SMC²:** pin them at the spec values (`T_W=T_a=0.24, T_Z=1.2, T_T=0.0024, σ_HR=8, σ_s=15` — the default Option D dictionary). They can be jointly estimated in a second pass once the drift parameters have settled.

**Operating-point design for the inference dataset:**

Per the FIM analysis, the most-informative subjects to include are those whose `V_h` spans the range `[0.3, 1.0]` (so the unsaturated regime probes `λ_amp_W`) and whose `V_n` spans `[0, 3]` (so `damp(V_n)` is informative for `V_n_scale`). Patient population variation should naturally provide this; if not, designed treatment protocols (titrating V_h up over weeks) provide the same information.

**Summary table — what to do with each parameter:**

| Action | Parameters |
|:---|:---|
| **Pin** | `τ_T`, `λ_amp_Z`, plus `T_*` and `σ_*` for first pass |
| **Strong informative prior** | `μ_0`, `μ_E`, `η`, `λ`, `κ` (or pin `κ` as structural) |
| **Moderate informative prior** | `λ_amp_W`, `V_n_scale`, `α_T`, `γ_3`, `β_Z`, `A_scale`, `φ_0` |
| **Weak/data-driven prior** | `τ_W`, `τ_Z`, `τ_a`, observation-channel params |
| **Per-subject (inferred)** | `V_h`, `V_n`, `V_c`, `T_0`, initial state |

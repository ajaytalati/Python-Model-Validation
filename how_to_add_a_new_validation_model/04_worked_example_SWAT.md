# 04 — Worked example: SWAT

A concrete tour of how SWAT was landed, with pointers to the actual
files. Read this for "what does each deliverable actually look like."

## The story

SWAT is a 4-state SDE for sleep–wake–adenosine–testosterone with 3
controls (V_h, V_n, V_c). It was originally specified upstream
(Python-Model-Development-Simulation) and vendored into OT-Control's
v1.1.0. **The vendored copy turned out to have a structural inversion
in the V_h pathway** — the testosterone-amplitude variable T was
maximised when V_h=0 (depleted vitality), the clinically-backwards
direction. This was discovered by an OT-Control optimisation that
recommended `V_h → 0` for a hormone-recovery scenario.

Issue #4 was opened on OT-Control. Rather than patch the vendored
copy in OT-Control directly, the validation repo was created to:

1. Reproduce the bug as a failing gating test against the vendored
   snapshot.
2. Devise a structural fix (Option C / Option D refinement).
3. Re-validate with FIM identifiability + Lyapunov stability
   analyses.
4. Tag the fixed snapshot as `swat-validated-*` so OT-Control could
   safely vendor it.

This story is documented at:

- [`docs/swat/option_c_results/README.md`](../docs/swat/option_c_results/README.md)
- The Issue #4 thread on OT-Control (V_h inversion analysis).

## What got produced

### 1. Vendored dynamics

[`src/model_validation/models/swat/`](../src/model_validation/models/swat/):

- `vendored_dynamics.py` — drift, diffusion, state_clip — verbatim
  copy of OT-Control v1.1.0's vendored SWAT model (the broken one).
- `vendored_parameters.py` — 26 parameters, hours → days converted
  at the boundary.
- `option_c_dynamics.py` — the Option D refinement: V_h removed
  from u_W's slow drift, amplitude formula reformulated to
  `σ(B+A) - σ(B-A)` with `A = λ_amp · V_h`, multiplicative
  V_n dampener `damp(V_n) = exp(-V_n / V_n_scale)`.
- `__init__.py` — exports `vendored_model()` and
  `option_c_model()`.

### 2. Gating tests

[`tests/swat/`](../tests/swat/) — 17 tests across 9 files:

- `test_anabolicity.py` — V_h is anabolic on T.
- `test_bifurcation.py` — E_dyn crosses zero at the documented
  threshold; T responds super-critically above it.
- `test_dose_response.py` — small V_h change → small T change.
- `test_equilibrium.py` — under healthy "do-nothing" controls, the
  state converges to the documented fixed point.
- `test_phase.py` — V_c shift > 0 desynchronises the system; V_c
  back to 0 resynchronises.
- `test_anti_anabolicity.py` — V_n is catabolic on T.
- `test_no_backwards_optimum.py` — argmax over (V_h, V_n) lies in
  the clinically-correct corner.
- `test_v_n_dampening.py` — damp(V_n) is monotonic in V_n.
- `test_lambda_robustness.py` — V_h responsiveness across (λ, λ_Z).

`conftest.py` exposes three CLI flags: `--variant`,
`--lambda-base`, `--lambda-z-base`. CI runs
`pytest tests/swat/ --variant option-c -v --tb=short`.

### 3. FIM identifiability

[`identifiability/swat/`](../identifiability/swat/):

- `compute_fim.py` — multi-operating-point FIM over 27 latent +
  observation parameters at three (V_h, V_n) operating points.
  Pins τ_T=2.0 and λ_amp_Z=8.0 to break documented degeneracies.
  Final result: rank 25/25, κ = 4.77×10⁹.
- `fisher_information_analysis.md` — write-up with three runs
  documenting the structural degeneracies that motivated the
  pinning.
- `results/` — heatmap, eigenvalue spectrum, JSON summary, per-
  parameter table.

**Lessons from SWAT's FIM:**

- The `λ ↔ κ` aliasing (corr 0.996) — `λ` and `κ` are nearly
  perfectly collinear over the analysed window. Recommended: pin
  `κ` as an architectural constant.
- `V_n_scale` was unidentifiable from data with V_n=0 only. Resolved
  by adding pathology operating points.
- 4 latent diffusion temperatures + 2 observation noise stds
  could not be analysed by deterministic FIM; need a separate
  variance-residual analysis (deferred).

### 4. Lyapunov stability

[`stability/swat/`](../stability/swat/):

- `corner_case_sweep.py` — 8-corner sweep over
  `(V_h, V_n, V_c) ∈ {0,1} × {0,5} × {0,6}`. JIT'd vmap'd diffrax
  Tsit5 solvers. Deterministic + 32 stochastic trajectories per
  corner. ~50 seconds on GPU.
- `lyapunov_stability_analysis.md` — write-up with the Stuart-
  Landau Lyapunov function derivation.
- `results/` — corner T(D) bar chart, T(t) trajectory plot, phase-
  plane plot, IC-sweep plot, JSON summary.

**Result:** corner 4 (V_h=1, V_n=0, V_c=0) is the unique super-
critical attractor (T(D) = 0.983); all 7 pathological corners
collapse to T = 0; basin at the healthy corner is the entire
physical state space.

### 5. Manifest + CI

- [`snapshots/manifest.json`](../snapshots/manifest.json) — has one
  pre-fix snapshot under `models.swat`.
- [`.github/workflows/swat_validation.yml`](../.github/workflows/swat_validation.yml)
  — three jobs (gating, identifiability+stability, tag-validated),
  paths trigger on `tests/swat/**`, `identifiability/swat/**`,
  `stability/swat/**`. Auto-tags `swat-validated-<date>-<sha>` on
  green merge to `main`.

## What was hard / non-obvious

**The V_h structural inversion** was not a bug in the upstream
spec — it was a consequence of how V_h modulated the fast
wake-promoting drift `u_W`. The fix wasn't "change V_h's role";
it was "remove V_h from u_W and route it through the entrainment
amplitude formula instead". Three iterations were needed to find
a fix that:
- Made V_h anabolic on T (the original goal),
- Did not break the W↔Z flip-flop dynamics that beta_Z=4 had
  established,
- Kept entrainment quality E ≈ 1 in the healthy corner.

**The V_n bell-shape** was a separate issue (#5): under the original
formulation, increasing V_n past a threshold made T rise instead of
fall (a pathology corner that looked clinically healthy). Fix: a
multiplicative dampener `damp(V_n) = exp(-V_n / V_n_scale)` outside
the σ-function.

**Tightening V_c pathology to ≥3 hours** was a calibration call
made during stability validation — the original |V_c| ≥ 6 threshold
was too lenient; a 3-hour shift is already clinically pathological.

**The FIM degeneracies** came one by one. First run: rank 25/27 with
two redundant pairs. Second run after pinning τ_T: rank 26/26 with
λ ↔ κ flagged. Third run after pinning λ_amp_Z: rank 25/25 with
κ < 1e10 — passes acceptance.

## What's still open

- 4 latent diffusion temperatures + 2 obs noise stds need a
  variance-residual analysis (separate from the deterministic FIM).
- λ ↔ κ aliasing (corr 0.996) — recommended pinning κ as an
  architectural constant; not yet enforced upstream.

## Reading order for someone landing FSA-high-res next

1. Read [`02_validation_contract.md`](02_validation_contract.md)
   for the formal acceptance criteria.
2. Skim SWAT's three deliverables (FIM, Lyapunov, gating tests)
   to see what the actual outputs look like.
3. Follow [`03_step_by_step_guide.md`](03_step_by_step_guide.md)
   step by step.
4. Use SWAT's `compute_fim.py` and `corner_case_sweep.py` as
   templates — most of the JAX/diffrax wiring transfers; only the
   model-specific bits (scenario corners, parameter list, basin
   condition) change.

That's the whole story.

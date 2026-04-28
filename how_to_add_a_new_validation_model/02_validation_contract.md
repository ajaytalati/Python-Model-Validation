# 02 — The validation contract

This is the reference document. Read once, refer back as needed.

Every model entry must produce four deliverables. Each has a precise
acceptance criterion. The CI workflow's auto-tag step
(`<model>-validated-<date>-<sha>`) only fires when all four pass on
`main`.

## Scope and limitations of this pipeline

Read this section before reading anything else. The validation
discipline this repo runs is **structural / mathematical**, not
empirical-clinical. Be precise about what that means before
labelling a model "validated":

**What this pipeline establishes:**

1. The model's equations + the deployed parameter values jointly
   produce the qualitative behaviours the model was *designed* to
   produce (gating tests).
2. The model's parameters are jointly identifiable from observations
   of the latent state up to a bounded condition number — there are
   no rank-deficient directions in the deterministic parameter space
   under the multi-operating-point design (FIM analysis).
3. The model has a bounded-orbit attractor structure under constant
   controls, with the predicted healthy attractor reached
   numerically and a (model-internal) Lyapunov argument supporting
   stability of the amplitude variable given the slow-state
   equilibrium (Lyapunov sweep).

**What this pipeline does NOT establish:**

1. **Agreement with reality.** Nothing in this pipeline compares
   model predictions to real patient data, real training logs, real
   biomarker trajectories, retrospective overtraining cases, or
   anything else outside the model itself. A model that internally
   passes all three deliverables can still be biophysically wrong
   in ways the pipeline cannot detect.
2. **Magnitude calibration.** The pipeline does not check that
   timescales (`tau_B`, `tau_F`, `tau_T`, `tau_W` …), coupling
   coefficients, or amplitude scales match published values for
   the relevant physiology. The deployed parameter set is taken
   as given.
3. **Coverage of the realised parameter space.** The pipeline tests
   one parameter point — the deployed defaults. It does not search
   the surrounding parameter space to find regions where the model
   would behave clinically wrongly. A small parameter perturbation
   could in principle move the model into such a region, undetected.
4. **Clinical expert review.** The pipeline does not involve a
   sport-medicine, sleep-medicine, or endocrinology expert
   asserting "yes, the parameter signs and magnitudes match what I
   see in clinic". That review is necessary for genuine clinical
   validity and is out of scope here.
5. **Observation-model fidelity.** The vendored copy is the latent
   dynamics only. The full upstream observation channels (HR /
   sleep / stress / steps for SWAT; HR / sleep / stress / steps
   for FSA) are not validated here — they're the concern of the
   upstream filtering pipeline.

**Practical consequence for downstream use.** The
`<model>-validated-<date>-<sha>` tag means "the model has passed
internal-consistency and identifiability gating against its own
design intent". It does **not** mean "this model is clinically
trustworthy". OT-Control vendoring blocks regression of the former;
it does not certify the latter. Any clinical claim derived from a
prescription the OT engine produces still carries the empirical-
validation gap above.

If a future user wants genuine clinical validity, they need:
empirical model-fit against held-out clinical datasets;
expert-physiologist review of the parameter values; comparison to
established sport-science / chrono-medicine literature; and
ideally a prospective study comparing optimised prescriptions to
standard-of-care outcomes. None of that is in this repo.

The deliverables below are the structural-gating discipline the
repo *does* enforce.

## Deliverable 1 — Structural gating tests

**Location:** `tests/<model>/`

**Form:** pytest test suite, ~10–25 tests, organised by the model's
designed structural properties. Each test takes a `model` fixture
and asserts a deterministic property the *model's design* says
should hold under its deployed parameter values. These tests catch
parameter-sign errors, equation regressions, and solver bugs.
**They do not establish agreement with reality** — see the scope
section above.

**Minimum coverage:** every model must include tests for:

- **Sign / direction.** Each control variable pushes the relevant
  amplitude in the direction the model's design intent says it
  should (e.g. SWAT's V_h is *designed* to be anabolic on T;
  FSA-high-res's T_B is *designed* to drive B up). The test
  confirms the deployed parameters preserve that designed sign;
  it does not confirm the sign matches real physiology — that's
  a separate empirical question.
- **Monotonicity.** As a control increases monotonically across its
  range, the response is monotonic in the expected direction
  (SWAT's V_h-anabolicity test; FSA's T_B-builds-fitness test).
- **Dose-response.** A small change in a control produces a small
  response; a large change produces a large response. No
  saturation cliffs in the healthy range.
- **Equilibrium.** Under "do-nothing" controls held constant, the
  state converges to a finite fixed point in the expected basin.
- **Bifurcation / phase.** The model's bifurcation parameter
  crosses zero at the documented threshold (e.g. SWAT's E_dyn
  super-criticality threshold; FSA's μ(B,F) > 0 threshold).
- **Anabolicity vs catabolicity.** The healthy-direction control
  raises the amplitude variable; the pathology-direction control
  lowers it. No clinically backwards optima.

**Variants.** If the model has named calibration variants (SWAT has
`vendored` / `option-c`), expose a pytest CLI flag in `conftest.py`
(SWAT uses `--variant`). The CI workflow runs the canonical variant
(SWAT runs `--variant option-c`).

**Acceptance:** `pytest tests/<model>/ --variant <canonical>` exits
0 and prints "N passed". A pre-fix snapshot (regression target) may
have its own marker (SWAT uses `expected_to_fail_pre_fix`).

**Runtime budget:** 5–10 minutes on CPU. Slower than that and CI
becomes painful.

## Deliverable 2 — Fisher-information identifiability

**Location:** `identifiability/<model>/`

**Required files:**
- `compute_fim.py` — main computational script.
- `fisher_information_analysis.md` — write-up: method, scope, table
  of per-parameter scores, interpretation, fix proposals if any
  rank-deficient direction is found.
- `results/` — `fim_summary.json`, sensitivity heatmap PNG,
  eigenvalue spectrum PNG, per-parameter table CSV.

**Method.** Sensitivity-based linearised FIM via JAX autograd
through the deterministic ODE solve. The standard practical proxy
for the rigorous particle-filter FIM:

1. Pick a reference parameter point θ₀ in the healthy operating
   regime.
2. Forward-simulate the deterministic ODE for D days at θ₀.
3. Compute predicted observation-channel mean trajectories (the
   adapter's `clinician_plots.generate_observations` works).
4. Compute the sensitivity matrix `J[k, j] = ∂y_pred[k] / ∂θ_j` via
   `jax.jacfwd` through the diffrax solve.
5. FIM proxy: `F = Jᵀ Σ_obs⁻¹ J` with `Σ_obs` diagonal observation
   noise.
6. SVD of `J`; eigendecomp of `F`. Report rank, condition number,
   per-parameter scores.

**Multi-operating-point design.** Some parameters are unidentifiable
at a single reference (e.g. dampener parameters that only "switch
on" under pathology). The script must compute J at multiple
operating points and stack them. SWAT uses three: healthy
`(V_h=1, V_n=0)`, mild stress `(V_h=1, V_n=1)`, severe stress
`(V_h=1, V_n=3)`.

**Acceptance:**

- Full-rank FIM at the multi-operating-point analysis: `rank(F) = N`
  where N is the number of analysed parameters.
- Condition number `κ(F) = σ_max² / σ_min² < 10¹⁰`.
- For every parameter: pass / fail on individual identifiability,
  with the most-collinear partner reported for any near-degenerate
  ones.

If a rank-deficient direction is found, the write-up must propose a
reformulation (pin one of a degenerate pair, drop a redundant
parameter, etc.). **The CI exits non-zero on rank deficiency or
κ > 10¹⁰** — manual override is not provided.

**Runtime budget:** 10–15 minutes on CPU.

## Deliverable 3 — Lyapunov stability sweep

**Location:** `stability/<model>/`

**Required files:**
- `corner_case_sweep.py` — main computational script.
- `lyapunov_stability_analysis.md` — write-up: analytical Lyapunov
  function (e.g. Stuart-Landau potential), numerical results,
  acceptance summary.
- `results/` — `corner_summary.json`, T(t) trajectories PNG, phase-
  plane PNG, IC sweep PNG, corner bar chart PNG.

**Method.**

1. **Corner sweep.** Enumerate the 2ⁿ binary corners of the n-
   dimensional control box. For each corner: deterministic + ≥32
   stochastic trajectories over D days (≥30× the slowest model
   timescale). Record the amplitude variable's terminal value.
2. **Initial-condition sweep at the healthy corner.** ≥32 random
   ICs spanning the physical state space. Confirm all converge to
   the same attractor.
3. **Analytical Lyapunov function.** Construct the Stuart-Landau or
   equivalent potential, show its stationary points, show
   `dV/dt ≤ 0` along trajectories at the healthy corner.
4. **Phase-plane plot.** Project to the (bifurcation-parameter,
   amplitude) plane. Show all corner trajectories overlaid.

**Acceptance:**

- The healthy corner converges to T(D) ≈ T* (deterministic, within
  5% of the model's predicted equilibrium).
- All n-1 pathological corners give T(D) < 0.3 deterministic.
- The healthy corner's basin of attraction includes the entire
  physical state space (all ≥32 random ICs converge).
- The analytical Lyapunov function exists and `dV/dt ≤ 0` is
  verified along trajectories at the healthy corner.

**The script exits non-zero if any corner fails its acceptance
threshold.**

**Runtime budget:** 5–10 minutes on CPU (use `jax.jit` + `jax.vmap`
across particles).

## Deliverable 4 — Manifest entry + CI workflow

**Location:** `snapshots/manifest.json` and
`.github/workflows/<model>_validation.yml`.

**Manifest format** (see `snapshots/README.md`):

```json
{
  "schema_version": 2,
  "models": {
    "<model>": {
      "snapshots": [
        {
          "upstream_sha":  "<commit-from-Python-Model-Development-Simulation>",
          "captured_at":   "ISO-8601 UTC timestamp",
          "status":        "validated",
          "notes":         "free text"
        }
      ]
    }
  }
}
```

**CI workflow** (copy and adapt from
`.github/workflows/swat_validation.yml`):

- Three jobs: gating-tests, identifiability-and-stability,
  tag-validated.
- Triggers on changes to `src/model_validation/**`, `tests/<model>/**`,
  `identifiability/<model>/**`, `stability/<model>/**`,
  `pyproject.toml`, the workflow itself.
- Auto-tag step on `main`: `<model>-validated-<date>-<short-sha>`.

**Acceptance:** the workflow is green on a fresh push to `main`,
and the auto-tag fires.

## What the validator checks vs what the author must check

| Check | Mechanism |
|:---|:---|
| Tests pass | CI exit code from pytest |
| FIM rank == N, κ < 10¹⁰ | `compute_fim.py` exits non-zero on failure |
| Stability corners pass | `corner_case_sweep.py` exits non-zero on failure |
| Manifest is valid JSON | (manual review) |
| Workflow path filters cover the model | (manual review) |
| Author's interpretation in the write-up is sound | (manual review) |

The first three are mechanically enforced. The last three rely on
PR review.

## What the validator does NOT check

- **Whether the model is a faithful copy of upstream.** That's the
  vendoring author's responsibility; the FIM and Lyapunov analyses
  test what was actually vendored, not what was meant to be
  vendored.
- **Whether the gating-test suite is complete.** A model that
  passes 5 trivial tests is technically validated. Reviewers must
  cross-check that the suite covers the structural properties of
  the *model under study*, not just easy ones.
- **Whether the identifiability scope is correct.** The FIM script
  picks a parameter set (e.g. SWAT's deterministic 26 params,
  excluding diffusion temperatures). If the picked set is too
  narrow, key degeneracies may be missed. The write-up must
  defend the scope.

## Next

Read [`03_step_by_step_guide.md`](03_step_by_step_guide.md) for the
procedure end-to-end.

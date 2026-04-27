# 03 — Step-by-step guide

This is the procedure. Assumes you've read `01_overview.md` and
skimmed `02_validation_contract.md`.

The example throughout uses `your_model` as a placeholder.
Substitute your actual model name (lowercase, no spaces, e.g.
`fsa_high_res`).

## Step 1 — Vendor the model dynamics

Goal: a self-contained JAX implementation under
`src/model_validation/models/your_model/`.

### Layout

```
src/model_validation/models/your_model/
├── __init__.py
├── README_vendored.md        # provenance + conversion notes
├── vendored_dynamics.py      # drift + diffusion + state_clip
└── vendored_parameters.py    # default parameter dictionary
```

If you need to express a calibration variant (e.g. SWAT's "Option
C" fix for the V_h inversion), add a sibling file like
`option_c_dynamics.py` that exports a refined drift / parameter
helper, and toggle it via a pytest variant flag.

### What to vendor

- The drift function `f(t, x, u, params) -> jnp.ndarray` (in DAYS).
- The diffusion function `g(x, params) -> jnp.ndarray` (per-
  component sigmas).
- The state-clip helper (physical-bounds clip applied after each
  Euler-Maruyama step).
- The default parameter dictionary, with all timescales in DAYS.

### What NOT to vendor

- The observation model. Validation is a latent-dynamics gate;
  observation channels live in upstream filtering pipelines.
- Estimation routines (particle filter, SMC²).
- The scipy-NumPy simulator from upstream (only the JAX form).

### Time-unit conversion

If upstream uses hours: convert ONCE in `vendored_parameters.py`,
not inside the dynamics. SWAT's `vendored_parameters.py` does this:

```python
_HOURS_PER_DAY = 24.0
p['tau_W']  = 2.0 / _HOURS_PER_DAY        # hours -> days
p['T_W']    = 0.05 * _HOURS_PER_DAY       # per-hour -> per-day variance
```

### `README_vendored.md`

Document:
- Source repo, commit SHA, date of vendoring.
- What's NOT vendored and why.
- Time-unit conversions applied.
- Update procedure for re-vendoring later.

## Step 2 — Write the gating tests

Location: `tests/your_model/`. Layout:

```
tests/your_model/
├── __init__.py
├── conftest.py                  # fixtures + pytest CLI flags
├── test_<property>.py           # one file per structural property
└── ...
```

### `conftest.py` skeleton

```python
def pytest_addoption(parser):
    parser.addoption(
        "--variant", action="store", default="vendored",
        choices=["vendored", "option-c"],   # whatever your variants are
        help="Calibration variant to test against.",
    )

@pytest.fixture(scope="session")
def model(request):
    variant = request.config.getoption("--variant")
    if variant == "vendored":
        return your_model_vendored()
    elif variant == "option-c":
        return your_model_option_c()
    raise ValueError(f"Unknown variant: {variant}")
```

### Required test categories

See `02_validation_contract.md` for the minimum coverage list. One
file per category is conventional but not required; SWAT uses 9
files (`test_anabolicity.py`, `test_bifurcation.py`, etc.).

### Pre-fix snapshot tests

If you're vendoring a known-broken upstream snapshot as a regression
target, mark the failing tests with a pytest marker:

```python
@pytest.mark.expected_to_fail_pre_fix
def test_v_h_anabolicity_under_option_c(model_option_c):
    ...
```

Register the marker in `pyproject.toml` so pytest doesn't warn:

```toml
[tool.pytest.ini_options]
markers = [
    "expected_to_fail_pre_fix: ...",
]
```

## Step 3 — Port the FIM identifiability analysis

Location: `identifiability/your_model/`.

### Layout

```
identifiability/your_model/
├── compute_fim.py
├── fisher_information_analysis.md
└── results/                  # populated when compute_fim.py runs
```

### `compute_fim.py` skeleton

Use SWAT's `identifiability/swat/compute_fim.py` as the reference.
Key pieces to adapt:

1. The parameter list: the union of the latent + observation params
   you want analysed. For your model, list them at the top of the
   file as a constant tuple.
2. The reference operating points: at minimum one healthy point,
   plus pathology points if the model has dampener / threshold
   parameters that switch on under pathology.
3. The forward simulation: use `runner._make_ode_solve` from this
   package — it handles the JAX/diffrax wiring.
4. The exit condition: `sys.exit(1)` if `rank < N` or `κ > 1e10`.

### `fisher_information_analysis.md` skeleton

Sections:
- Method (1-page summary of FIM via sensitivity Jacobian).
- Scope (which parameters are analysed; defend the scope).
- Results (rank, κ, per-parameter table).
- Degeneracies and resolutions (if any).
- Inference recommendations (priors, pinned parameters,
  hierarchical SMC² assignments — useful for downstream filtering).

## Step 4 — Port the Lyapunov stability sweep

Location: `stability/your_model/`.

### Layout

```
stability/your_model/
├── corner_case_sweep.py
├── lyapunov_stability_analysis.md
└── results/                  # populated when corner_case_sweep.py runs
```

### `corner_case_sweep.py` skeleton

Use SWAT's `stability/swat/corner_case_sweep.py` as the reference.
Key pieces to adapt:

1. Identify the n control dimensions and their healthy / pathology
   binary values. For SWAT this is `(V_h, V_n, V_c)` ∈ `{0,1} ×
   {0,5} × {0,6}` (8 corners).
2. The horizon must be ≥30× the slowest model timescale (D ≥ 30 ×
   τ_slowest). SWAT uses 60 days against `τ_T = 2 days`.
3. JIT + vmap the stochastic sweep across particles. Use
   `jax.random.split` for trajectory-independent noise.
4. Initial-condition sweep at the healthy corner: ≥32 random ICs.
5. The exit condition: `sys.exit(1)` if any corner fails its
   acceptance threshold.

### `lyapunov_stability_analysis.md` skeleton

Sections:
- TL;DR (one paragraph: "X corner uniquely dominates").
- Method (corner enumeration, deterministic + stochastic).
- Results (corner table, T(t) plot, phase plane).
- IC sweep at the healthy corner.
- Analytical Lyapunov function (Stuart-Landau potential or
  equivalent; show `dV/dt ≤ 0`).
- Implications for OT-Control (if relevant).
- Reproducibility (`python stability/your_model/corner_case_sweep.py`).

## Step 5 — Register in the manifest

Edit `snapshots/manifest.json`. Append a new top-level entry under
`models`:

```json
{
  "schema_version": 2,
  "models": {
    "swat":      { "snapshots": [...] },
    "your_model": {
      "snapshots": [
        {
          "upstream_sha":  "<commit-from-upstream-model-dev-repo>",
          "captured_at":   "<ISO-8601 UTC>",
          "status":        "pre-fix",
          "notes":         "Initial vendoring; <known issue> still expected to fail."
        }
      ]
    }
  }
}
```

After CI passes the first time on `main`, append a second entry
with `"status": "validated"` and the new commit's SHA.

## Step 6 — Add the CI workflow

Copy `.github/workflows/swat_validation.yml` to
`.github/workflows/your_model_validation.yml` and substitute paths:

- `name:` field → `your_model validation`
- Path filters → `tests/your_model/**`,
  `identifiability/your_model/**`, `stability/your_model/**`,
  `.github/workflows/your_model_validation.yml`
- Job 1 pytest invocation: `pytest tests/your_model/
  --variant <canonical>`
- Job 2 script paths: `identifiability/your_model/compute_fim.py`
  and `stability/your_model/corner_case_sweep.py`
- Job 2 artefact upload paths similarly.
- Job 3 tag prefix: `your_model-validated-`.

## Step 7 — Optional: scripts and clinician-view docs

If you have ad-hoc calibration sweep scripts or scenario-comparison
plots, put them under `scripts/your_model/` and `docs/your_model/`
respectively. SWAT has:

- `scripts/swat/option_c_lambda_sweep.py` — λ-sensitivity sweep.
- `scripts/swat/option_c_heatmap.py` — V_h × V_n grid comparison.
- `scripts/swat/clinician_scenario_plots.py` — 4 canonical-scenario
  plots.
- `docs/swat/option_c_results/README.md` — companion to OT-Control
  issue #4.
- `docs/swat/clinician_views/{option_c_calibrated,pre_fix_vendored}/`
  — 4 scenario × 3 panel plots.

These are not mandatory, but they're the natural home for "I
needed to convince myself the calibration is right" outputs.

## Step 8 — Run it locally first

Before pushing:

```bash
pip install -e .[dev]

pytest tests/your_model/ --variant <canonical> -v
python identifiability/your_model/compute_fim.py
python stability/your_model/corner_case_sweep.py
```

All three must exit 0. Read the write-ups end-to-end. Inspect the
plots — anything counterintuitive in the corner T(t) trajectories
or the FIM eigenvalue spectrum is a red flag worth investigating
before you push.

## Step 9 — Open the PR

Title: `feat: validation entry for <your_model>`.

PR body should include:
- Link to the upstream commit SHA being vendored.
- Summary of gating-test coverage.
- FIM rank, κ, and any pinned-parameter decisions.
- Lyapunov corner summary (T(D) for each corner).
- Any open issues / known limitations.

Ask one reviewer to sanity-check the scope of the FIM and the
choice of corners for the Lyapunov sweep.

## Step 10 — Re-vendoring later

When upstream changes the model:

1. Re-vendor `vendored_dynamics.py` / `vendored_parameters.py` from
   the new upstream commit.
2. Bump `README_vendored.md` with the new SHA + date.
3. Append a new entry to `snapshots/manifest.json` under your
   model's `snapshots`. Set status to `pre-fix` until CI re-tags.
4. Re-run the gating tests + FIM + Lyapunov locally. Fix any
   regressions.
5. Open a PR. CI re-runs. On green merge, the auto-tag step adds
   a fresh `your_model-validated-*` tag — OT-Control's vendor-
   sync sees it and may bump.

## Next

Read [`04_worked_example_SWAT.md`](04_worked_example_SWAT.md) for
how this looked in practice for SWAT, including the V_h inversion
fix that motivated the whole repo.

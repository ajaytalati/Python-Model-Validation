# 01 — Overview

## What this repo is

Python-Model-Validation is a **gate**, not a model library.

```
┌────────────────────────────────────────┐
│  Python-Model-Development-Simulation   │   models are defined here
│  (drift, diffusion, observation        │   — their first home
│   model, parameters, spec docs)        │
└──────────────┬─────────────────────────┘
               │ snapshot of upstream commit
               ▼
┌────────────────────────────────────────┐
│  Python-Model-Validation  ← THIS REPO  │   models are validated here
│  (gating tests, FIM identifiability,   │   before vendoring downstream
│   Lyapunov stability, manifest, CI)    │
└──────────────┬─────────────────────────┘
               │ "<model>-validated-<date>-<sha>" tag
               ▼
┌────────────────────────────────────────┐
│  Python-Model-OT-Control               │   validated models are
│  (vendored copy + adapter +            │   deployed under optimal
│   optimal-control engine)              │   control here
└────────────────────────────────────────┘
```

Each arrow is a **one-way dependency**. This repo never imports
from OT-Control. OT-Control's vendor-sync script polls *this* repo
to decide what is safe to vendor. A model becomes "vendor-safe" the
moment its CI workflow here tags a commit `<model>-validated-*`.

## What a model entry consists of

Four deliverables. All four must pass for the model to be tagged
validated:

1. **Gating tests** (`tests/<model>/`). Pytest suite covering
   monotonicity, dose-response, equilibrium, bifurcation, sign,
   anabolicity/catabolicity. These are the structural-correctness
   tests. ~5 min on CPU.
2. **Fisher-information identifiability** (`identifiability/<model>/`).
   FIM rank, condition number, per-parameter scores. Catches
   structural redundancy in the parameter set. ~10–15 min on CPU.
3. **Lyapunov stability sweep** (`stability/<model>/`). Corner-case
   sweep over the control box, deterministic + stochastic. Confirms
   the model has a unique super-critical attractor at the healthy
   corner. ~5–10 min on CPU.
4. **Manifest entry** (`snapshots/manifest.json` + a CI workflow at
   `.github/workflows/<model>_validation.yml`). Registry of
   upstream commit hashes plus their validation status, plus the
   automation that re-runs the three analyses on every change.

## Why this layering matters

Three reasons.

**1. Reproducibility.** Vendoring an upstream snapshot means a model
entry can be checked out at any historical commit and still reproduce
the validation results, even if upstream has moved on. The snapshot's
`upstream_sha` records the provenance.

**2. Fault isolation.** A bug in OT-Control's optimiser is fixed in
OT-Control. A bug in a model's dynamics is fixed upstream and re-
vendored. The validation pipeline is the boundary between the two.

**3. Engine-side guarantees survive.** When OT-Control's vendor-sync
script blocks an upstream commit lacking a `<model>-validated-*` tag,
that block applies to *every* downstream vendor — so structural
regressions in a model never silently escape into production
optimisation.

## What's already model-agnostic

The package at `src/model_validation/` is model-agnostic
infrastructure:

- `runner.py` — generic ODE/SDE harness (`_make_ode_solve`,
  `_make_sde_solve`, `ModelInterface`,
  `t_end_under_constant_controls`).
- `clinician_plots.py` — generic plot suite, parameterised by a
  `ModelInterface` instance.
- `snapshot.py` — vendoring helper.

Per-model code goes under `models/<model>/`. The harness never
imports from a specific `models/<model>/` subpackage; it takes a
`ModelInterface` instance from the caller.

## What's NOT in scope

- **The optimal-control loss / optimiser.** That lives in
  OT-Control. Validation only checks the *model*, not how the
  model is used downstream.
- **Observation-model parameters.** This repo validates the latent
  dynamics. Observation-channel parameters (HR, sleep, stress,
  steps, etc.) are an estimation concern handled in upstream
  filtering pipelines.
- **Estimation routines** (particle filters, SMC², Gaussian-kernel
  filters). Those live upstream.
- **Model variants** at the conceptual level. SWAT v3 vs SWAT v4 is
  an upstream decision; this repo only validates the *current*
  vendored snapshot.

## Sizing

A typical model entry is:

- ~300 lines of vendored dynamics + parameters
- ~500 lines of pytest gating suite (10–20 tests)
- ~400 lines of FIM script + ~300-line write-up
- ~500 lines of Lyapunov sweep + ~300-line write-up
- ~150 lines of CI workflow
- Total: ~2200 lines across 8–12 files

SWAT (the canonical worked example, see `04_worked_example_SWAT.md`)
took ~3 weeks of work end-to-end including resolving the V_h
inversion bug (issue #4 on OT-Control) and the V_n bell-shape
problem (issue #5).

## Next

Read [`02_validation_contract.md`](02_validation_contract.md) for
the precise acceptance criteria of each of the four deliverables.

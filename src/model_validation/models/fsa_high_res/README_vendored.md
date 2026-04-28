# Vendored FSA-High-Res Model

**Source:** `Python-Model-OT-Control/version_1/_vendored_models/fsa_high_res/` (which is itself a vendoring of `Python-Model-Development-Simulation/version_1/models/fsa_high_res/`)
**Vendored on:** 27 April 2026
**Source commit:** OT-Control PR #7 (`feature/fsa-adapter` branch, v1.2.1).
**Purpose:** Self-contained JAX-native re-implementation of the FSA-high-res SDE for use by the validation pipeline (gating tests + FIM + Lyapunov sweep).

## What's here

- `vendored_dynamics.py` (renamed from upstream `dynamics_jax.py`) — JAX drift, diffusion, state-clip, amplitude projector, healthy-attractor predicate for the three-state $(B, F, A)$ SDE.
- `vendored_parameters.py` (renamed from upstream `parameters.py`) — the 13-parameter dynamics dictionary tuned for the 14-day proof-of-principle horizon.

## Naming convention

This repo's convention is `vendored_*.py` (matches SWAT in `src/model_validation/models/swat/`). The upstream OT-Control vendoring uses unprefixed names (`dynamics_jax.py`, `parameters.py`); the rename is cosmetic — function names inside (`fsa_drift`, `fsa_diffusion`, `default_fsa_parameters`, etc.) are unchanged.

## Implementation notes

The deployed `mu_0 = +0.02` keeps `μ(B, F)` super-critical across the 14-day POC horizon, putting A near the Stuart-Landau fixed point sqrt(μ/η) of roughly 0.7 mid-run.

The diffusion is regularised eps-under-sqrt (rather than upstream's clip-then-sqrt) so JAX-AD gradients stay finite at the boundary. Mathematically equivalent for x well inside (0, 1); structurally important for the FIM Jacobian computation.

The control signature is scalar `(T_B, Phi)` per timestep — upstream takes per-bin arrays (96 bins/day) for sub-daily HR / sleep observations. The validation pipeline uses daily piecewise-constant controls, matching OT-Control's piecewise-constant policy.

## What's NOT here (and why)

- The upstream observation model (HR, sleep, stress, steps, 19 obs parameters). Validation gates the latent dynamics; observation channels are an estimation concern.
- The 15-min bin lookup machinery for time-varying T_B(t), Phi(t).
- The scipy-NumPy simulator from upstream.
- Estimation routines.

## Update procedure

If OT-Control's FSA vendoring changes (e.g. when Python-Model-Development-Simulation re-pushes the spec):

1. Re-copy `dynamics_jax.py` → `vendored_dynamics.py` and `parameters.py` → `vendored_parameters.py`.
2. Bump this README with the new commit + date.
3. Append a new entry to `snapshots/manifest.json` under `models.fsa_high_res.snapshots`. Set status `pre-fix` until CI re-tags.
4. Re-run gating tests + FIM + Lyapunov locally. Fix any regressions.

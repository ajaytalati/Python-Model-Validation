# Python-Model-Validation

Rigorous mathematical and clinical-validity gating for physiological models, sitting between the model-definition repo ([Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)) and the optimal-control repo ([Python-Model-OT-Control](https://github.com/ajaytalati/Python-Model-OT-Control)).

**Why this repo exists.** The model-definition repo contains the model SDE, parameters, and ad-hoc verification scripts. It does *not* check that each control variable behaves the way its clinical interpretation says it should. When the model is vendored downstream into a control-optimisation framework, those controls are promoted from per-subject *constants* to *time-varying optimisable controls* — and the optimiser will exploit any clinically-inverted gradient. See [Python-Model-OT-Control issue #4](https://github.com/ajaytalati/Python-Model-OT-Control/issues/4) for the case study that motivated this library.

This library applies a battery of monotonicity and clinical-scenario tests to every upstream snapshot. Vendoring downstream is only allowed from snapshots that pass.

## Layout

The repo is organised by **per-model** directories so a new model
(FSA-high-res, in flight) drops in alongside SWAT without overlap.

```
how_to_add_a_new_validation_model/   guide for landing a new model (5 docs)
src/model_validation/                model-agnostic harness
├── runner.py                       generic ODE/SDE solver helpers
├── snapshot.py                     vendoring helper
├── clinician_plots.py              generic plot suite
└── models/<model>/                 per-model vendored dynamics + params
tests/<model>/                      per-model pytest gating suite
identifiability/<model>/            per-model Fisher-info analysis
stability/<model>/                  per-model Lyapunov sweep
scripts/<model>/                    per-model ad-hoc calibration scripts (optional)
docs/<model>/                       per-model clinician views + write-ups (optional)
snapshots/manifest.json             multi-model snapshot registry (schema v2)
.github/workflows/<model>_validation.yml   per-model CI
```

Currently populated for `<model> = swat`. Adding a new model is purely
additive — see [`how_to_add_a_new_validation_model/`](how_to_add_a_new_validation_model/).

## Vendoring workflow

1. **Snapshot upstream:** copy the model's drift / diffusion / parameters
   from [Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)
   into `src/model_validation/models/<model>/vendored_*.py` and append
   an entry under `models.<model>.snapshots` in `snapshots/manifest.json`
   (status `pre-fix` until validated).
2. **Validate:** `pytest tests/<model>/` + `python identifiability/<model>/compute_fim.py`
   + `python stability/<model>/corner_case_sweep.py`. All must exit 0.
   CI does this automatically on push / PR.
3. **Tag:** when CI passes on `main`, the workflow auto-tags the commit
   `<model>-validated-<date>-<short-sha>`.
4. **Consume:** `Python-Model-OT-Control`'s vendor-sync script reads
   this repo's `snapshots/manifest.json` and the tag list; it only
   bumps `_vendored_models/<model>/` from a commit marked `validated`.

## Running locally

```bash
git clone https://github.com/ajaytalati/Python-Model-Validation.git
cd Python-Model-Validation
pip install -e .[dev]

# Run SWAT gating tests against the current snapshot:
pytest tests/swat/

# Run SWAT gating tests against the Option C variant:
pytest --variant option-c tests/swat/

# Run SWAT identifiability + stability:
python identifiability/swat/compute_fim.py
python stability/swat/corner_case_sweep.py

# Run the lambda-sensitivity sweep:
python scripts/swat/option_c_lambda_sweep.py
```

## License

MIT.

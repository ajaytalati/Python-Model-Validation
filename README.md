# Python-Model-Validation

Rigorous mathematical and clinical-validity gating for physiological models, sitting between the model-definition repo ([Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)) and the optimal-control repo ([Python-Model-OT-Control](https://github.com/ajaytalati/Python-Model-OT-Control)).

**Why this repo exists.** The model-definition repo contains the model SDE, parameters, and ad-hoc verification scripts. It does *not* check that each control variable behaves the way its clinical interpretation says it should. When the model is vendored downstream into a control-optimisation framework, those controls are promoted from per-subject *constants* to *time-varying optimisable controls* — and the optimiser will exploit any clinically-inverted gradient. See [Python-Model-OT-Control issue #4](https://github.com/ajaytalati/Python-Model-OT-Control/issues/4) for the case study that motivated this library.

This library applies a battery of monotonicity and clinical-scenario tests to every upstream snapshot. Vendoring downstream is only allowed from snapshots that pass.

## Layout

```
src/model_validation/        # the package
├── runner.py               # t_end_under_constant_controls — central fixture
├── snapshot.py             # pulls upstream model code, writes manifest
└── models/swat/            # currently SWAT only (one model per subdir)
    ├── vendored_dynamics.py
    └── vendored_parameters.py

tests/swat/                  # 9 gating tests, ported from
                             # Python-Model-OT-Control's upstream_gating_tests.md
snapshots/manifest.json      # commit-hash → tests-passed status
docs/                        # vendoring workflow, how to add a test
```

## Vendoring workflow

1. Snapshot upstream: `python -m model_validation.snapshot --upstream <commit-sha>` copies `models/swat/_dynamics.py`, `simulation.py::PARAM_SET_*`, `INIT_STATE_*` into `vendored_*.py` and records the commit hash in `snapshots/manifest.json`.
2. Validate: `pytest tests/`. CI does this automatically.
3. Tag: passing snapshots are auto-tagged `validation-passed-<upstream-sha>-<date>`.
4. Consume: `Python-Model-OT-Control`'s vendor-sync script reads this repo's `manifest.json` and refuses to bump its `_vendored_models/swat/` unless the source commit is marked passed.

## Running locally

```bash
git clone https://github.com/ajaytalati/Python-Model-Validation.git
cd Python-Model-Validation
pip install -e .[dev]

# Run all tests against the current snapshot:
pytest tests/swat/

# Run against the Option C variant:
pytest --variant option-c tests/swat/

# Run the lambda-sensitivity sweep:
python scripts/swat/option_c_lambda_sweep.py
```

## License

MIT.

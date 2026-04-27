# How to add a new model to Python-Model-Validation

This folder explains how to land a new physiological model in the
validation pipeline. The pipeline sits between
[Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)
(where models are *defined*) and
[Python-Model-OT-Control](https://github.com/ajaytalati/Python-Model-OT-Control)
(where validated models are *deployed* under optimal control). This
repo is the **gate**: a model is not safe to vendor downstream until
it has passed the gating tests, the Fisher-information identifiability
analysis, and the Lyapunov stability sweep documented here.

## Read these in order

1. **[`01_overview.md`](01_overview.md)** — what this repo is, the
   dependency direction, what a model entry consists of. 5-minute
   read.
2. **[`02_validation_contract.md`](02_validation_contract.md)** — the
   four deliverables every model must produce, the acceptance
   thresholds for each, and the tag that gets emitted on pass.
   Reference document.
3. **[`03_step_by_step_guide.md`](03_step_by_step_guide.md)** — the
   procedure end-to-end: vendor the model code, write the gating
   tests, port the FIM analysis, port the Lyapunov sweep, register
   in the manifest, copy the CI workflow.
4. **[`04_worked_example_SWAT.md`](04_worked_example_SWAT.md)** — a
   tour of how SWAT was done. Read this for concrete examples of
   what each deliverable actually looks like.

## What a model entry looks like

Once a model is fully landed, its footprint is:

```
src/model_validation/models/<model>/      vendored dynamics + parameters
tests/<model>/                             pytest gating suite
identifiability/<model>/                   FIM analysis (script + write-up + results)
stability/<model>/                         Lyapunov sweep (script + write-up + results)
scripts/<model>/                           ad-hoc calibration scripts (optional)
docs/<model>/                              clinician-view plots, fix-plan write-ups (optional)
snapshots/manifest.json                    one entry under "models.<model>.snapshots"
.github/workflows/<model>_validation.yml   CI workflow
```

Adding a new model is purely additive — every directory above is
keyed by `<model>`, so SWAT and FSA-high-res sit alongside each
other without overlap.

## Common pitfalls (skim before starting)

* **Time-unit consistency.** Validation uses **days** throughout
  (matches OT-Control). If your upstream uses hours, do the
  conversion once at vendoring time, not inside the dynamics.
* **Fix the model, don't soften the test.** When a gating test
  fails, fix the dynamics — don't lower the threshold or skip the
  test. A passing-by-relaxation entry is worse than no entry,
  because OT-Control will vendor it as if validated.
* **Identifiability before stability.** If FIM exposes a non-
  identifiable direction, the model needs reformulation before a
  Lyapunov argument is meaningful. Run identifiability first,
  resolve any rank-deficiency, then run stability.
* **Acceptance is binary.** Each of the four deliverables either
  PASSES or FAILS; the CI workflow's auto-tag step only fires when
  all four pass on `main`. Manual override is not provided by
  design.

## When you're done

Your model entry is complete when:

1. `pytest tests/<model>/ --variant <your-variant>` — green.
2. `python identifiability/<model>/compute_fim.py` — exit 0.
3. `python stability/<model>/corner_case_sweep.py` — exit 0.
4. `snapshots/manifest.json` has a top-level entry under
   `models.<model>` with at least one snapshot marked `validated`.
5. `.github/workflows/<model>_validation.yml` is green on a fresh
   push to `main`, and the auto-tag step fires
   (`<model>-validated-<date>-<sha>`).

## See also

- The OT-Control how-to (`how_to_add_a_new_model_adapter/` in the
  downstream repo) explains how a *validated* model gets adapted
  into the optimal-control engine. Validation comes first; adapter
  comes second.

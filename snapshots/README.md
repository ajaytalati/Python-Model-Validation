# Snapshot manifest

`manifest.json` is a registry mapping each vendored model to the upstream commit hashes captured for it, with the validation outcome of each snapshot.

OT-Control's vendor-sync script (when it lands) reads this file to decide whether a candidate upstream commit is safe to vendor.

## Schema (v2 — multi-model)

```json
{
  "schema_version": 2,
  "models": {
    "<model-name>": {
      "snapshots": [
        {
          "upstream_sha":  "...",                   // commit hash from upstream model-dev repo
          "captured_at":   "ISO-8601 UTC timestamp",
          "status":        "pre-fix" | "validated" | "deprecated",
          "notes":         "free text"
        }
      ]
    }
  }
}
```

`status` values:

- `pre-fix` — captured pre-validation; expected to fail one or more gating tests. Useful as regression target.
- `validated` — passed gating + identifiability + Lyapunov stability. Vendoring approved.
- `deprecated` — superseded by a later snapshot; do not vendor.

## v1 → v2 schema migration

The original v1 schema was flat (`{"snapshots": [...]}` at the top level), implicitly single-model and assumed SWAT. v2 is keyed by model name to support FSA-high-res and any future models alongside SWAT, without restructuring.

When adding a new model, append a new top-level entry to `models`. Don't reuse model names.

## Auto-generation

The CI workflow `.github/workflows/<model>_validation.yml` produces tags of the form `<model>-validated-<date>-<sha>` when the validation pipeline passes on `main`. A future cron job will append matching `validated` entries to this file; right now updates are by hand.

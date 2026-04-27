"""snapshot.py — pull upstream model code, write manifest.

Currently a stub. The full implementation will:
  1. Take an upstream commit SHA from CLI.
  2. git-pull the specified commit from
     https://github.com/ajaytalati/Python-Model-Development-Simulation
     into a temp directory.
  3. Copy the SWAT model files (_dynamics.py, simulation.py PARAM_SET_*,
     INIT_STATE_*) into src/model_validation/models/swat/vendored_*.py,
     transforming hours -> days for time units.
  4. Append an entry to snapshots/manifest.json with the commit SHA,
     timestamp, and "status": "pending".
  5. After tests run (in CI), the manifest entry is updated to
     "status": "passed" or "status": "failed".

For now, the initial snapshot is hand-copied from the OT-Control repo
at version_1/_vendored_models/swat/. See the pre-fix commit for the
state used to demonstrate the validation framework.
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "snapshots" / "manifest.json"


def append_manifest_entry(upstream_sha: str, status: str = "pending"):
    """Append a snapshot entry to manifest.json."""
    if MANIFEST.exists():
        data = json.loads(MANIFEST.read_text())
    else:
        data = {"snapshots": []}
    data["snapshots"].append({
        "upstream_sha": upstream_sha,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
    })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True,
                    help="Upstream commit SHA to snapshot")
    args = p.parse_args()
    print(f"[snapshot] (stub) recording manifest entry for {args.upstream}")
    append_manifest_entry(args.upstream)
    print(f"[snapshot] full vendoring not implemented yet; "
          "current SWAT files were hand-copied from OT-Control v1.1.0.")


if __name__ == "__main__":
    main()

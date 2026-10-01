#!/usr/bin/env python
"""Freeze the chosen baseline plans for all later phases.

Copies signals/optimised.add.xml -> signals/frozen_baseline.add.xml and
signals/actuated.add.xml -> signals/frozen_actuated.add.xml (program ids
renamed), writes signals/README.md and a SHA-256 manifest (signals/FROZEN.json)
that tests/test_frozen.py checks.

Usage:
    python scripts/freeze_signals.py [--force]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re

import _bootstrap  # noqa: F401

from testbed.signals import SIGNALS_CONFIG, load_signals_config, plan_file, signals_dir

FROZEN = {"frozen_baseline": "optimised", "frozen_actuated": "actuated"}
NOTE = "These plans are frozen and must not be re-optimised in later phases."


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signals-config", default=str(SIGNALS_CONFIG))
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen set")
    args = ap.parse_args()
    scfg = load_signals_config(args.signals_config)
    d = signals_dir(scfg)
    manifest_path = d / "FROZEN.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} exists - plans are already frozen (use --force deliberately).")
    manifest = {"note": NOTE, "frozen_on": dt.date.today().isoformat(), "files": {}}
    for frozen, src in FROZEN.items():
        s = plan_file(scfg, src)
        if not s.exists():
            raise SystemExit(f"{s} missing - build it first.")
        txt = s.read_text()
        txt = re.sub(r'programID="[^"]+"', f'programID="{frozen}"', txt)
        txt = txt.replace("<additional>", f"<!-- FROZEN ({manifest['frozen_on']}), copied from {s.name}. "
                                          f"{NOTE} -->\n<additional>", 1)
        out = plan_file(scfg, frozen)
        out.write_text(txt)
        manifest["files"][out.name] = {"source": s.name, "sha256": sha256(out)}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    opt = json.loads((d / "optimised.json").read_text())
    (d / "README.md").write_text(f"""# Signal plans (grid3x3)

**{NOTE}**

| File | Plan | Notes |
|---|---|---|
| `frozen_baseline.add.xml` | fixed-time, optimised (CMA-ES on D1 `normal_peak`, training seeds {opt['info'].get('train_seeds')}) | common cycle {opt['cycle']} s, offsets per junction |
| `frozen_actuated.add.xml` | SUMO `type="actuated"`, gap-based extension on the E1 stop-line loops | same phase sequence, min/max green from `configs/signals.yaml` |
| `default.add.xml` | SUMO/netgenerate default (90 s, offsets 0) | reference only |
| `webster.add.xml` | Webster from E1 flows under the default plan | reference only |
| `optimised.add.xml`, `actuated.add.xml` | working copies the frozen files were made from | do not edit |

Integrity: `FROZEN.json` stores SHA-256 hashes; `tests/test_frozen.py` fails if a frozen file changes.
Frozen on {manifest['frozen_on']}.
""")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

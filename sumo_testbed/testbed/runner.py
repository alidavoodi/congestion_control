"""Run one simulation end to end and write its metrics.

Insertion runs until ``insertion_end``; afterwards the simulation continues
until the network is empty, capped at ``cooldown_max_s``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from testbed.metrics import (MetricsCollector, statistics_metrics, tripinfo_metrics,
                             write_metrics)
from testbed.network import load_mapping
from testbed.sim import SimSpec, SumoSim, TeleportError

log = logging.getLogger(__name__)


@dataclass
class RunSpec:
    """A simulation plus its evaluation window."""

    sim: SimSpec
    window: tuple[float, float]
    insertion_end: float
    mapping_file: Path
    label: dict[str, Any]  # identifying info copied into metrics.json


def run(spec: RunSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute a run, write ``metrics.json`` + time series, and return the metrics.

    Raises:
        TeleportError: if any vehicle teleported (metrics are written first).
    """
    # TODO(Phase 2-4): accept a BaseController and call observe/act here.
    out = spec.sim.out_dir
    out.mkdir(parents=True, exist_ok=True)
    e2 = [r for r in load_mapping(spec.mapping_file) if r["det_type"] == "E2"]
    cap = spec.insertion_end + float(cfg["demand"]["cooldown_max_s"])
    sim = SumoSim(spec.sim, cfg)
    sim.start()
    try:
        coll = MetricsCollector(sim, cfg, e2, spec.window)
        while True:
            sim.step()
            coll.record()
            t = sim.time
            if t >= spec.insertion_end and sim.remaining_demand() == 0:
                break
            if t >= cap:
                break
        m = coll.finalize(out)
    finally:
        sim.close(check_teleports=False)
    m |= tripinfo_metrics(out / "tripinfo.xml", spec.window)
    m |= statistics_metrics(out / "statistics.xml")
    m["emptied"] = m["unserved_total"] == 0
    metrics = dict(spec.label) | {"window": list(spec.window), "seed": spec.sim.seed} | m
    write_metrics(metrics, out)
    if m["teleports"] != 0:
        raise TeleportError(f"{m['teleports']} teleports in {out}")
    return metrics

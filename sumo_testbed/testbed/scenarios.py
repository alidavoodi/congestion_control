"""Demand scenarios: metadata on disk and construction of run specifications.

Each scenario folder ``demand/<scenario>/`` holds ``seed_<k>.rou.xml`` files
and a ``meta.yaml`` (level, multiplier, profile, analysis window, seeds and any
drawn random values). A run of scenario seed k always uses SUMO ``--seed k``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from testbed.config import network_files
from testbed.demand import demand_dir
from testbed.runner import RunSpec
from testbed.sim import SimSpec


def scenario_dir(cfg: dict[str, Any], scenario: str) -> Path:
    """Folder of a scenario's route files."""
    return demand_dir(cfg) / scenario


def route_file(cfg: dict[str, Any], scenario: str, seed: int) -> Path:
    """Route file of (scenario, seed)."""
    return scenario_dir(cfg, scenario) / f"seed_{seed}.rou.xml"


def load_meta(cfg: dict[str, Any], scenario: str) -> dict[str, Any]:
    """Read ``meta.yaml`` of a scenario."""
    p = scenario_dir(cfg, scenario) / "meta.yaml"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run scripts/build_demand.py first.")
    return yaml.safe_load(p.read_text())


def make_runspec(cfg: dict[str, Any], scenario: str, seed: int, tls_file: Path | None,
                 out_dir: Path, label: dict[str, Any] | None = None) -> RunSpec:
    """Run specification for one scenario seed under a given signal program file."""
    meta = load_meta(cfg, scenario)
    net = meta["network"]
    f = network_files(cfg, net)
    sim = SimSpec(f["net"], route_file(cfg, scenario, seed), f["det"], tls_file, out_dir, seed,
                  step_length=cfg["sumo"]["step_length"], use_libsumo=cfg["sumo"]["use_libsumo"])
    return RunSpec(sim, tuple(meta["analysis_window"]), float(meta["insertion_end"]), f["mapping"],
                   {"network": net, "scenario": scenario} | (label or {}))

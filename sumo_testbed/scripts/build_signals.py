#!/usr/bin/env python
"""Build signal plans for grid3x3 with the network's (frozen) phase sequence.

Steps (run in order; each can be run alone):
  default     export the network's built-in program to signals/default.add.xml
  saturation  measure saturation flow from E1 discharge headways (default plan)
  webster     Webster plan from E1 flows measured under the default plan on D1
  actuated    SUMO actuated plan using the E1 loops (needs signals/optimised.json,
              falls back to webster.json as the green-time guide)

Usage:
    python scripts/build_signals.py [default saturation webster actuated]
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp

import _bootstrap  # noqa: F401
import numpy as np
import yaml

from testbed.config import load_config, network_files
from testbed.demand import constant_demand
from testbed.log import setup_logging
from testbed.runner import run
from testbed.scenarios import make_runspec
from testbed.signals import (SIGNALS_CONFIG, Plan, default_plan, e1_flows, load_signals_config,
                             measure_saturation, phase_structure, plan_file, signals_dir,
                             webster_plan, write_plan)
from testbed.sim import SimSpec, SumoSim
from testbed.config import PROJECT_ROOT

log = logging.getLogger("build_signals")


def _measure_run(args):
    cfg, scfg, seed = args
    w = scfg["webster"]
    out = PROJECT_ROOT / scfg["results_dir"] / "webster_measurement" / f"seed_{seed}"
    setup_logging(out / "run.log")
    run(make_runspec(cfg, w["design_scenario"], seed, None, out, {"plan": "default"}), cfg)
    return seed, e1_flows(out / "e1.xml", tuple(w["flow_window"]))


def step_default(cfg, scfg, struct, files) -> None:
    plan = default_plan(files["net"], struct)
    write_plan(plan, struct, plan_file(scfg, "default"), scfg, note="Exported from the network.")
    plan.to_json(signals_dir(scfg) / "default.json")
    log.info("default: cycle %d s, greens B1 %s -> %s", plan.cycle, plan.greens["B1"], plan_file(scfg, "default"))


def step_saturation(cfg, scfg, struct, files) -> float:
    m = scfg["webster"]["saturation_measurement"]
    out = PROJECT_ROOT / scfg["results_dir"] / "saturation"
    spec = SimSpec(files["net"], constant_demand(cfg, float(m["multiplier"])), files["det"], None,
                   out, int(m["seed"]), step_length=cfg["sumo"]["step_length"], use_libsumo=True)
    sim = SumoSim(spec, cfg)
    sim.start()
    try:
        res = measure_saturation(sim, struct, files["mapping"], m, float(m["window"][1]))
    finally:
        sim.close()
    (out / "saturation.json").write_text(json.dumps(res, indent=2))
    log.info("saturation flow: %.0f veh/h/lane (mean headway %.2f s, n=%d; lane0 %.0f, lane1 %.0f)",
             res["saturation_flow_vphpl"], res["mean_headway_s"], res["n_headways"],
             res["per_lane_index_vphpl"].get("0", float("nan")),
             res["per_lane_index_vphpl"].get("1", float("nan")))
    return res["saturation_flow_vphpl"]


def saturation_flow(cfg, scfg, struct, files) -> float:
    v = scfg["webster"]["saturation_flow_vphpl"]
    if v is not None:
        return float(v)
    p = PROJECT_ROOT / scfg["results_dir"] / "saturation" / "saturation.json"
    if p.exists():
        return float(json.loads(p.read_text())["saturation_flow_vphpl"])
    return step_saturation(cfg, scfg, struct, files)


def step_webster(cfg, scfg, struct, files) -> None:
    s = saturation_flow(cfg, scfg, struct, files)
    seeds = scfg["webster"]["seeds"]
    with mp.get_context("spawn").Pool(min(len(seeds), 4)) as pool:
        res = dict(pool.map(_measure_run, [(cfg, scfg, k) for k in seeds]))
    setup_logging(PROJECT_ROOT / cfg["paths"]["logs_dir"] / "build_signals.log")
    dets = sorted(set().union(*[r.keys() for r in res.values()]))
    flows = {d: float(np.mean([res[k].get(d, 0.0) for k in seeds])) for d in dets}
    plan = webster_plan(flows, s, struct, scfg)
    plan.info["flow_source"] = (f"E1 nVehContrib, default plan, {scfg['webster']['design_scenario']} "
                                f"seeds {seeds}, window {scfg['webster']['flow_window']}")
    write_plan(plan, struct, plan_file(scfg, "webster"), scfg, note="Webster from measured E1 flows.")
    plan.to_json(signals_dir(scfg) / "webster.json")
    log.info("webster: common cycle %d s", plan.cycle)
    for tid in sorted(struct):
        j = plan.info["junctions"][tid]
        log.info("  %s Y=%.3f C0=%s greens(N,E,S,W)=%s", tid, j["Y"], j["C0_raw"], plan.greens[tid])


def step_actuated(cfg, scfg, struct, files) -> None:
    src = signals_dir(scfg) / "optimised.json"
    if not src.exists():
        src = signals_dir(scfg) / "webster.json"
        log.warning("optimised.json not found - using webster.json as the green-time guide")
    guide = Plan.from_json(src)
    plan = Plan("actuated", guide.cycle, guide.greens, guide.offsets, {"guide": src.name})
    write_plan(plan, struct, plan_file(scfg, "actuated"), scfg, actuated=True,
               note=f"Gap-based actuation on E1 stop-line loops; green durations from {src.name}.")
    plan.to_json(signals_dir(scfg) / "actuated.json")
    log.info("actuated -> %s (guide %s, cycle %d s)", plan_file(scfg, "actuated"), src.name, guide.cycle)


STEPS = {"default": step_default, "saturation": step_saturation, "webster": step_webster,
         "actuated": step_actuated}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("steps", nargs="*", default=["default", "webster"], choices=list(STEPS) + [[]])
    ap.add_argument("--config", default=None)
    ap.add_argument("--signals-config", default=str(SIGNALS_CONFIG))
    args = ap.parse_args()
    cfg, scfg = load_config(args.config), load_signals_config(args.signals_config)
    setup_logging(PROJECT_ROOT / cfg["paths"]["logs_dir"] / "build_signals.log")
    files = network_files(cfg, scfg["network"])
    struct = phase_structure(files["net"], files["mapping"])
    for step in args.steps:
        STEPS[step](cfg, scfg, struct, files)


if __name__ == "__main__":
    main()

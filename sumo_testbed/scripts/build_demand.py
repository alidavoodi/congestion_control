#!/usr/bin/env python
"""Generate demand scenarios (route files per scenario and seed).

Scenarios (configs/phase0.yaml -> scenarios):
  normal_peak (D1), day_to_day (D2), directional_shift (D3), surge (D4),
  light, oversaturated.
Output: demand/<scenario>/seed_<k>.rou.xml + meta.yaml (+ draws for D2).

The D4 surge is checked with a test run under the default signals; if no
spillback occurs during the surge, the increase is raised in steps until it does.

Usage:
    python scripts/build_demand.py [--scenarios normal_peak surge] [--seeds 1 2 3] [--write-od]
"""
from __future__ import annotations

import argparse
import logging
import shutil

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from testbed.config import load_config, load_demand_levels, project_path
from testbed.demand import (DemandSpec, Surge, analysis_window, day_to_day, directional_shift,
                            load_od, make_od_base, od_path, profile_end, write_flows)
from testbed.log import setup_logging
from testbed.runner import run
from testbed.scenarios import make_runspec, route_file, scenario_dir

log = logging.getLogger("build_demand")

SCENARIO_ORDER = ["normal_peak", "day_to_day", "directional_shift", "surge", "light", "oversaturated"]
# Distinct RNG stream per scenario so draws never coincide across scenarios.
SCENARIO_STREAM = {name: k for k, name in enumerate(SCENARIO_ORDER)}


def write_meta(cfg, scenario: str, sc: dict, mult: float, seeds: list[int], extra: dict) -> None:
    """Write demand/<scenario>/meta.yaml."""
    meta = {
        "scenario": scenario,
        "network": cfg["demand"]["network"],
        "level": sc["level"],
        "multiplier": mult,
        "profile": sc["profile"],
        "analysis_window": list(analysis_window(cfg, sc["profile"])),
        "insertion_end": profile_end(cfg, sc["profile"]),
        "seeds": seeds,
        "sumo_seed": "equal to the file's seed (seed_<k> -> sumo --seed k)",
        "od_file": cfg["demand"]["od_file"],
    } | extra
    (scenario_dir(cfg, scenario) / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def surge_spillbacks(cfg, sc: dict, mult: float, increase: float) -> tuple[int, int]:
    """Test-run the surge (check seed, default signals); return (#spillbacks in surge window, total)."""
    od = load_od(cfg)
    seed = int(sc["check_seed"])
    s0, s1 = float(sc["start_s"]), float(sc["start_s"]) + float(sc["duration_s"])
    tmp = scenario_dir(cfg, "surge") / "_check"
    rf = tmp / f"seed_{seed}.rou.xml"
    write_flows(DemandSpec(od, mult, sc["profile"], Surge(sc["side"], s0, s1, increase)), cfg, rf)
    spec = make_runspec(cfg, "surge", seed, None, tmp / f"inc_{increase:.2f}",
                        {"check_increase": increase})
    spec.sim.route_file = rf
    m = run(spec, cfg)
    ev = pd.read_csv(tmp / f"inc_{increase:.2f}" / "spillbacks.csv")
    in_surge = int(((ev["end"] > s0) & (ev["start"] < s1)).sum()) if len(ev) else 0
    return in_surge, int(m["spillback_count"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--write-od", action="store_true", help="regenerate configs/demand/od_base.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(project_path(cfg, "logs_dir") / "build_demand.log")
    if args.write_od:
        net_path = project_path(cfg, "networks_dir") / f"{cfg['demand']['network']}.net.xml"
        make_od_base(net_path, cfg).to_csv(od_path(cfg), index=False)
    od = load_od(cfg)
    levels = load_demand_levels(cfg)
    seeds = args.seeds or list(cfg["scenarios"]["seeds"])
    for name in args.scenarios or SCENARIO_ORDER:
        sc = cfg["scenarios"][name]
        mult = levels[sc["level"]]
        d = scenario_dir(cfg, name)
        d.mkdir(parents=True, exist_ok=True)
        write_meta(cfg, name, sc, mult, seeds, {})  # needed by the surge test run
        extra: dict = {}
        if name == "day_to_day":
            draws = []
            for k in seeds:
                rng = np.random.default_rng([k, SCENARIO_STREAM[name]])
                od_k, total, cell = day_to_day(od, rng, sc["total_sd"], tuple(sc["total_clip"]), sc["cell_sigma"])
                write_flows(DemandSpec(od_k, mult, sc["profile"], meta={"scenario": name, "seed": k,
                                                                         "total_factor": f"{total:.4f}"}),
                            cfg, route_file(cfg, name, k))
                draws.append({"seed": k, "total_factor": round(total, 6)})
                pd.DataFrame({"origin": od["origin"], "dest": od["dest"], "cell_factor": cell,
                              "vph": od_k["vph"]}).to_csv(d / f"seed_{k}_od.csv", index=False)
            extra["draws"] = draws
        elif name == "surge":
            inc, step, top = float(sc["increase"]), float(sc["increase_step"]), float(sc["increase_max"])
            while True:
                n_surge, n_all = surge_spillbacks(cfg, sc, mult, inc)
                log.info("surge check: +%.0f%% -> %d spillback events during surge (%d in analysis window)",
                         100 * inc, n_surge, n_all)
                if n_surge > 0 or inc >= top - 1e-9:
                    break
                inc = round(inc + step, 6)
            if n_surge == 0:
                log.warning("no spillback even at +%.0f%% (increase_max)", 100 * inc)
            s0 = float(sc["start_s"])
            surge = Surge(sc["side"], s0, s0 + float(sc["duration_s"]), inc)
            for k in seeds:
                write_flows(DemandSpec(od, mult, sc["profile"], surge, {"scenario": name, "seed": k}),
                            cfg, route_file(cfg, name, k))
            extra["surge"] = {"side": sc["side"], "start_s": s0, "duration_s": float(sc["duration_s"]),
                              "increase_configured": float(sc["increase"]), "increase_final": inc,
                              "check_seed": int(sc["check_seed"]),
                              "check_spillbacks_during_surge": n_surge}
            shutil.rmtree(d / "_check", ignore_errors=True)
        else:
            od_s = (directional_shift(od, sc["main_directions"], sc["main_share"])
                    if name == "directional_shift" else od)
            for k in seeds:
                write_flows(DemandSpec(od_s, mult, sc["profile"], meta={"scenario": name, "seed": k}),
                            cfg, route_file(cfg, name, k))
            if name == "directional_shift":
                extra["main_share"] = sc["main_share"]
                extra["main_directions"] = sc["main_directions"]
        write_meta(cfg, name, sc, mult, seeds, extra)
        log.info("%s: level %s (x%.3f), profile %s, %d seeds -> %s", name, sc["level"], mult,
                 sc["profile"], len(seeds), d)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate networks, detectors (E1 + E2), the detector mapping and a layout PNG.

Usage:
    python scripts/build_network.py [--net grid3x3 corridor] [--config configs/phase0.yaml]
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from testbed.config import load_config, network_files, project_path
from testbed.log import setup_logging
from testbed.network import (build_network, detector_records, grid_tls_ids, plot_network,
                             validate_network, write_detectors)

log = logging.getLogger("build_network")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--net", nargs="*", default=None, help="networks to build (default: all in config)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(project_path(cfg, "logs_dir") / "build_network.log")
    for name in args.net or list(cfg["networks"]):
        files = network_files(cfg, name)
        net_path = build_network(cfg, name)
        tls = grid_tls_ids(cfg["networks"][name]["netgenerate"])
        validate_network(net_path, tls)
        recs = detector_records(net_path, cfg)
        write_detectors(recs, cfg, files["det"], files["mapping"])
        n_lanes = len({r.lane for r in recs})
        log.info("%s: %d signalised junctions, %d incoming lanes, %d E1 + %d E2",
                 name, len(tls), n_lanes, sum(r.det_type == "E1" for r in recs),
                 sum(r.det_type == "E2" for r in recs))
        plot_network(net_path, recs, files["png"], f"{name}: {len(tls)} signalised junctions, "
                     f"{n_lanes} monitored incoming lanes")


if __name__ == "__main__":
    main()

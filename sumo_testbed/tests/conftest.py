"""Shared fixtures: build networks into a temporary folder."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from testbed.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def tmp_cfg(tmp_path_factory) -> dict:
    """Config whose networks/results folders point to a temp dir (absolute paths)."""
    cfg = copy.deepcopy(load_config())
    tmp = tmp_path_factory.mktemp("testbed")
    for key in ("networks_dir", "results_dir", "calibration_dir", "logs_dir"):
        cfg["paths"][key] = str(tmp / key)
    cfg["demand"]["od_file"] = str(tmp / "od_base.csv")
    cfg["demand"]["out_dir"] = str(tmp / "demand")
    return cfg


@pytest.fixture(scope="session")
def grid(tmp_cfg) -> dict:
    """Built grid3x3 network + detectors in the temp folder."""
    from testbed.config import network_files
    from testbed.network import build_network, detector_records, write_detectors

    files = network_files(tmp_cfg, "grid3x3")
    net = build_network(tmp_cfg, "grid3x3")
    recs = detector_records(net, tmp_cfg)
    write_detectors(recs, tmp_cfg, files["det"], files["mapping"])
    return {"files": files, "recs": recs}

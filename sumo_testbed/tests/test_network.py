"""Network and detector tests."""
from __future__ import annotations

import subprocess
from collections import Counter

import pytest

from testbed.network import grid_tls_ids, read_net, validate_network
from testbed.sumo_env import sumo_binary


def test_network_builds_and_is_protected(grid, tmp_cfg):
    tls = grid_tls_ids(tmp_cfg["networks"]["grid3x3"]["netgenerate"])
    assert len(tls) == 9
    validate_network(grid["files"]["net"], tls)


def test_every_incoming_lane_has_one_e1_and_one_e2(grid):
    net = read_net(grid["files"]["net"])
    incoming = {ln.getID() for t in net.getTrafficLights() for ln, _o, _i in t.getConnections()}
    assert len(incoming) == 72
    per_lane = Counter((r.lane, r.det_type) for r in grid["recs"])
    for lane in incoming:
        assert per_lane[(lane, "E1")] == 1 and per_lane[(lane, "E2")] == 1, lane
    assert set(r.lane for r in grid["recs"]) == incoming


def test_e2_covers_full_lane_and_e1_near_stop_line(grid, tmp_cfg):
    setback = tmp_cfg["detectors"]["e1"]["setback_m"]
    for r in grid["recs"]:
        if r.det_type == "E2":
            assert r.pos == 0.0
        else:
            assert r.pos == pytest.approx(r.lane_length - setback)
        assert r.phases.count(";") == 0, "split phasing: each lane served by exactly one phase"


def test_validator_rejects_permissive_layout(tmp_path):
    net = tmp_path / "perm.net.xml"
    subprocess.run([sumo_binary("netgenerate"), "--grid", "--grid.number", "3", "--grid.attach-length", "200",
                    "--default.lanenumber", "2", "--tls.set", "B1", "--tls.layout", "opposites",
                    "-o", str(net)], check=True, capture_output=True)
    with pytest.raises(AssertionError):
        validate_network(net, ["B1"])

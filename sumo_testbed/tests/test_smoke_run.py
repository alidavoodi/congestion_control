"""A short (5-min) run completes with zero teleports."""
from __future__ import annotations

from testbed.demand import constant_demand
from testbed.sim import SimSpec, SumoSim


def test_five_minute_run_no_teleports(grid, tmp_cfg, tmp_path):
    f = grid["files"]
    spec = SimSpec(f["net"], constant_demand(tmp_cfg, 0.6), f["det"], None,
                   tmp_path / "run", seed=1)
    sim = SumoSim(spec, tmp_cfg)
    sim.start()
    while sim.time < tmp_cfg["smoke_test"]["duration_s"]:
        sim.step()
    assert sim.traci.simulation.getArrivedNumber() >= 0
    sim.close()  # raises TeleportError if any teleport happened
    assert sim.teleports == 0
    assert (tmp_path / "run" / "tripinfo.xml").exists()

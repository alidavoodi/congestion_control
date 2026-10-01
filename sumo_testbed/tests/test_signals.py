"""Signal plan constraints and phase structure."""
from __future__ import annotations

import pytest

from testbed.signals import cycle_bounds, default_plan, load_signals_config, phase_structure, split_greens, webster_plan

SCFG = load_signals_config()


@pytest.mark.parametrize("cycle", [48, 63, 90, 120])
@pytest.mark.parametrize("weights", [[1, 1, 1, 1], [10, 1, 1, 1], [0.0, 0.0, 0.0, 5.0], [0.3, 0.2, 0.25, 0.25]])
def test_split_greens_respects_constraints(cycle, weights):
    c = SCFG["constraints"]
    g = split_greens(weights, cycle, SCFG)
    assert sum(g) + 4 * (c["yellow_s"] + c["all_red_s"]) == cycle
    assert min(g) >= c["min_green_s"] and max(g) <= c["max_green_s"]


def test_cycle_bounds_account_for_min_green():
    assert cycle_bounds(SCFG, 4) == (48, 120)  # 4 x (7 + 5) = 48 > 40


def test_phase_structure_and_default_plan(grid):
    f = grid["files"]
    struct = phase_structure(f["net"], f["mapping"])
    assert len(struct) == 9 and all(len(p) == 4 for p in struct.values())
    for phases in struct.values():
        assert sum(len(ph.lanes) for ph in phases) == 8  # 4 approaches x 2 lanes
    plan = default_plan(f["net"], struct)
    assert plan.cycle == 90 and plan.greens["B1"] == [16, 18, 18, 18]


def test_webster_common_cycle(grid):
    f = grid["files"]
    struct = phase_structure(f["net"], f["mapping"])
    flows = {d: 300.0 for phases in struct.values() for ph in phases for d in ph.e1}
    plan = webster_plan(flows, 1600.0, struct, SCFG)
    # y = 300/1600 per phase, Y = 0.75, L = 20 -> C0 = 35 / 0.25 = 140 -> clipped to 120
    assert plan.cycle == 120
    assert all(g == [25, 25, 25, 25] for g in plan.greens.values())

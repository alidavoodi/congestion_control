"""Demand: OD structure, profile, scenario transforms and reproducibility."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from testbed.demand import (DemandSpec, Surge, constant_demand, day_to_day, directional_shift,
                            fringe_sides, load_od, profile_slices, write_flows)


def _md5(p) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def test_fringe_sides(grid):
    entries, exits = fringe_sides(grid["files"]["net"])
    assert len(entries) == 12 and len(exits) == 12
    for sides in (entries, exits):
        assert sorted(sides.values()) == sorted(["W", "E", "S", "N"] * 3)


def test_od_excludes_uturn_and_same_side_and_splits_through(grid, tmp_cfg):
    od = load_od(tmp_cfg)
    assert len(od) == 12 * 9  # 3 opposite + 6 adjacent exits per entry
    assert not (od["origin_side"] == od["dest_side"]).any()
    per_origin = od.groupby("origin")["vph"].sum()
    assert np.allclose(per_origin, tmp_cfg["demand"]["base_origin_flow_vph"])
    thr = od[od.movement == "through"].groupby("origin")["vph"].sum() / per_origin
    assert np.allclose(thr, tmp_cfg["demand"]["through_share"])


def test_peak_profile_shape(tmp_cfg):
    f = [s.factor for s in profile_slices(tmp_cfg, "peak")]
    assert len(f) == 17  # 85 min in 5-min slices
    assert f[:3] == [0.5] * 3 and f[7:13] == [1.0] * 6
    assert f[3] < f[4] < f[5] < f[6] < 1.0 and f[13] > f[14] > f[15] > f[16] > 0.5


def test_constant_demand_reproducible_and_scales(grid, tmp_cfg):
    a = constant_demand(tmp_cfg, 0.5, overwrite=True)
    h = _md5(a)
    assert _md5(constant_demand(tmp_cfg, 0.5, overwrite=True)) == h
    od = load_od(tmp_cfg)
    n1 = write_flows(DemandSpec(od, 0.5, "constant"), tmp_cfg, a)
    n2 = write_flows(DemandSpec(od, 1.0, "constant"), tmp_cfg, a.with_name("x.rou.xml"))
    assert n2 == pytest.approx(2 * n1, rel=1e-6)
    assert n2 == pytest.approx(12 * 300 * 75 / 60, rel=1e-6)


def test_directional_shift_keeps_total(grid, tmp_cfg):
    od = load_od(tmp_cfg)
    sh = directional_shift(od, [["W", "E"], ["S", "N"]], 0.7)
    main = ((sh.origin_side == "W") & (sh.dest_side == "E")) | ((sh.origin_side == "S") & (sh.dest_side == "N"))
    assert sh["vph"].sum() == pytest.approx(od["vph"].sum())
    assert sh.loc[main, "vph"].sum() / sh["vph"].sum() == pytest.approx(0.7)


def test_day_to_day_reproducible(grid, tmp_cfg):
    od = load_od(tmp_cfg)
    a = day_to_day(od, np.random.default_rng(3), 0.1, (0.8, 1.2), 0.15)
    b = day_to_day(od, np.random.default_rng(3), 0.1, (0.8, 1.2), 0.15)
    assert a[1] == b[1] and np.array_equal(a[2], b[2])
    assert 0.8 <= a[1] <= 1.2
    assert a[0]["vph"].sum() == pytest.approx(a[1] * od["vph"].sum())


def test_surge_only_affects_one_side(grid, tmp_cfg, tmp_path):
    od = load_od(tmp_cfg)
    base = write_flows(DemandSpec(od, 1.0, "constant"), tmp_cfg, tmp_path / "a.rou.xml")
    surged = write_flows(DemandSpec(od, 1.0, "constant", Surge("W", 900, 2700, 0.4)), tmp_cfg,
                         tmp_path / "b.rou.xml")
    w_vph = od[od.origin_side == "W"]["vph"].sum()
    assert surged - base == pytest.approx(w_vph * 0.4 * 0.5, abs=2)  # 30 min at +40%

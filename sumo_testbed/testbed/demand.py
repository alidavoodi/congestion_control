"""OD-based, time-sliced fringe-to-fringe demand.

* The base OD (``configs/demand/od_base.csv``) lists every allowed fringe entry
  edge -> fringe exit edge pair with its flow in veh/h at multiplier 1.0.
  U-turns and same-side trips are excluded; each entry's flow is split into
  through (opposite side) and turning (adjacent side) traffic.
* A global multiplier scales the whole OD; a time profile scales it over time.
* Each OD cell becomes one ``<flow>`` per 5-min slice with Poisson arrivals
  (``period="exp(rate)"``). SUMO routes each vehicle on the fastest free-flow
  path at insertion. Arrival randomness is driven by SUMO's ``--seed``, which
  the runner sets to the scenario seed; scenario-level randomness (D2) is drawn
  here with a seeded numpy RNG and stored next to the route files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from testbed.config import PROJECT_ROOT, project_path
from testbed.network import read_net

log = logging.getLogger(__name__)

SIDES = ("W", "E", "S", "N")
OPPOSITE = {"W": "E", "E": "W", "S": "N", "N": "S"}


@dataclass(frozen=True)
class Slice:
    """A demand time slice [begin, end) with a profile factor."""

    begin: float
    end: float
    factor: float


@dataclass
class Surge:
    """Extra demand on all trips entering from ``side`` during [start, end)."""

    side: str
    start: float
    end: float
    increase: float  # +0.4 = +40 %


@dataclass
class DemandSpec:
    """Everything needed to write one route file."""

    od: pd.DataFrame  # columns origin, dest, origin_side, dest_side, movement, vph
    multiplier: float
    profile: str
    surge: Surge | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- OD
def fringe_sides(net_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Classify fringe entry/exit edges by the side of the network they touch.

    Returns:
        (entries, exits): ``{edge_id: side}`` with side in W/E/S/N, determined
        from the fringe node's position relative to the network bounding box.
    """
    net = read_net(net_path)
    tls = {t.getID() for t in net.getTrafficLights()}
    (xmin, ymin), (xmax, ymax) = net.getBBoxXY()

    def side(node) -> str:
        x, y = node.getCoord()
        d = {"W": x - xmin, "E": xmax - x, "S": y - ymin, "N": ymax - y}
        return min(d, key=d.get)

    entries, exits = {}, {}
    for e in net.getEdges():
        f, t = e.getFromNode(), e.getToNode()
        if f.getID() not in tls and t.getID() in tls:
            entries[e.getID()] = side(f)
        elif f.getID() in tls and t.getID() not in tls:
            exits[e.getID()] = side(t)
    return dict(sorted(entries.items())), dict(sorted(exits.items()))


def make_od_base(net_path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build the base OD: per entry, ``through_share`` to opposite-side exits, rest to adjacent sides.

    Flow is divided equally among the exits of each movement class.
    """
    d = cfg["demand"]
    base, through = float(d["base_origin_flow_vph"]), float(d["through_share"])
    entries, exits = fringe_sides(net_path)
    rows = []
    for o, os_ in entries.items():
        thr = [x for x, s in exits.items() if s == OPPOSITE[os_]]
        trn = [x for x, s in exits.items() if s not in (os_, OPPOSITE[os_])]
        for group, share, mv in ((thr, through, "through"), (trn, 1 - through, "turn")):
            for x in group:
                rows.append(dict(origin=o, dest=x, origin_side=os_, dest_side=exits[x],
                                 movement=mv, vph=round(base * share / len(group), 4)))
    return pd.DataFrame(rows)


def od_path(cfg: dict[str, Any]) -> Path:
    """Location of the base OD CSV."""
    return PROJECT_ROOT / cfg["demand"]["od_file"]


def load_od(cfg: dict[str, Any], write_if_missing: bool = True) -> pd.DataFrame:
    """Load the base OD CSV, generating it from the config if it does not exist."""
    p = od_path(cfg)
    if not p.exists():
        if not write_if_missing:
            raise FileNotFoundError(p)
        net_path = project_path(cfg, "networks_dir") / f"{cfg['demand']['network']}.net.xml"
        p.parent.mkdir(parents=True, exist_ok=True)
        make_od_base(net_path, cfg).to_csv(p, index=False)
        log.info("wrote base OD -> %s", p)
    return pd.read_csv(p)


# ---------------------------------------------------------------------- profile
def profile_slices(cfg: dict[str, Any], profile: str) -> list[Slice]:
    """Expand a profile's segments into time slices (factor at slice midpoint)."""
    step = float(cfg["demand"]["slice_s"])
    out: list[Slice] = []
    t0 = 0.0
    for dur, f0, f1 in cfg["demand"]["profiles"][profile]["segments"]:
        n = int(round(dur / step))
        if abs(n * step - dur) > 1e-9:
            raise ValueError(f"segment duration {dur} is not a multiple of slice_s={step}")
        for k in range(n):
            mid = (k + 0.5) / n
            out.append(Slice(t0 + k * step, t0 + (k + 1) * step, f0 + (f1 - f0) * mid))
        t0 += dur
    return out


def profile_end(cfg: dict[str, Any], profile: str) -> float:
    """End of insertion for a profile [s]."""
    return float(sum(s[0] for s in cfg["demand"]["profiles"][profile]["segments"]))


def analysis_window(cfg: dict[str, Any], profile: str) -> tuple[float, float]:
    """Metric analysis window [start, end) for a profile."""
    a, b = cfg["demand"]["profiles"][profile]["analysis"]
    return float(a), float(b)


# -------------------------------------------------------------------- scenarios
def directional_shift(od: pd.DataFrame, main_dirs: list[list[str]], share: float) -> pd.DataFrame:
    """Rescale the OD so trips in ``main_dirs`` make up ``share`` of the total (total unchanged)."""
    od = od.copy()
    main = np.zeros(len(od), dtype=bool)
    for a, b in main_dirs:
        main |= (od["origin_side"] == a).to_numpy() & (od["dest_side"] == b).to_numpy()
    total, m = od["vph"].sum(), od.loc[main, "vph"].sum()
    od.loc[main, "vph"] *= share * total / m
    od.loc[~main, "vph"] *= (1 - share) * total / (total - m)
    return od


def day_to_day(od: pd.DataFrame, rng: np.random.Generator, total_sd: float,
               clip: tuple[float, float], cell_sigma: float) -> tuple[pd.DataFrame, float, np.ndarray]:
    """Draw a day: total factor ~ N(1, sd) clipped, cell noise ~ lognormal(mean 1, sigma).

    Cell noise is renormalised so the OD total equals ``total_factor`` x base total.

    Returns:
        (perturbed OD, total_factor, cell_factors)
    """
    total = float(np.clip(rng.normal(1.0, total_sd), *clip))
    cell = rng.lognormal(-0.5 * cell_sigma**2, cell_sigma, len(od))
    od = od.copy()
    base_total = od["vph"].sum()
    od["vph"] = od["vph"] * cell
    od["vph"] *= total * base_total / od["vph"].sum()
    return od, total, cell


# ------------------------------------------------------------------------ write
def write_flows(spec: DemandSpec, cfg: dict[str, Any], path: Path) -> int:
    """Write a route file with one Poisson flow per OD cell and slice.

    Returns:
        Expected number of vehicles (sum of rate x slice length).
    """
    vt, ins = cfg["demand"]["vehicle_type"], cfg["demand"]["insertion"]
    attrs = " ".join(f'{k}="{v}"' for k, v in vt.items() if k != "id")
    s = spec.surge
    flows: list[tuple[float, str, str]] = []
    expected = 0.0
    for k, sl in enumerate(profile_slices(cfg, spec.profile)):
        for row in spec.od.itertuples(index=False):
            vph = row.vph * spec.multiplier * sl.factor
            if s is not None and row.origin_side == s.side:
                overlap = max(0.0, min(sl.end, s.end) - max(sl.begin, s.start))
                vph *= 1 + s.increase * overlap / (sl.end - sl.begin)
            if vph <= 0:
                continue
            rate = vph / 3600.0
            expected += rate * (sl.end - sl.begin)
            fid = f"{row.origin}.{row.dest}.{k}"
            flows.append((sl.begin, fid,
                          f'    <flow id="{fid}" type="{vt["id"]}" begin="{sl.begin:.0f}" end="{sl.end:.0f}" '
                          f'from="{row.origin}" to="{row.dest}" period="exp({rate:.8f})" '
                          f'departLane="{ins["depart_lane"]}" departSpeed="{ins["depart_speed"]}"/>'))
    flows.sort(key=lambda x: (x[0], x[1]))
    head = " ".join(f"{k}={v}" for k, v in spec.meta.items())
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f"<!-- Generated by testbed/demand.py. multiplier={spec.multiplier:.4f} "
             f"profile={spec.profile} {head} -->",
             "<routes>",
             f'    <vType id="{vt["id"]}" {attrs}/>']
    lines += [f[2] for f in flows]
    lines.append("</routes>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(round(expected))


def demand_dir(cfg: dict[str, Any]) -> Path:
    """Root folder for generated demand files."""
    return PROJECT_ROOT / cfg["demand"]["out_dir"]


def constant_demand(cfg: dict[str, Any], multiplier: float, overwrite: bool = False) -> Path:
    """Constant-profile demand file used for calibration and smoke runs (seed-independent)."""
    path = demand_dir(cfg) / "calibration" / f"m{multiplier:.3f}.rou.xml"
    if overwrite or not path.exists():
        n = write_flows(DemandSpec(load_od(cfg), multiplier, "constant"), cfg, path)
        log.info("demand %s: ~%d vehicles", path.name, n)
    return path

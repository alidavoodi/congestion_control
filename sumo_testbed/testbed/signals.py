"""Fixed-time and actuated signal plans with a frozen phase structure.

The phase sequence (split phasing: one protected green per approach, each
followed by yellow and all-red) is read from the network's default program and
never altered. A :class:`Plan` only sets the common cycle, the green time of
each phase and an offset per junction.
"""
from __future__ import annotations

import json
import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from testbed.config import PROJECT_ROOT
from testbed.network import load_mapping

log = logging.getLogger(__name__)
SIGNALS_CONFIG = PROJECT_ROOT / "configs" / "signals.yaml"


def load_signals_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``configs/signals.yaml``."""
    return yaml.safe_load(Path(path or SIGNALS_CONFIG).read_text())


def signals_dir(scfg: dict[str, Any]) -> Path:
    """Folder holding ``<plan>.add.xml`` files."""
    return PROJECT_ROOT / scfg["out_dir"]


def plan_file(scfg: dict[str, Any], name: str) -> Path:
    """Path of a plan's additional file."""
    return signals_dir(scfg) / f"{name}.add.xml"


# ---------------------------------------------------------------- structure
@dataclass(frozen=True)
class ApproachPhase:
    """One protected green with its yellow and all-red states."""

    green: str
    yellow: str
    all_red: str
    lanes: tuple[str, ...]  # incoming lanes served
    e1: tuple[str, ...]  # stop-line loops on those lanes


def phase_structure(net_path: Path, mapping_path: Path) -> dict[str, list[ApproachPhase]]:
    """Read the default program's phase sequence per junction.

    Raises:
        ValueError: if a program is not a repetition of (green, yellow, all-red).
    """
    rows = [r for r in load_mapping(mapping_path) if r["det_type"] == "E1"]
    out: dict[str, list[ApproachPhase]] = {}
    for tl in ET.parse(net_path).getroot().iter("tlLogic"):
        tid = tl.get("id")
        states = [p.get("state") for p in tl.findall("phase")]
        if len(states) % 3:
            raise ValueError(f"{tid}: expected (G, y, r) triples, got {len(states)} phases")
        phases = []
        for k in range(0, len(states), 3):
            g, y, r = states[k:k + 3]
            if "G" not in g or "y" not in y or set(r) != {"r"}:
                raise ValueError(f"{tid}: unexpected phase triple {g} {y} {r}")
            served = [row for row in rows if row["junction"] == tid and str(k) in row["phases"].split(";")]
            phases.append(ApproachPhase(g, y, r, tuple(row["lane"] for row in served),
                                        tuple(row["detector_id"] for row in served)))
        out[tid] = phases
    return out


# --------------------------------------------------------------------- plan
@dataclass
class Plan:
    """Common-cycle fixed-time plan."""

    name: str
    cycle: int
    greens: dict[str, list[int]]
    offsets: dict[str, int] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=float))

    @classmethod
    def from_json(cls, path: Path) -> "Plan":
        return cls(**json.loads(path.read_text()))


def intergreen(scfg: dict[str, Any]) -> int:
    """Yellow + all-red per phase [s]."""
    c = scfg["constraints"]
    return int(c["yellow_s"]) + int(c["all_red_s"])


def cycle_bounds(scfg: dict[str, Any], n_phases: int) -> tuple[int, int]:
    """Feasible common-cycle range given min/max greens and inter-greens."""
    c = scfg["constraints"]
    lo = max(int(c["cycle_min_s"]), n_phases * (int(c["min_green_s"]) + intergreen(scfg)))
    hi = min(int(c["cycle_max_s"]), n_phases * (int(c["max_green_s"]) + intergreen(scfg)))
    return lo, hi


def split_greens(weights: list[float], cycle: int, scfg: dict[str, Any]) -> list[int]:
    """Integer greens proportional to ``weights`` that respect min/max green and sum to C - lost.

    Greens are clip(lam * w, min, max) with lam chosen (bisection) so they sum to
    C - lost exactly, then rounded with the largest-remainder method.
    """
    c = scfg["constraints"]
    lo, hi = int(c["min_green_s"]), int(c["max_green_s"])
    n = len(weights)
    total = cycle - n * intergreen(scfg)
    if not n * lo <= total <= n * hi:
        raise ValueError(f"cycle {cycle} infeasible for {n} phases")
    w = np.maximum(np.asarray(weights, dtype=float), 1e-9)
    # sum(clip(lam * w, lo, hi)) is monotone in lam: bisect for the exact total.
    a, b = 0.0, hi / w.min()
    for _ in range(200):
        lam = 0.5 * (a + b)
        if np.clip(lam * w, lo, hi).sum() < total:
            a = lam
        else:
            b = lam
    g = np.clip(b * w, lo, hi)
    base = np.floor(g + 1e-9).astype(int)
    short = total - base.sum()
    order = np.argsort(-(g - base))
    for i in order:
        if short <= 0:
            break
        if base[i] < hi:
            base[i] += 1
            short -= 1
    assert base.sum() == total and base.min() >= lo and base.max() <= hi, (base, total)
    return [int(x) for x in base]


def _tl_xml(tid: str, phases: list[ApproachPhase], greens: list[int], offset: int, prog: str,
            scfg: dict[str, Any], actuated: bool = False) -> list[str]:
    c = scfg["constraints"]
    typ = "actuated" if actuated else "static"
    lines = [f'    <tlLogic id="{tid}" type="{typ}" programID="{prog}" offset="{offset}">']
    if actuated:
        a = scfg["actuated"]
        lines += [f'        <param key="max-gap" value="{a["max_gap_s"]}"/>',
                  f'        <param key="passing-time" value="{a["passing_time_s"]}"/>',
                  f'        <param key="show-detectors" value="{str(a["show_detectors"]).lower()}"/>']
        for ph in phases:  # use the existing E1 stop-line loops as actuation detectors
            for lane, det in zip(ph.lanes, ph.e1):
                lines.append(f'        <param key="{lane}" value="{det}"/>')
    for ph, g in zip(phases, greens):
        extra = f' minDur="{c["min_green_s"]}" maxDur="{c["max_green_s"]}"' if actuated else ""
        lines += [f'        <phase duration="{g}" state="{ph.green}"{extra}/>',
                  f'        <phase duration="{c["yellow_s"]}" state="{ph.yellow}"/>',
                  f'        <phase duration="{c["all_red_s"]}" state="{ph.all_red}"/>']
    lines.append("    </tlLogic>")
    return lines


def write_plan(plan: Plan, struct: dict[str, list[ApproachPhase]], path: Path,
               scfg: dict[str, Any], actuated: bool = False, note: str = "") -> Path:
    """Write a plan as a SUMO additional file (program id = plan name)."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f"<!-- Signal plan '{plan.name}' ({'actuated' if actuated else 'fixed-time'}), "
             f"cycle {plan.cycle} s. {note} -->",
             "<additional>"]
    for tid in sorted(struct):
        lines += _tl_xml(tid, struct[tid], plan.greens[tid], int(plan.offsets.get(tid, 0)),
                         plan.name, scfg, actuated)
    lines.append("</additional>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def default_plan(net_path: Path, struct: dict[str, list[ApproachPhase]]) -> Plan:
    """The network's built-in program as a Plan (greens, cycle, offsets)."""
    greens, offsets, cycles = {}, {}, set()
    for tl in ET.parse(net_path).getroot().iter("tlLogic"):
        d = [int(float(p.get("duration"))) for p in tl.findall("phase")]
        greens[tl.get("id")] = d[0::3]
        offsets[tl.get("id")] = int(float(tl.get("offset")))
        cycles.add(sum(d))
    if len(cycles) != 1:
        raise ValueError(f"default programs have different cycles: {cycles}")
    return Plan("default", cycles.pop(), greens, offsets, {"source": "netgenerate default"})


# ------------------------------------------------------------------ Webster
def e1_flows(e1_xml: Path, window: tuple[float, float]) -> dict[str, float]:
    """Per-detector flow [veh/h] in ``window`` from E1 output (partial intervals pro-rated)."""
    t0, t1 = window
    counts: dict[str, float] = {}
    for _ev, el in ET.iterparse(e1_xml, events=("end",)):
        if el.tag != "interval":
            continue
        b, e = float(el.get("begin")), float(el.get("end"))
        ov = max(0.0, min(e, t1) - max(b, t0))
        if ov > 0:
            counts[el.get("id")] = counts.get(el.get("id"), 0.0) + float(el.get("nVehContrib")) * ov / (e - b)
        el.clear()
    return {k: v * 3600.0 / (t1 - t0) for k, v in counts.items()}


def webster_plan(flows: dict[str, float], sat_flow: float, struct: dict[str, list[ApproachPhase]],
                 scfg: dict[str, Any]) -> Plan:
    """Webster per junction, then one common cycle (max of junction cycles) with re-split greens.

    y_phase = max over its lanes of q/s;  Y = sum(y);  L = n x lost_time_per_phase;
    C0 = (1.5 L + 5) / (1 - Y), clipped to the feasible cycle range;
    greens proportional to y_phase over C - n x inter-green.
    """
    lost = float(scfg["webster"]["lost_time_per_phase_s"])
    info: dict[str, Any] = {"saturation_flow_vphpl": sat_flow, "junctions": {}}
    ys, c0s = {}, {}
    n = len(next(iter(struct.values())))
    lo, hi = cycle_bounds(scfg, n)
    for tid, phases in struct.items():
        y = [max(flows.get(d, 0.0) for d in ph.e1) / sat_flow for ph in phases]
        Y = sum(y)
        L = n * lost
        c0 = (1.5 * L + 5) / (1 - Y) if Y < 1 else float("inf")
        ys[tid], c0s[tid] = y, c0
        info["junctions"][tid] = {"y": [round(v, 4) for v in y], "Y": round(Y, 4),
                                  "C0_raw": round(c0, 1) if math.isfinite(c0) else None}
    cycle = int(min(hi, max(lo, math.ceil(max(c0s.values())))))
    info["common_cycle_rule"] = "max of junction C0, rounded up, clipped to feasible range"
    greens = {tid: split_greens(ys[tid], cycle, scfg) for tid in struct}
    return Plan("webster", cycle, greens, {tid: 0 for tid in struct}, info)


# ------------------------------------------------------- saturation flow
def measure_saturation(sim, struct: dict[str, list[ApproachPhase]], mapping_path: Path,
                       mcfg: dict[str, Any], end: float) -> dict[str, Any]:
    """Measure saturation flow from stop-line discharge headways.

    A green is "saturated" if the lane's queue (E2 jam length, evaluation only)
    at green start is at least ``min_queue_m``. Within it, headways between
    successive E1 crossings are collected after the first ``skip_first``
    vehicles, until a gap exceeds ``max_headway_s`` (queue cleared) or the green
    ends. Saturation flow = 3600 / mean headway.

    Args:
        sim: A started SumoSim (default plan, high demand).
        end: Simulation time at which to stop.
    """
    tr = sim.traci
    lane_info = []  # (tls, phase_idx, lane, e1, e2)
    for tid, phases in struct.items():
        for k, ph in enumerate(phases):
            for lane, det in zip(ph.lanes, ph.e1):
                lane_info.append((tid, k, lane, det, "e2_" + lane))
    t0, t1 = mcfg["window"]
    skip, hmax, qmin = int(mcfg["skip_first"]), float(mcfg["max_headway_s"]), float(mcfg["min_queue_m"])
    green_prev = {li[2]: False for li in lane_info}
    active: dict[str, dict] = {}  # lane -> {"n": crossings, "last": time, "done": bool}
    prev_ids: dict[str, set] = {li[2]: set() for li in lane_info}  # vehicles on the loop last step
    headways: list[float] = []
    per_lane_index: dict[str, list[float]] = {"0": [], "1": []}
    while sim.time < end:
        sim.step()
        t = sim.time
        prog_idx = {tid: tr.trafficlight.getPhase(tid) for tid in struct}
        for tid, k, lane, det, e2 in lane_info:
            green = prog_idx[tid] == 3 * k
            if green and not green_prev[lane] and t0 <= t <= t1:
                if tr.lanearea.getJamLengthMeters(e2) >= qmin:
                    active[lane] = {"n": 0, "last": None, "done": False}
            if not green:
                active.pop(lane, None)
            green_prev[lane] = green
            data = tr.inductionloop.getVehicleData(det)
            now_ids = {d[0] for d in data}
            new = sorted((d for d in data if d[0] not in prev_ids[lane]), key=lambda d: d[2])
            prev_ids[lane] = now_ids
            for vid, _len, entry, _exit, _typ in new:
                a = active.get(lane)
                if a is None or a["done"]:
                    continue
                a["n"] += 1
                if a["last"] is not None and a["n"] > skip:
                    h = entry - a["last"]
                    if h > hmax:
                        a["done"] = True
                    else:
                        headways.append(h)
                        per_lane_index[lane.rsplit("_", 1)[1]].append(h)
                a["last"] = entry
    h = np.asarray(headways)
    return {
        "saturation_flow_vphpl": float(3600.0 / h.mean()),
        "mean_headway_s": float(h.mean()),
        "median_headway_s": float(np.median(h)),
        "n_headways": int(h.size),
        "per_lane_index_vphpl": {k: float(3600.0 / np.mean(v)) for k, v in per_lane_index.items() if v},
    }

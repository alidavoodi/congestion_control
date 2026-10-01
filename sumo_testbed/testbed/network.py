"""Network generation, validation, detector placement and layout plotting.

Detector roles (enforced downstream):
  * E1 induction loops, ``e1_<lane>``: stop-line loops on every incoming lane of
    every signalised junction. The ONLY data controllers may receive.
  * E2 lane-area detectors, ``e2_<lane>``: cover the full incoming lane. Used
    ONLY for evaluation (queues, spillback).
"""
from __future__ import annotations

import csv
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

from testbed.config import network_files
from testbed.sumo_env import find_sumo_home, sumo_binary

log = logging.getLogger(__name__)

E1_PREFIX = "e1_"
E2_PREFIX = "e2_"


@dataclass(frozen=True)
class DetectorRecord:
    """One row of the detector mapping file."""

    detector_id: str
    det_type: str  # "E1" | "E2"
    lane: str
    edge: str
    junction: str  # controlling tls id
    approach_from: str  # upstream node id
    phases: str  # ';'-separated green phase indices in the default program
    lane_length: float
    pos: float


def grid_tls_ids(ng: dict[str, Any]) -> list[str]:
    """Inner (signalised) node ids that netgenerate assigns to a grid: <letter><row>."""
    nx, ny = int(ng["grid.x-number"]), int(ng["grid.y-number"])
    if nx > 26:
        raise ValueError("grid.x-number > 26 not supported by the node naming scheme")
    return [f"{chr(65 + i)}{j}" for i in range(nx) for j in range(ny)]


def build_network(cfg: dict[str, Any], name: str) -> Path:
    """Generate ``networks/<name>.net.xml`` with netgenerate.

    Only the inner grid nodes are signalised (the fringe dead-ends stay
    unsignalised). Signal phases use the configured layout (split phasing).
    """
    files = network_files(cfg, name)
    files["net"].parent.mkdir(parents=True, exist_ok=True)
    ng = cfg["networks"][name]["netgenerate"]
    sig = cfg["signals"]
    cmd = [sumo_binary("netgenerate")]
    for key, val in ng.items():
        if val is True:
            cmd.append(f"--{key}")
        elif val is not False and val is not None:
            cmd += [f"--{key}", str(val)]
    cmd += [
        "--default-junction-type", "priority",
        "--tls.set", ",".join(grid_tls_ids(ng)),
        "--tls.layout", sig["layout"],
        "--tls.cycle.time", str(sig["default_cycle_s"]),
        "--tls.yellow.time", str(sig["yellow_s"]),
        "--tls.allred.time", str(sig["all_red_s"]),
        "--tls.min-dur", str(sig["min_green_s"]),
        "--tls.max-dur", str(sig["max_green_s"]),
        "--seed", "42",
        "--output-file", str(files["net"]),
    ]
    log.info("netgenerate: %s", " ".join(cmd[1:]))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"netgenerate failed:\n{res.stderr}")
    return files["net"]


def read_net(path: Path):
    """Read a SUMO network with sumolib (including internal lanes info)."""
    find_sumo_home()
    import sumolib

    return sumolib.net.readNet(str(path), withPrograms=True, withInternal=False)


def _foes(net_path: Path) -> dict[str, list[str]]:
    """Return ``{junction_id: foes_bitstrings}`` from the junction request table."""
    import xml.etree.ElementTree as ET

    out: dict[str, list[str]] = {}
    for j in ET.parse(net_path).getroot().iter("junction"):
        if j.get("type") != "traffic_light":
            continue
        reqs = sorted(j.findall("request"), key=lambda r: int(r.get("index")))
        out[j.get("id")] = [r.get("foes") for r in reqs]
    return out


def validate_network(net_path: Path, tls_ids: list[str]) -> None:
    """Check that every signalised junction is 4-leg and fully protected.

    Raises:
        AssertionError: on any violation.
    """
    net = read_net(net_path)
    foes = _foes(net_path)
    found = sorted(t.getID() for t in net.getTrafficLights())
    assert found == sorted(tls_ids), f"signalised junctions {found} != expected {tls_ids}"
    for tid in tls_ids:
        node = net.getNode(tid)
        n_in, n_out = len(node.getIncoming()), len(node.getOutgoing())
        assert n_in == 4 and n_out == 4, f"{tid}: {n_in} in / {n_out} out legs (expected 4/4)"
        tls = net.getTLS(tid)
        prog = next(iter(tls.getPrograms().values()))
        f = foes[tid]
        n = len(f)
        for k, ph in enumerate(prog.getPhases()):
            green = [i for i, c in enumerate(ph.state) if c in "Gg"]
            for a in green:
                for b in green:
                    # foes string is written with the highest link index first
                    if a != b and f[a][n - 1 - b] == "1":
                        raise AssertionError(
                            f"{tid} phase {k} ({ph.state}): conflicting greens on links {a},{b}"
                        )
            assert "g" not in ph.state, f"{tid} phase {k} has permissive (g) movements"
    log.info("validated %d signalised junctions: 4-leg, protected phases", len(tls_ids))


def detector_records(net_path: Path, cfg: dict[str, Any]) -> list[DetectorRecord]:
    """Build E1 + E2 detector records for every incoming lane of every TLS."""
    net = read_net(net_path)
    setback = float(cfg["detectors"]["e1"]["setback_m"])
    recs: list[DetectorRecord] = []
    for tls in sorted(net.getTrafficLights(), key=lambda t: t.getID()):
        tid = tls.getID()
        prog = next(iter(tls.getPrograms().values()))
        phases = prog.getPhases()
        lane_links: dict[str, set[int]] = {}
        for in_lane, _out_lane, link_idx in tls.getConnections():
            lane_links.setdefault(in_lane.getID(), set()).add(link_idx)
        for lane_id in sorted(lane_links):
            lane = net.getLane(lane_id)
            edge = lane.getEdge()
            length = lane.getLength()
            green = [
                str(k) for k, ph in enumerate(phases)
                if any(ph.state[i] in "Gg" for i in lane_links[lane_id])
            ]
            common = dict(
                lane=lane_id, edge=edge.getID(), junction=tid,
                approach_from=edge.getFromNode().getID(), phases=";".join(green),
                lane_length=round(length, 2),
            )
            recs.append(DetectorRecord(E1_PREFIX + lane_id, "E1", pos=round(length - setback, 2), **common))
            recs.append(DetectorRecord(E2_PREFIX + lane_id, "E2", pos=0.0, **common))
    return recs


def write_detectors(recs: list[DetectorRecord], cfg: dict[str, Any], det_path: Path,
                    map_path: Path, out_prefix: str = "") -> None:
    """Write ``<net>.det.add.xml`` and the detector mapping CSV.

    E1/E2 output files are written relative to the SUMO working directory as
    ``e1.xml`` / ``e2.xml`` (the runner sets the cwd to the run folder).
    """
    e1c, e2c = cfg["detectors"]["e1"], cfg["detectors"]["e2"]
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             "<!-- Generated by scripts/build_network.py. E1 = controller input, "
             "E2 = evaluation only. -->",
             "<additional>"]
    for r in recs:
        if r.det_type == "E1":
            lines.append(
                f'    <inductionLoop id={quoteattr(r.detector_id)} lane={quoteattr(r.lane)} '
                f'pos="{r.pos}" period="{e1c["period_s"]}" file="{out_prefix}e1.xml" friendlyPos="true"/>'
            )
        else:
            lines.append(
                f'    <laneAreaDetector id={quoteattr(r.detector_id)} lane={quoteattr(r.lane)} '
                f'pos="0" endPos="{r.lane_length}" period="{e2c["period_s"]}" '
                f'file="{out_prefix}e2.xml" friendlyPos="true" '
                f'timeThreshold="1" speedThreshold="{e2c["halting_speed_mps"]}" '
                f'jamThreshold="{e2c["jam_threshold_m"]}"/>'
            )
    lines.append("</additional>")
    det_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(map_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(DetectorRecord.__dataclass_fields__))
        w.writeheader()
        for r in recs:
            w.writerow(r.__dict__)
    log.info("wrote %d detectors -> %s, mapping -> %s", len(recs), det_path.name, map_path.name)


def load_mapping(map_path: Path) -> list[dict[str, str]]:
    """Read the detector mapping CSV as a list of dicts."""
    with open(map_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_network(net_path: Path, recs: list[DetectorRecord], png_path: Path, title: str) -> None:
    """Save a PNG of the network layout with signals and detectors marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = read_net(net_path)
    tls_ids = {t.getID() for t in net.getTrafficLights()}
    e2_lanes = {r.lane for r in recs if r.det_type == "E2"}
    fig, ax = plt.subplots(figsize=(9, 9 if len(tls_ids) > 5 else 4.5))
    for edge in net.getEdges():
        for lane in edge.getLanes():
            xs, ys = zip(*lane.getShape())
            monitored = lane.getID() in e2_lanes
            ax.plot(xs, ys, color="#2a7ab0" if monitored else "#9a9a9a",
                    lw=2.2 if monitored else 1.6, solid_capstyle="butt", zorder=1)
    for r in recs:
        if r.det_type == "E1":
            x, y = net.getLane(r.lane).getShape()[-1]
            sx, sy = net.getLane(r.lane).getShape()[-2]
            # place the marker `setback` metres upstream of the stop line
            L = max(((x - sx) ** 2 + (y - sy) ** 2) ** 0.5, 1e-9)
            d = r.lane_length - r.pos
            ax.plot(x - (x - sx) * d / L, y - (y - sy) * d / L, "s", ms=3.5, color="#d1495b", zorder=3)
    for node in net.getNodes():
        x, y = node.getCoord()
        if node.getID() in tls_ids:
            ax.plot(x, y, "o", ms=16, mfc="#f2c14e", mec="#333", zorder=2)
            ax.annotate(node.getID(), (x, y), ha="center", va="center", fontsize=8, weight="bold", zorder=4)
        else:
            ax.plot(x, y, "o", ms=4, color="#555", zorder=2)
            ax.annotate(node.getID(), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6, color="#555")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color="#2a7ab0", lw=2.2, label="incoming lane (full-length E2, evaluation only)"),
        Line2D([], [], color="#9a9a9a", lw=1.6, label="other lane"),
        Line2D([], [], marker="s", ls="", color="#d1495b", label="E1 stop-line loop (controller input)"),
        Line2D([], [], marker="o", ls="", mfc="#f2c14e", mec="#333", ms=10, label="signalised junction"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("saved network plot -> %s", png_path)

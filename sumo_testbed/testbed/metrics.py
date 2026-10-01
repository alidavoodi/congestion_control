"""Run metrics: TTS, delay, throughput, unserved demand, queues, spillback, gridlock.

Evaluation uses E2 lane-area detectors (true queues). Controllers never see
these values; they are only read here.

Definitions (analysis window [t0, t1)):
  * TTS [veh.h] = sum over steps of (vehicles in network + insertion backlog) x dt.
  * Mean delay [s] = mean tripinfo ``timeLoss`` of vehicles whose *desired*
    departure (depart - departDelay) lies in the window, including vehicles
    still unfinished at the end (``write-unfinished``), so oversaturated runs
    are less biased towards the vehicles that got through. Vehicles never
    inserted have no tripinfo record; their cost appears in TTS (insertion
    backlog) and in unserved demand.
  * Throughput [veh/h] = arrivals in the window / window length.
  * Unserved demand = vehicles generated but not arrived when the run ends
    (= not inserted + still in network).
  * Spillback event: a lane's E2 jam length >= ratio x lane length for at
    least ``min_duration`` s continuously. Counted if it overlaps the window;
    durations are clipped to the window.
  * Gridlock: network mean speed < threshold for > min_duration while vehicles
    are present.
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Event:
    """A spillback or gridlock episode."""

    key: str  # lane id ('network' for gridlock)
    junction: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class RunTracker:
    """Detect continuous runs where a boolean condition holds for >= min_duration.

    Call :meth:`update` once per step with the step's end time; a step at time
    t represents the interval (t - dt, t].
    """

    def __init__(self, dt: float, min_duration: float) -> None:
        self.dt = dt
        self.min_duration = min_duration
        self._open: dict[str, float] = {}
        self._junction: dict[str, str] = {}
        self.events: list[Event] = []

    def update(self, key: str, t: float, flag: bool, junction: str = "") -> None:
        if flag:
            if key not in self._open:
                self._open[key] = t - self.dt
                self._junction[key] = junction
        elif key in self._open:
            self._close(key, t - self.dt)

    def _close(self, key: str, end: float) -> None:
        start = self._open.pop(key)
        if end - start >= self.min_duration - 1e-9:
            self.events.append(Event(key, self._junction.get(key, ""), start, end))

    def finalize(self, t: float) -> list[Event]:
        for key in list(self._open):
            self._close(key, t)
        return self.events


def clip_events(events: list[Event], t0: float, t1: float) -> list[Event]:
    """Events overlapping [t0, t1), with start/end clipped to the window."""
    return [Event(e.key, e.junction, max(e.start, t0), min(e.end, t1))
            for e in events if e.end > t0 and e.start < t1]


class MetricsCollector:
    """Per-step metric collection through TraCI subscriptions.

    Args:
        sim: A started :class:`testbed.sim.SumoSim`.
        cfg: Experiment config.
        e2: Rows of the detector mapping with det_type == "E2".
        window: Analysis window (t0, t1) in seconds.
    """

    def __init__(self, sim, cfg: dict[str, Any], e2: list[dict[str, str]],
                 window: tuple[float, float]) -> None:
        import traci.constants as tc

        self.sim, self.tr, self.tc = sim, sim.traci, tc
        m = cfg["metrics"]
        self.dt = float(sim.spec.step_length)
        self.window = window
        self.sample = float(m["queue_sample_s"])
        self.e2 = {r["detector_id"]: r for r in e2}
        self.thresh = {d: float(m["spillback_ratio"]) * float(r["lane_length"]) for d, r in self.e2.items()}
        self.spill = RunTracker(self.dt, float(m["spillback_min_duration_s"]))
        self.grid = RunTracker(self.dt, float(m["gridlock_min_duration_s"]))
        self.grid_speed = float(m["gridlock_speed_mps"])
        self.tts_vs = 0.0  # veh.s
        self.max_backlog = 0
        self.queue_rows: list[list[float]] = []
        self.system_rows: list[dict[str, float]] = []
        for d in self.e2:
            self.tr.lanearea.subscribe(d, [tc.JAM_LENGTH_METERS])
        self.edges = [e for e in self.tr.edge.getIDList() if not e.startswith(":")]
        for e in self.edges:
            self.tr.edge.subscribe(e, [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_VEHICLE_NUMBER])

    def record(self) -> None:
        """Collect one step (call right after ``sim.step()``)."""
        t = self.sim.time
        tc = self.tc
        n_net, backlog = self.sim.vehicles_in_network(), self.sim.insertion_backlog()
        if self.window[0] < t <= self.window[1]:
            self.tts_vs += (n_net + backlog) * self.dt
        self.max_backlog = max(self.max_backlog, backlog)
        jam = {d: self.tr.lanearea.getSubscriptionResults(d)[tc.JAM_LENGTH_METERS] for d in self.e2}
        for d, j in jam.items():
            r = self.e2[d]
            self.spill.update(r["lane"], t, j >= self.thresh[d], r["junction"])
        num = den = 0.0
        for e in self.edges:
            res = self.tr.edge.getSubscriptionResults(e)
            n = res[tc.LAST_STEP_VEHICLE_NUMBER]
            num += res[tc.LAST_STEP_MEAN_SPEED] * n
            den += n
        speed = num / den if den else float("nan")
        self.grid.update("network", t, den > 0 and speed < self.grid_speed)
        if abs(t / self.sample - round(t / self.sample)) < 1e-6:
            self.queue_rows.append([t] + [jam[d] for d in self.e2])
            self.system_rows.append(dict(time=t, in_network=n_net, backlog=backlog,
                                         mean_speed=speed,
                                         arrived=self.tr.simulation.getArrivedNumber()))

    def finalize(self, out_dir: Path) -> dict[str, Any]:
        """Close open episodes, write time series and return collector metrics."""
        t_end = self.sim.time
        t0, t1 = self.window
        spill = clip_events(self.spill.finalize(t_end), t0, t1)
        grid = clip_events(self.grid.finalize(t_end), t0, t1)
        lanes = [self.e2[d]["lane"] for d in self.e2]
        q = pd.DataFrame(self.queue_rows, columns=["time"] + lanes)
        q.to_csv(out_dir / "queues.csv", index=False)
        pd.DataFrame(self.system_rows).to_csv(out_dir / "system.csv", index=False)
        pd.DataFrame([asdict(e) | {"duration": e.duration} for e in spill],
                     columns=["key", "junction", "start", "end", "duration"]).rename(
            columns={"key": "lane"}).to_csv(out_dir / "spillbacks.csv", index=False)
        qw = q[(q["time"] > t0) & (q["time"] <= t1)]
        return {
            "tts_veh_h": self.tts_vs / 3600.0,
            "max_insertion_backlog": int(self.max_backlog),
            "spillback_count": len(spill),
            "spillback_duration_s": float(sum(e.duration for e in spill)),
            "spillback_lanes": sorted({e.key for e in spill}),
            "spillback_junctions": sorted({e.junction for e in spill}),
            "gridlock": bool(grid),
            "gridlock_duration_s": float(sum(e.duration for e in grid)),
            "queue_mean_m": float(qw[lanes].to_numpy().mean()) if len(qw) else 0.0,
            "queue_max_m": float(qw[lanes].to_numpy().max()) if len(qw) else 0.0,
            "sim_end_s": t_end,
        }


def tripinfo_metrics(tripinfo: Path, window: tuple[float, float]) -> dict[str, Any]:
    """Delay / travel time / throughput from a tripinfo file (see module docstring)."""
    t0, t1 = window
    rows = []
    for _ev, el in ET.iterparse(tripinfo, events=("end",)):
        if el.tag == "tripinfo":
            rows.append((float(el.get("depart")), float(el.get("departDelay", 0)),
                         float(el.get("arrival")), float(el.get("duration")),
                         float(el.get("timeLoss"))))
            el.clear()
    a = np.array(rows, dtype=float).reshape(-1, 5)
    depart, ddelay, arrival, duration, tloss = a.T
    finished = arrival >= 0
    in_win = ((depart - ddelay) >= t0) & ((depart - ddelay) < t1)
    arr_win = finished & (arrival >= t0) & (arrival < t1)

    def mean(x: np.ndarray) -> float:
        return float(x.mean()) if x.size else float("nan")

    return {
        "vehicles_in_window": int(in_win.sum()),
        "mean_delay_s": mean(tloss[in_win]),
        "mean_delay_finished_s": mean(tloss[in_win & finished]),
        "mean_depart_delay_s": mean(ddelay[in_win]),
        "mean_travel_time_s": mean(duration[in_win & finished]),
        "throughput_veh": int(arr_win.sum()),
        "throughput_vph": float(arr_win.sum() / ((t1 - t0) / 3600.0)),
        "arrived_total": int(finished.sum()),
    }


def statistics_metrics(stats: Path) -> dict[str, Any]:
    """Loaded/inserted/running/waiting counts and teleports from SUMO's statistic output."""
    root = ET.parse(stats).getroot()
    v, tp = root.find("vehicles"), root.find("teleports")
    loaded, inserted = int(v.get("loaded")), int(v.get("inserted"))
    running, waiting = int(v.get("running")), int(v.get("waiting"))
    return {
        "loaded": loaded, "inserted": inserted,
        "unserved_not_inserted": loaded - inserted,
        "unserved_in_network": running,
        "unserved_total": loaded - inserted + running,
        "teleports": int(tp.get("total")),
    }


def write_metrics(metrics: dict[str, Any], out_dir: Path) -> Path:
    """Write ``metrics.json``."""
    p = out_dir / "metrics.json"
    p.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    return p

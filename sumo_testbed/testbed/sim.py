"""SUMO/TraCI wrapper: start, step, close, with per-run outputs.

Each run gets its own output folder. Detector definitions are copied into
that folder with output paths rewritten, so parallel runs never share files.
Use ``use_libsumo=True`` for speed (no GUI, same API).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from testbed.sumo_env import find_sumo_home, sumo_binary

log = logging.getLogger(__name__)


class TeleportError(AssertionError):
    """Raised when a run produced teleports (forbidden: gridlock must stay visible)."""


@dataclass
class SimSpec:
    """Everything needed to (re)start one SUMO run."""

    net_file: Path
    route_file: Path
    det_file: Path
    tls_file: Path | None
    out_dir: Path
    seed: int
    step_length: float = 1.0
    gui: bool = False
    use_libsumo: bool = False
    extra_additional: list[Path] = field(default_factory=list)  # e.g. Phase 1 incident files


def ensure_display() -> None:
    """Make sure sumo-gui can reach an X server (XQuartz on macOS).

    Without this check a GUI that dies on start-up makes TraCI retry forever.

    Raises:
        RuntimeError: with installation/start instructions if no display is available.
    """
    if sys.platform != "darwin":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise RuntimeError("sumo-gui needs a display but DISPLAY is not set.")
        return
    if not Path("/Applications/Utilities/XQuartz.app").exists():
        raise RuntimeError("sumo-gui needs XQuartz: `brew install --cask xquartz`, then log out/in.")
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
    if os.environ["DISPLAY"].startswith(":") and not Path("/tmp/.X11-unix/X0").exists():
        subprocess.run(["open", "-a", "XQuartz"], check=False)
        for _ in range(20):
            if Path("/tmp/.X11-unix/X0").exists():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("XQuartz is not running. Start it with `open -a XQuartz` and retry.")


def import_traci(use_libsumo: bool):
    """Return the ``traci`` or ``libsumo`` module (same API)."""
    find_sumo_home()
    if use_libsumo:
        import libsumo as traci_mod
    else:
        import traci as traci_mod
    return traci_mod


class SumoSim:
    """Thin wrapper around one SUMO process.

    Attributes:
        traci: The TraCI-compatible module for this run (``traci`` or ``libsumo``).
        teleports: Number of teleports observed so far.
    """

    def __init__(self, spec: SimSpec, cfg: dict[str, Any]) -> None:
        self.spec = spec
        self.cfg = cfg
        self.traci = import_traci(spec.use_libsumo and not spec.gui)
        self.teleports = 0
        self._running = False

    # ------------------------------------------------------------------ setup
    def _write_run_additionals(self) -> Path:
        """Copy detector definitions into the run folder and add edgedata output."""
        out = self.spec.out_dir
        txt = self.spec.det_file.read_text(encoding="utf-8")
        txt = re.sub(r'file="(e[12]\.xml)"', lambda m: f'file="{out / m.group(1)}"', txt)
        period = self.cfg["sumo"]["edgedata_period_s"]
        txt = txt.replace(
            "</additional>",
            f'    <edgeData id="edgedata" period="{period}" file="{out / "edgedata.xml"}" '
            f'excludeEmpty="true"/>\n</additional>',
        )
        p = out / "detectors.add.xml"
        p.write_text(txt, encoding="utf-8")
        return p

    def command(self) -> list[str]:
        """Build the SUMO command line for this run."""
        s, sc = self.spec, self.cfg["sumo"]
        s.out_dir.mkdir(parents=True, exist_ok=True)
        adds = [self._write_run_additionals()]
        if s.tls_file is not None:
            adds.append(s.tls_file)
        adds += s.extra_additional
        out = s.out_dir
        cmd = [
            sumo_binary("sumo-gui" if s.gui else "sumo"),
            "--net-file", str(s.net_file),
            "--route-files", str(s.route_file),
            "--additional-files", ",".join(str(a) for a in adds),
            "--step-length", str(s.step_length),
            "--time-to-teleport", str(sc["time_to_teleport"]),
            "--collision.action", sc["collision_action"],
            "--seed", str(s.seed),
            "--tripinfo-output", str(out / "tripinfo.xml"),
            "--tripinfo-output.write-unfinished",
            "--summary-output", str(out / "summary.xml"),
            "--statistic-output", str(out / "statistics.xml"),
            "--log", str(out / "sumo.log"),
        ]
        cmd += list(sc.get("extra_args", []))
        if s.gui:
            cmd += ["--start", "--delay", str(sc["gui_delay_ms"])]
        return cmd

    def start(self) -> None:
        """Launch SUMO and connect TraCI."""
        if self.spec.gui:
            ensure_display()
        cmd = self.command()
        log.debug("SUMO: %s", " ".join(cmd))
        self.traci.start(cmd)
        self._running = True

    # ------------------------------------------------------------------- loop
    @property
    def time(self) -> float:
        """Current simulation time [s]."""
        return float(self.traci.simulation.getTime())

    def step(self) -> None:
        """Advance one simulation step and count teleports."""
        self.traci.simulationStep()
        self.teleports += self.traci.simulation.getStartingTeleportNumber()

    def vehicles_in_network(self) -> int:
        """Vehicles currently running."""
        return int(self.traci.vehicle.getIDCount())

    def insertion_backlog(self) -> int:
        """Vehicles that should have departed but are still waiting to be inserted."""
        return len(self.traci.simulation.getPendingVehicles())

    def remaining_demand(self) -> int:
        """Vehicles running, pending insertion or still to be loaded."""
        return int(self.traci.simulation.getMinExpectedNumber())

    def close(self, check_teleports: bool = True) -> None:
        """Close the connection and assert that no vehicle teleported."""
        if self._running:
            self.traci.close()
            self._running = False
        if check_teleports and self.teleports != 0:
            raise TeleportError(f"{self.teleports} teleports in {self.spec.out_dir} (must be 0)")

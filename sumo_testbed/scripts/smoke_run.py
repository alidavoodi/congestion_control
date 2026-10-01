#!/usr/bin/env python
"""Short visual check run (default 5 min) with the default signal program.

Saves snapshots of vehicle positions and signal states to
``results/smoke/<net>/snapshots.png``. With ``--gui`` opens sumo-gui instead
(requires an X server, e.g. XQuartz on macOS).
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from testbed.config import load_config, network_files, project_path, tls_file
from testbed.demand import constant_demand
from testbed.log import setup_logging
from testbed.network import read_net
from testbed.sim import SimSpec, SumoSim

log = logging.getLogger("smoke_run")


def snapshot(sim: SumoSim) -> dict:
    """Vehicle positions/speeds and TLS link states at the current step."""
    tr = sim.traci
    veh = [(tr.vehicle.getPosition(v), tr.vehicle.getSpeed(v)) for v in tr.vehicle.getIDList()]
    tls = {t: tr.trafficlight.getRedYellowGreenState(t) for t in tr.trafficlight.getIDList()}
    return {"t": sim.time, "veh": veh, "tls": tls, "backlog": sim.insertion_backlog()}


def plot_snapshots(net_path, snaps: list[dict], png) -> None:
    """Render snapshot panels: lanes coloured by signal state, vehicles by speed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = read_net(net_path)
    fig, axes = plt.subplots(1, len(snaps), figsize=(5.2 * len(snaps), 5.6))
    colour = {"G": "#2ca02c", "g": "#2ca02c", "y": "#f2c14e", "r": "#d62728"}
    sc = None
    for ax, s in zip(axes, snaps):
        for e in net.getEdges():
            for ln in e.getLanes():
                xs, ys = zip(*ln.getShape())
                ax.plot(xs, ys, color="#cfcfcf", lw=1.4, zorder=1)
        for tls in net.getTrafficLights():
            state = s["tls"].get(tls.getID(), "")
            for in_lane, _o, idx in tls.getConnections():
                x, y = in_lane.getShape()[-1]
                ax.plot(x, y, "o", ms=3.5, color=colour.get(state[idx], "k") if idx < len(state) else "k", zorder=3)
        if s["veh"]:
            (pts, spd) = zip(*s["veh"])
            xs, ys = zip(*pts)
            sc = ax.scatter(xs, ys, c=spd, cmap="viridis", vmin=0, vmax=14, s=9, zorder=2)
        ax.set_title(f"t = {s['t']:.0f} s | {len(s['veh'])} veh in net | backlog {s['backlog']}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    if sc is not None:
        fig.colorbar(sc, ax=axes, shrink=0.7, label="vehicle speed [m/s]")
    fig.suptitle("Smoke run: dots at stop lines = signal state (green/yellow/red)")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--net", default="grid3x3")
    ap.add_argument("--multiplier", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--program", default="default", help="signal program (tls file suffix)")
    ap.add_argument("--duration", type=int, default=None, help="simulated seconds (default: smoke_test.duration_s)")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = project_path(cfg, "results_dir").parent / "smoke" / args.net
    setup_logging(out / "smoke.log")
    mult = args.multiplier if args.multiplier is not None else cfg["demand"]["smoke_multiplier"]
    if args.net != cfg["demand"]["network"]:
        raise SystemExit(f"Demand (configs/demand/od_base.csv) is defined for "
                         f"{cfg['demand']['network']} only; --net {args.net} is not supported yet.")
    files = network_files(cfg, args.net)
    tf = tls_file(cfg, args.net, args.program)
    spec = SimSpec(files["net"], constant_demand(cfg, mult), files["det"],
                   tf if tf.exists() else None, out, args.seed,
                   step_length=cfg["sumo"]["step_length"], gui=args.gui,
                   use_libsumo=cfg["sumo"]["use_libsumo"])
    log.info("smoke run %s, multiplier %.2f, seed %d, program %s%s", args.net, mult, args.seed,
             args.program, "" if tf.exists() else " (net built-in)")
    sim = SumoSim(spec, cfg)
    sim.start()
    dur = args.duration or cfg["smoke_test"]["duration_s"]
    marks = {dur // 5, dur // 2, dur}
    snaps = []
    try:
        while sim.time < dur:
            sim.step()
            if round(sim.time) in marks:
                snaps.append(snapshot(sim))
    except sim.traci.exceptions.FatalTraCIError:
        if not args.gui:
            raise
        log.info("GUI window closed - run stopped after %d teleports", sim.teleports)
        return
    n_net, backlog = sim.vehicles_in_network(), sim.insertion_backlog()
    sim.close()
    log.info("done at t=%.0f s: teleports=%d, in network=%d, insertion backlog=%d", dur,
             sim.teleports, n_net, backlog)
    if not args.gui:
        plot_snapshots(files["net"], snaps, out / "snapshots.png")
        log.info("snapshots -> %s", out / "snapshots.png")


if __name__ == "__main__":
    main()

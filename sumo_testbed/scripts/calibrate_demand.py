#!/usr/bin/env python
"""Demand sweep under the default fixed-time signals to find the capacity "knee".

For every multiplier x seed a constant-demand run is executed (no teleporting);
metrics are aggregated, plotted, the knee is chosen with the rule documented in
``configs/phase0.yaml`` (calibration.knee_slope_factor) and the demand levels
are written to ``configs/demand_levels.yaml``.

Usage:
    python scripts/calibrate_demand.py [--workers 4] [--force] [--knee 0.8]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from testbed.config import PROJECT_ROOT, load_config, network_files, project_path, tls_file
from testbed.demand import analysis_window, constant_demand, profile_end
from testbed.log import setup_logging
from testbed.runner import RunSpec, run
from testbed.sim import SimSpec

log = logging.getLogger("calibrate_demand")


def run_dir(cfg: dict[str, Any], m: float, seed: int) -> Path:
    """Output folder of one calibration run."""
    return project_path(cfg, "calibration_dir") / f"m{m:.2f}" / f"seed_{seed}"


def one_run(args: tuple[dict[str, Any], float, int]) -> dict[str, Any]:
    """Worker: run (multiplier, seed) and return its metrics."""
    cfg, m, seed = args
    c = cfg["calibration"]
    net = c["network"]
    f = network_files(cfg, net)
    tf = tls_file(cfg, net, c["controller"])
    prof = c["profile"]
    spec = RunSpec(
        sim=SimSpec(f["net"], constant_demand(cfg, m), f["det"], tf if tf.exists() else None,
                    run_dir(cfg, m, seed), seed, step_length=cfg["sumo"]["step_length"],
                    use_libsumo=cfg["sumo"]["use_libsumo"]),
        window=analysis_window(cfg, prof), insertion_end=profile_end(cfg, prof),
        mapping_file=f["mapping"], label={"network": net, "controller": c["controller"], "multiplier": m},
    )
    setup_logging(run_dir(cfg, m, seed) / "run.log")
    return run(spec, cfg)


def find_knee(ms: np.ndarray, delay: np.ndarray, factor: float) -> tuple[float | None, pd.DataFrame]:
    """First m_i whose marginal delay slope exceeds ``factor`` x the average slope below it.

    slope_i = (d_i - d_{i-1}) / (m_i - m_{i-1});  avg_i = (d_{i-1} - d_0) / (m_{i-1} - m_0),
    evaluated from i = 2 so the average uses at least two points.
    """
    rows, knee = [], None
    for i in range(1, len(ms)):
        slope = (delay[i] - delay[i - 1]) / (ms[i] - ms[i - 1])
        avg = (delay[i - 1] - delay[0]) / (ms[i - 1] - ms[0]) if i >= 2 else np.nan
        hit = i >= 2 and avg > 0 and slope > factor * avg
        rows.append(dict(multiplier=ms[i], marginal_slope=slope, avg_slope_below=avg, exceeds=hit))
        if hit and knee is None:
            knee = float(ms[i])
    return knee, pd.DataFrame(rows)


def throughput_saturation(ms: np.ndarray, thr: np.ndarray, offered: np.ndarray) -> float | None:
    """Secondary indicator: first m_i where the throughput gain is < 50% of the offered-demand gain."""
    for i in range(1, len(ms)):
        if (thr[i] - thr[i - 1]) < 0.5 * (offered[i] - offered[i - 1]):
            return float(ms[i])
    return None


def plot(summary: pd.DataFrame, runs: pd.DataFrame, knee: float | None, thr_sat: float | None,
         levels: dict[str, float], png: Path) -> None:
    """Three panels: mean delay, throughput (with offered demand) and spillbacks vs multiplier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, series, ref = "#1f2328", "#6e7781", "#e6e8eb", "#2a6fb0", "#9aa1a9"
    panels = [("mean_delay_s", "Mean delay per vehicle [s]"),
              ("throughput_vph", "Throughput in analysis hour [veh/h]"),
              ("spillback_count", "Spillback events in analysis hour")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, title) in zip(axes, panels):
        ax.scatter(runs["multiplier"], runs[col], s=14, color=series, alpha=0.35, lw=0, zorder=2)
        ax.plot(summary["multiplier"], summary[f"{col}_mean"], color=series, lw=2, marker="o", ms=5,
                zorder=3, label="mean of seeds")
        if col == "throughput_vph":
            ax.plot(summary["multiplier"], summary["offered_vph"], color=ref, lw=1.5, ls="--",
                    zorder=1, label="offered demand")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        if knee is not None:
            ax.axvline(knee, color=ink, lw=1, ls=":", zorder=1)
            ax.annotate(f"knee {knee:.2f}", (knee, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points", fontsize=8, color=ink)
        for name, v in levels.items():
            ax.axvline(v, color=ref, lw=0.8, zorder=0)
            ax.annotate(name, (v, 0), xycoords=("data", "axes fraction"), xytext=(3, 4),
                        textcoords="offset points", fontsize=7, color=muted, rotation=90)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlabel("demand multiplier (1.0 = 300 veh/h per entry)", fontsize=9, color=muted)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    sub = f"knee rule: marginal delay slope > factor x avg slope below" + (
        f"  |  throughput saturation indicator: {thr_sat:.2f}" if thr_sat else "")
    fig.suptitle("Capacity calibration - grid3x3, default fixed-time signals, constant demand\n" + sub,
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run even if metrics.json exists")
    ap.add_argument("--knee", type=float, default=None, help="override the automatically chosen knee")
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["calibration"]
    out = project_path(cfg, "calibration_dir")
    setup_logging(project_path(cfg, "logs_dir") / "calibrate_demand.log")

    ms = [float(m) for m in c["multipliers"]]
    for m in ms:  # build demand files once, before forking workers
        constant_demand(cfg, m)
    todo = [(cfg, m, s) for m in ms for s in c["seeds"]
            if args.force or not (run_dir(cfg, m, s) / "metrics.json").exists()]
    log.info("%d runs to do (%d cached), %d workers", len(todo),
             len(ms) * len(c["seeds"]) - len(todo), args.workers or c["workers"])
    if todo:
        with mp.get_context("spawn").Pool(args.workers or c["workers"]) as pool:
            for k, r in enumerate(pool.imap_unordered(one_run, todo), 1):
                log.info("[%d/%d] m=%.2f seed=%d delay=%.0fs thr=%.0f spill=%d gridlock=%s",
                         k, len(todo), r["multiplier"], r["seed"], r["mean_delay_s"],
                         r["throughput_vph"], r["spillback_count"], r["gridlock"])
        setup_logging(project_path(cfg, "logs_dir") / "calibrate_demand.log")

    runs = pd.DataFrame([json.loads((run_dir(cfg, m, s) / "metrics.json").read_text())
                         for m in ms for s in c["seeds"]])
    runs = runs.drop(columns=["spillback_lanes", "spillback_junctions", "window"])
    runs.to_csv(out / "calibration_runs.csv", index=False)
    cols = ["mean_delay_s", "throughput_vph", "spillback_count", "spillback_duration_s", "tts_veh_h",
            "unserved_total", "max_insertion_backlog", "gridlock"]
    g = runs.groupby("multiplier")[cols]
    summary = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
    summary["offered_vph"] = summary["multiplier"] * 12 * cfg["demand"]["base_origin_flow_vph"]

    knee, diag = find_knee(summary["multiplier"].to_numpy(), summary["mean_delay_s_mean"].to_numpy(),
                           float(c["knee_slope_factor"]))
    thr_sat = throughput_saturation(summary["multiplier"].to_numpy(),
                                    summary["throughput_vph_mean"].to_numpy(),
                                    summary["offered_vph"].to_numpy())
    summary = summary.merge(diag, on="multiplier", how="left")
    summary.to_csv(out / "calibration_summary.csv", index=False)
    chosen = args.knee if args.knee is not None else knee
    if chosen is None:
        raise SystemExit("No knee found by the rule; inspect the plot and pass --knee.")
    levels = {k: round(f * chosen, 3) for k, f in c["level_factors"].items()}
    plot(summary, runs, chosen, thr_sat, levels, out / "calibration.png")

    doc = {
        "knee": chosen,
        "knee_rule": f"first multiplier whose marginal delay slope exceeds "
                     f"{c['knee_slope_factor']}x the average slope below it"
                     + (" (overridden manually)" if args.knee is not None else ""),
        "knee_by_rule": knee,
        "throughput_saturation_indicator": thr_sat,
        "base_origin_flow_vph": cfg["demand"]["base_origin_flow_vph"],
        "level_factors": c["level_factors"],
        "levels": levels,
        "created": dt.date.today().isoformat(),
    }
    lp = PROJECT_ROOT / cfg["paths"]["demand_levels_file"]
    lp.write_text("# Written by scripts/calibrate_demand.py\n" + yaml.safe_dump(doc, sort_keys=False))
    show = ["multiplier", "mean_delay_s_mean", "throughput_vph_mean", "offered_vph",
            "spillback_count_mean", "unserved_total_mean", "gridlock_mean", "marginal_slope",
            "avg_slope_below", "exceeds"]
    log.info("\n%s", summary[show].round(2).to_string(index=False))
    log.info("knee (rule) = %s, chosen = %s, throughput saturation = %s", knee, chosen, thr_sat)
    log.info("levels = %s -> %s", levels, lp)
    log.info("plot -> %s", out / "calibration.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate frozen signal plans on demand scenarios (no re-optimisation).

Runs every plan x scenario x seed, then writes a summary (mean +/- 95% CI over
seeds) as CSV and Markdown, plus bar plots per scenario.

Usage:
    python scripts/evaluate_plans.py --plans default webster --scenarios normal_peak --seeds 1 2 3 --tag d1_train
    python scripts/evaluate_plans.py --tag validation          # defaults from configs/signals.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from testbed.config import PROJECT_ROOT, load_config
from testbed.log import setup_logging
from testbed.runner import run
from testbed.scenarios import make_runspec
from testbed.signals import SIGNALS_CONFIG, load_signals_config, plan_file

log = logging.getLogger("evaluate_plans")

METRICS = [("tts_veh_h", "TTS [veh.h]"), ("mean_delay_s", "Mean delay [s]"),
           ("throughput_vph", "Throughput [veh/h]"), ("unserved_total", "Unserved [veh]"),
           ("spillback_count", "Spillbacks [#]"), ("spillback_duration_s", "Spillback duration [s]"),
           ("gridlock", "Gridlock [share of runs]")]


def run_dir(scfg: dict[str, Any], tag: str, scenario: str, plan: str, seed: int) -> Path:
    return PROJECT_ROOT / scfg["results_dir"] / "eval" / tag / scenario / plan / f"seed_{seed}"


def one(args) -> dict[str, Any]:
    cfg, scfg, tag, scenario, plan, seed = args
    out = run_dir(scfg, tag, scenario, plan, seed)
    setup_logging(out / "run.log")
    pf = plan_file(scfg, plan)
    return run(make_runspec(cfg, scenario, seed, pf, out, {"plan": plan}), cfg)


def t_ci(x: np.ndarray) -> float:
    """Half-width of the 95% t confidence interval of the mean."""
    x = x[~np.isnan(x)]
    if x.size < 2:
        return float("nan")
    try:
        from scipy.stats import t

        q = float(t.ppf(0.975, x.size - 1))
    except ImportError:  # t quantiles for small n
        q = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(x.size, 1.96)
    return q * float(x.std(ddof=1)) / np.sqrt(x.size)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sc, plan), g in runs.groupby(["scenario", "plan"], sort=False):
        r = {"scenario": sc, "plan": plan, "n": len(g)}
        for col, _ in METRICS:
            x = g[col].astype(float).to_numpy()
            r[f"{col}_mean"], r[f"{col}_ci95"] = float(np.mean(x)), t_ci(x)
        rows.append(r)
    return pd.DataFrame(rows)


def markdown(summary: pd.DataFrame) -> str:
    head = "| scenario | plan | n | " + " | ".join(lbl for _, lbl in METRICS) + " |"
    sep = "|" + "---|" * (3 + len(METRICS))
    lines = [head, sep]
    for _, r in summary.iterrows():
        cells = []
        for col, _ in METRICS:
            m, ci = r[f"{col}_mean"], r[f"{col}_ci95"]
            fmt = "{:.2f}" if col == "gridlock" else "{:.1f}"
            cells.append(fmt.format(m) + ("" if np.isnan(ci) else " ± " + fmt.format(ci)))
        lines.append(f"| {r['scenario']} | {r['plan']} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(summary: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#6e7781", "#e6e8eb"
    palette = ["#2a6fb0", "#e07b39", "#3a9e6b", "#a35bb5", "#c9a227"]  # fixed order per plan
    plans = list(dict.fromkeys(summary["plan"]))
    colour = {p: palette[i % len(palette)] for i, p in enumerate(plans)}
    show = METRICS[:6]
    for sc, g in summary.groupby("scenario", sort=False):
        fig, axes = plt.subplots(1, len(show), figsize=(3.1 * len(show), 3.6))
        for ax, (col, lbl) in zip(axes, show):
            x = np.arange(len(g))
            ax.bar(x, g[f"{col}_mean"], yerr=g[f"{col}_ci95"].fillna(0), width=0.7,
                   color=[colour[p] for p in g["plan"]], ecolor=ink, capsize=3, error_kw={"lw": 1})
            ax.set_xticks(x, g["plan"], rotation=30, ha="right", fontsize=8, color=muted)
            ax.set_title(lbl, fontsize=9, color=ink, loc="left")
            top = float((g[f"{col}_mean"] + g[f"{col}_ci95"].fillna(0)).max())
            ax.set_ylim(0, top * 1.1 if top > 0 else 1)
            ax.grid(axis="y", color=grid, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(grid)
            ax.tick_params(colors=muted, labelsize=8)
        fig.suptitle(f"{sc}: mean ± 95% CI over seeds", fontsize=10, color=ink, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"bars_{sc}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--signals-config", default=str(SIGNALS_CONFIG))
    ap.add_argument("--plans", nargs="*", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--tag", default="validation", help="results subfolder name")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg, scfg = load_config(args.config), load_signals_config(args.signals_config)
    v = scfg["validation"]
    plans, scenarios = args.plans or v["plans"], args.scenarios or v["scenarios"]
    seeds = args.seeds or v["held_out_seeds"]
    out = PROJECT_ROOT / scfg["results_dir"] / "eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "evaluate.log")
    for p in plans:
        if not plan_file(scfg, p).exists():
            raise SystemExit(f"{plan_file(scfg, p)} missing - build it first.")
    jobs = [(cfg, scfg, args.tag, sc, p, k) for sc in scenarios for p in plans for k in seeds]
    todo = [j for j in jobs if args.force or not (run_dir(scfg, *j[2:]) / "metrics.json").exists()]
    log.info("%d runs (%d cached)", len(todo), len(jobs) - len(todo))
    if todo:
        with mp.get_context("spawn").Pool(args.workers or v["workers"]) as pool:
            for k, r in enumerate(pool.imap_unordered(one, todo), 1):
                log.info("[%d/%d] %s %s seed %d: TTS %.0f delay %.0fs spill %d", k, len(todo),
                         r["scenario"], r["plan"], r["seed"], r["tts_veh_h"], r["mean_delay_s"],
                         r["spillback_count"])
        setup_logging(out / "evaluate.log")
    runs = pd.DataFrame([json.loads((run_dir(scfg, *j[2:]) / "metrics.json").read_text()) for j in jobs])
    runs["gridlock"] = runs["gridlock"].astype(float)
    runs.drop(columns=["spillback_lanes", "spillback_junctions", "window"]).to_csv(out / "runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(markdown(summary))
    plot(summary, out)
    log.info("\n%s", markdown(summary))
    log.info("-> %s", out)


if __name__ == "__main__":
    main()

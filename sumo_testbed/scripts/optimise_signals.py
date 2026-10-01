#!/usr/bin/env python
"""Simulation-based optimisation of a common-cycle fixed-time plan (CMA-ES).

Decision variables (normalised vector x):
  x[0]                 common cycle: C = round(Cmin + clip(x0, 0, 1) * (Cmax - Cmin))
  x[1 : 1+4J]          split logits per junction/phase: greens = split_greens(exp(z), C)
                       (always satisfies min/max green and sum = C - lost time)
  x[1+4J : 1+5J]       offset fraction per junction: offset = round((x mod 1) * C)
Objective: mean over training seeds of TTS [veh.h] (incl. insertion backlog)
           + penalty for gridlock + penalty per unserved vehicle (no teleporting).
Warm start: CMA-ES mean = Webster plan (offsets 0).

Outputs (results/signals/optimisation/): evaluations.csv, convergence.png,
time_space.png; signals/optimised.add.xml and signals/optimised.json.

Usage:
    python scripts/optimise_signals.py [--evaluations 240] [--workers 4] [--plot-only]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import cma
import numpy as np
import pandas as pd

from testbed.config import PROJECT_ROOT, load_config, network_files
from testbed.log import setup_logging
from testbed.network import read_net
from testbed.runner import run
from testbed.scenarios import make_runspec
from testbed.signals import (SIGNALS_CONFIG, ApproachPhase, Plan, cycle_bounds, intergreen,
                             load_signals_config, phase_structure, plan_file, signals_dir,
                             split_greens, write_plan)

log = logging.getLogger("optimise_signals")


# ------------------------------------------------------------------ encoding
class Encoding:
    """Map between the CMA-ES vector and a Plan."""

    def __init__(self, struct: dict[str, list[ApproachPhase]], scfg: dict[str, Any]) -> None:
        self.tids = sorted(struct)
        self.n_ph = len(struct[self.tids[0]])
        self.scfg = scfg
        self.cmin, self.cmax = cycle_bounds(scfg, self.n_ph)
        self.dim = 1 + self.n_ph * len(self.tids) + len(self.tids)

    def decode(self, x: np.ndarray, name: str = "candidate") -> Plan:
        c = int(round(self.cmin + float(np.clip(x[0], 0, 1)) * (self.cmax - self.cmin)))
        greens, offsets = {}, {}
        J, n = len(self.tids), self.n_ph
        for j, tid in enumerate(self.tids):
            z = x[1 + n * j: 1 + n * (j + 1)]
            w = np.exp(z - z.max())
            greens[tid] = split_greens(list(w), c, self.scfg)
            offsets[tid] = int(round((float(x[1 + n * J + j]) % 1.0) * c)) % c
        return Plan(name, c, greens, offsets)

    def encode(self, plan: Plan) -> np.ndarray:
        x = [(plan.cycle - self.cmin) / (self.cmax - self.cmin)]
        for tid in self.tids:
            x += list(np.log(np.asarray(plan.greens[tid], dtype=float)))
        x += [plan.offsets.get(tid, 0) / plan.cycle for tid in self.tids]
        return np.asarray(x, dtype=float)


# ---------------------------------------------------------------- evaluation
def eval_one(args) -> dict[str, Any]:
    """Worker: one (candidate, seed) run; keeps only metrics.json on disk."""
    cfg, scfg, eid, plan_xml, seed = args
    o = scfg["optimisation"]
    out = PROJECT_ROOT / scfg["results_dir"] / "optimisation" / "runs" / f"e{eid:04d}" / f"seed_{seed}"
    setup_logging(None, level=logging.WARNING)
    try:
        m = run(make_runspec(cfg, o["scenario"], seed, Path(plan_xml), out, {"eval": eid}), cfg)
    finally:
        for p in out.iterdir() if out.exists() else []:
            if p.name != "metrics.json":
                p.unlink()
    return {"eval": eid, "seed": seed, "tts": m["tts_veh_h"], "gridlock": m["gridlock"],
            "unserved": m["unserved_total"], "delay": m["mean_delay_s"], "spill": m["spillback_count"]}


def objective(rows: list[dict[str, Any]], pen: dict[str, Any]) -> float:
    """Mean over seeds of TTS + gridlock penalty + unserved penalty."""
    vals = [r["tts"] + (pen["gridlock_veh_h"] if r["gridlock"] else 0.0)
            + pen["unserved_veh_h_per_veh"] * r["unserved"] for r in rows]
    return float(np.mean(vals))


class Evaluator:
    """Evaluates batches of plans in parallel and logs every evaluation."""

    def __init__(self, cfg, scfg, struct, enc: Encoding, pool) -> None:
        self.cfg, self.scfg, self.struct, self.enc, self.pool = cfg, scfg, struct, enc, pool
        self.dir = PROJECT_ROOT / scfg["results_dir"] / "optimisation"
        self.plans_dir = self.dir / "plans"
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "evaluations.csv"
        self.records: list[dict[str, Any]] = []
        self.next_id = 0

    def __call__(self, xs: list[np.ndarray], generation: int, kind: str = "cma") -> list[float]:
        o = self.scfg["optimisation"]
        jobs, plans = [], []
        for x in xs:
            eid = self.next_id
            self.next_id += 1
            plan = self.enc.decode(x, f"e{eid:04d}")
            pf = write_plan(plan, self.struct, self.plans_dir / f"e{eid:04d}.add.xml", self.scfg)
            plans.append((eid, plan, x))
            jobs += [(self.cfg, self.scfg, eid, str(pf), s) for s in o["train_seeds"]]
        res: dict[int, list] = {}
        for r in self.pool.imap_unordered(eval_one, jobs):
            res.setdefault(r["eval"], []).append(r)
        fs = []
        for eid, plan, x in plans:
            rows = res[eid]
            f = objective(rows, o["penalty"])
            fs.append(f)
            rec = {"eval": eid, "generation": generation, "kind": kind, "objective": f,
                   "cycle": plan.cycle,
                   "tts_mean": np.mean([r["tts"] for r in rows]),
                   "delay_mean": np.mean([r["delay"] for r in rows]),
                   "spill_mean": np.mean([r["spill"] for r in rows]),
                   "gridlock_any": any(r["gridlock"] for r in rows),
                   "unserved_mean": np.mean([r["unserved"] for r in rows]),
                   "greens": json.dumps(plan.greens), "offsets": json.dumps(plan.offsets),
                   "x": json.dumps([round(float(v), 6) for v in x])}
            self.records.append(rec)
        pd.DataFrame(self.records).to_csv(self.csv, index=False)
        return fs

    def best(self) -> dict[str, Any]:
        return min(self.records, key=lambda r: r["objective"])


# ------------------------------------------------------------------ plotting
def plot_convergence(df: pd.DataFrame, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, series = "#1f2328", "#6e7781", "#e6e8eb", "#2a6fb0"
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cma_rows = df[df.kind == "cma"]
    ax.scatter(cma_rows["eval"], cma_rows["objective"], s=10, color=series, alpha=0.35, lw=0,
               label="candidate (mean of training seeds)")
    ax.plot(df["eval"], df["objective"].cummin(), color=series, lw=2, label="best so far")
    w = df[df.kind == "webster"]
    if len(w):
        ax.axhline(float(w["objective"].iloc[0]), color=muted, ls="--", lw=1)
        ax.annotate(f"Webster {float(w['objective'].iloc[0]):.1f}", (df["eval"].max(), float(w["objective"].iloc[0])),
                    xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=muted)
    lo = df["objective"].min()
    ax.set_ylim(lo * 0.97, min(df["objective"].quantile(0.95), lo * 1.6))
    ax.set_xlabel("evaluation", fontsize=9, color=muted)
    ax.set_title("CMA-ES convergence: objective = mean TTS [veh.h] + penalties (D1, training seeds)",
                 fontsize=10, color=ink, loc="left")
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def green_windows(plan: Plan, tid: str, k: int, ig: int, t_end: float) -> list[tuple[float, float]]:
    """Green intervals of phase k at junction tid in [0, t_end) (cycle starts at the offset)."""
    g = plan.greens[tid]
    start_in_cycle = sum(g[:k]) + k * ig
    out = []
    t = plan.offsets.get(tid, 0) + start_in_cycle - plan.cycle * 2
    while t < t_end:
        out.append((t, t + g[k]))
        t += plan.cycle
    return out


def plot_time_space(plan: Plan, net_path: Path, scfg: dict[str, Any], png: Path, speed: float) -> None:
    """Green bands along the middle row (E/W) and middle column (N/S) with free-flow trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = read_net(net_path)
    ig = intergreen(scfg)
    tids = sorted(plan.greens)
    xs = sorted({t[0] for t in tids})
    ys = sorted({t[1:] for t in tids})
    row = [f"{c}{ys[len(ys) // 2]}" for c in xs]          # e.g. A1, B1, C1
    col = [f"{xs[len(xs) // 2]}{r}" for r in ys]          # e.g. B0, B1, B2
    # phase index by approach side: sequence is N, E, S, W
    PH = {"N": 0, "E": 1, "S": 2, "W": 3}
    t_end = 3 * plan.cycle
    ink, muted, grid = "#1f2328", "#6e7781", "#e6e8eb"
    c_fwd, c_bwd = "#2a6fb0", "#e07b39"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, chain, axis, (fwd, bwd), title in (
            (axes[0], row, 0, ("W", "E"), "Middle row: eastbound (served by W approach) / westbound (E approach)"),
            (axes[1], col, 1, ("S", "N"), "Middle column: northbound (S approach) / southbound (N approach)")):
        pos = [net.getNode(t).getCoord()[axis] for t in chain]
        for t, p in zip(chain, pos):
            for side, colr, dy in ((fwd, c_fwd, -6), (bwd, c_bwd, 6)):
                for a, b in green_windows(plan, t, PH[side], ig, t_end):
                    ax.plot([max(a, 0), min(b, t_end)], [p + dy, p + dy], color=colr, lw=5,
                            solid_capstyle="butt") if b > 0 else None
            ax.annotate(t, (t_end, p), xytext=(4, -3), textcoords="offset points", fontsize=8, color=ink,
                        annotation_clip=False)
        # free-flow trajectories starting at each forward green start at the first junction
        for a, _b in green_windows(plan, chain[0], PH[fwd], ig, t_end):
            ax.plot([a, a + (pos[-1] - pos[0]) / speed], [pos[0], pos[-1]], color=c_fwd, lw=0.8, ls=":")
        for a, _b in green_windows(plan, chain[-1], PH[bwd], ig, t_end):
            ax.plot([a, a + (pos[-1] - pos[0]) / speed], [pos[-1], pos[0]], color=c_bwd, lw=0.8, ls=":")
        ax.set_xlim(0, t_end)
        ax.set_ylim(pos[0] - 60, pos[-1] + 60)
        ax.set_xlabel("time in cycle sequence [s]", fontsize=9, color=muted)
        ax.set_ylabel("position [m]", fontsize=9, color=muted)
        ax.set_title(title, fontsize=9, color=ink, loc="left")
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    from matplotlib.lines import Line2D

    fig.legend(handles=[Line2D([], [], color=c_fwd, lw=5, label="green, forward direction"),
                        Line2D([], [], color=c_bwd, lw=5, label="green, reverse direction"),
                        Line2D([], [], color=muted, lw=0.8, ls=":", label=f"free-flow trajectory ({speed:.1f} m/s)")],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Time-space diagram, plan '{plan.name}' (C = {plan.cycle} s)", fontsize=10, color=ink,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--signals-config", default=str(SIGNALS_CONFIG))
    ap.add_argument("--evaluations", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true", help="re-plot from evaluations.csv")
    args = ap.parse_args()
    cfg, scfg = load_config(args.config), load_signals_config(args.signals_config)
    o = scfg["optimisation"]
    cfg = copy.deepcopy(cfg)
    cfg["sumo"]["use_libsumo"] = bool(o["use_libsumo"])
    out = PROJECT_ROOT / scfg["results_dir"] / "optimisation"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "optimise.log")
    files = network_files(cfg, scfg["network"])
    struct = phase_structure(files["net"], files["mapping"])
    enc = Encoding(struct, scfg)
    speed = float(cfg["demand"]["vehicle_type"]["maxSpeed"])

    if not args.plot_only:
        n_eval = args.evaluations or int(o["evaluations"])
        webster = Plan.from_json(signals_dir(scfg) / "webster.json")
        x0 = enc.encode(webster)
        assert enc.decode(x0).greens == webster.greens, "Webster encoding round-trip failed"
        opts = {"seed": int(o["seed"]), "verbose": -9,
                "bounds": [[0.0] + [-np.inf] * (enc.dim - 1), [1.0] + [np.inf] * (enc.dim - 1)]}
        if o.get("popsize"):
            opts["popsize"] = int(o["popsize"])
        es = cma.CMAEvolutionStrategy(x0, float(o["sigma0"]), opts)
        log.info("CMA-ES: dim %d, popsize %d, %d evaluations x %d seeds, cycle range [%d, %d] s",
                 enc.dim, es.popsize, n_eval, len(o["train_seeds"]), enc.cmin, enc.cmax)
        shutil.rmtree(out / "runs", ignore_errors=True)
        with mp.get_context("spawn").Pool(args.workers or int(o["workers"])) as pool:
            ev = Evaluator(cfg, scfg, struct, enc, pool)
            f0 = ev([x0], 0, "webster")[0]
            log.info("Webster (warm start): objective %.2f", f0)
            gen = 0
            while ev.next_id - 1 < n_eval:
                gen += 1
                xs = es.ask()
                fs = ev(xs, gen)
                es.tell(xs, fs)
                b = ev.best()
                log.info("gen %3d | evals %4d | gen best %.2f | overall best %.2f (eval %d, C=%d) | sigma %.3f",
                         gen, ev.next_id - 1, min(fs), b["objective"], b["eval"], b["cycle"], es.sigma)
            fm = ev([es.result.xfavorite], gen, "cma_mean")[0]
            log.info("final CMA mean: objective %.2f", fm)
        shutil.rmtree(out / "runs", ignore_errors=True)
        b = ev.best()
        best = enc.decode(np.asarray(json.loads(b["x"])), "optimised")
        best.info = {"objective_train": b["objective"], "tts_train_mean": b["tts_mean"],
                     "eval": b["eval"], "kind": b["kind"], "webster_objective_train": f0,
                     "train_seeds": o["train_seeds"], "scenario": o["scenario"],
                     "splits": {t: [round(g / sum(best.greens[t]), 4) for g in best.greens[t]]
                                for t in best.greens},
                     "evaluations": ev.next_id}
        write_plan(best, struct, plan_file(scfg, "optimised"), scfg,
                   note=f"CMA-ES on {o['scenario']} seeds {o['train_seeds']}, objective {b['objective']:.2f}.")
        best.to_json(signals_dir(scfg) / "optimised.json")
        log.info("optimised: C=%d s, objective %.2f (Webster %.2f) -> %s", best.cycle, b["objective"], f0,
                 plan_file(scfg, "optimised"))
    df = pd.read_csv(out / "evaluations.csv")
    plot_convergence(df, out / "convergence.png")
    best = Plan.from_json(signals_dir(scfg) / "optimised.json")
    plot_time_space(best, files["net"], scfg, out / "time_space.png", speed)
    plot_time_space(Plan.from_json(signals_dir(scfg) / "webster.json"), files["net"], scfg,
                    out / "time_space_webster.png", speed)
    log.info("plots -> %s", out)


if __name__ == "__main__":
    main()

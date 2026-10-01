# Traffic signal control testbed

This repository provides a reproducible SUMO-based simulation testbed for comparing traffic signal controllers in a synthetic signalised network. Phase 0 focuses on a stable baseline scenario with standard fixed-time and actuated signal control under normal demand.

## Repository layout

- `scripts/`: build network, demand, signals, calibration, experiment, analysis
- `testbed/`: TraCI wrapper, detectors, metrics, controller interfaces
- `configs/phase0.yaml`: all runtime configuration parameters
- `networks/`: generated network and detector files
- `results/phase0/`: experiment outputs
- `tests/`: reproducibility and correctness tests

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code automatically detects `SUMO_HOME` from the environment or the installed Python package location and exits with a clear error if SUMO is not available.

## Core workflow

```bash
python scripts/build_network.py
python scripts/build_signals.py
python scripts/calibrate_demand.py
python scripts/run_experiment.py --seeds 5
python scripts/analyse_results.py
```

## Phase 0 scope

- baseline 3x3 grid network and optional corridor network
- E1 stop-line loops and E2 lane-area detectors
- transient demand calibration under fixed-time control
- default, Webster fixed-time, and actuated controllers
- metrics for TTS, delay, throughput, queues, spillback, and gridlock
- future-ready controller interfaces for Phase 1–4 experiments

## Notes

- Controllers receive only E1 detector data and never read E2 data.
- Teleporting is disabled to preserve visible gridlock behaviour.
- The simulation is designed so future incident and model-based controllers can be plugged in without rewriting the runner.

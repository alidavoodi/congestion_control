"""Configuration loading and project paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Path = PROJECT_ROOT / "configs" / "phase0.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML experiment configuration.

    Args:
        path: Config file; defaults to ``configs/phase0.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` relative to the project root."""
    return PROJECT_ROOT / cfg["paths"][key]


def network_files(cfg: dict[str, Any], net: str) -> dict[str, Path]:
    """Standard file locations for a generated network."""
    d = project_path(cfg, "networks_dir")
    return {
        "net": d / f"{net}.net.xml",
        "det": d / f"{net}.det.add.xml",
        "mapping": d / f"{net}.detectors.csv",
        "png": d / f"{net}.png",
    }


def tls_file(cfg: dict[str, Any], net: str, program: str) -> Path:
    """Additional file holding the signal program ``program`` for ``net``."""
    return project_path(cfg, "networks_dir") / f"{net}.tls_{program}.add.xml"


def load_demand_levels(cfg: dict[str, Any]) -> dict[str, float]:
    """Load calibrated demand multipliers (written by calibrate_demand.py)."""
    p = project_path(cfg, "demand_levels_file")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run scripts/calibrate_demand.py first."
        )
    with open(p, encoding="utf-8") as fh:
        return {k: float(v) for k, v in yaml.safe_load(fh)["levels"].items()}

"""Locate a SUMO installation and set ``SUMO_HOME``.

Order of precedence: existing ``SUMO_HOME`` env var, the ``eclipse-sumo`` pip
package, then ``sumo`` on ``PATH``. Fails with a clear message otherwise.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


class SumoNotFoundError(RuntimeError):
    """Raised when no usable SUMO installation can be found."""


def _valid(home: Path) -> bool:
    return (home / "bin").is_dir() and (home / "tools").is_dir()


def find_sumo_home() -> Path:
    """Detect SUMO, export ``SUMO_HOME`` and add ``$SUMO_HOME/tools`` to ``sys.path``.

    Returns:
        Path to the SUMO installation.

    Raises:
        SumoNotFoundError: if SUMO cannot be located.
    """
    candidates: list[Path] = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]))
    try:
        import sumo  # eclipse-sumo pip package

        candidates.append(Path(sumo.SUMO_HOME))
    except ImportError:
        pass
    exe = shutil.which("sumo")
    if exe:
        p = Path(exe).resolve().parent.parent
        candidates += [p, p / "share" / "sumo"]
    for home in candidates:
        if _valid(home):
            os.environ["SUMO_HOME"] = str(home)
            tools = str(home / "tools")
            if tools not in sys.path:
                sys.path.append(tools)
            return home
    raise SumoNotFoundError(
        "SUMO not found. Install it with `pip install -r requirements.txt` "
        "(eclipse-sumo) or set SUMO_HOME to an existing SUMO installation."
    )


def sumo_binary(name: str) -> str:
    """Absolute path of a SUMO executable, e.g. ``sumo``, ``sumo-gui``, ``netgenerate``."""
    path = find_sumo_home() / "bin" / name
    if not path.exists():
        raise SumoNotFoundError(f"SUMO binary {name!r} not found at {path}")
    return str(path)


def sumo_tool(rel: str) -> Path:
    """Path of a script under ``$SUMO_HOME/tools``."""
    path = find_sumo_home() / "tools" / rel
    if not path.exists():
        raise SumoNotFoundError(f"SUMO tool {rel!r} not found at {path}")
    return path

"""Frozen signal plans must not change (skipped until plans are frozen)."""
from __future__ import annotations

import hashlib
import json

import pytest

from testbed.signals import load_signals_config, signals_dir

MANIFEST = signals_dir(load_signals_config()) / "FROZEN.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="plans not frozen yet")
def test_frozen_plans_unchanged():
    m = json.loads(MANIFEST.read_text())
    for name, info in m["files"].items():
        p = MANIFEST.parent / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == info["sha256"], f"{name} was modified"

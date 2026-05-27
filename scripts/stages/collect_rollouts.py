#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mimo_s2s_gepa.skillopt import run_openclaw_rollouts


run_openclaw_rollouts("configs/collect_rollouts.yaml")

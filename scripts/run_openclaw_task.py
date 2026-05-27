#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimo_s2s_gepa.skillopt.openclaw_runner import run_openclaw_task


run_openclaw_task("configs/openclaw_task.yaml")

#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mimo_s2s_gepa.skillopt.validation_gate import run_skill_validation


run_skill_validation("configs/validate_candidate.yaml")

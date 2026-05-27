#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimo_s2s_gepa.skillopt.skill_candidate import run_skill_candidate_proposal


run_skill_candidate_proposal("configs/propose_skill_candidate.yaml")

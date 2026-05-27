#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mimo_s2s_gepa.skillopt.trajectory_analyzer import run_trajectory_analysis


run_trajectory_analysis("configs/analyze_trajectories.yaml")

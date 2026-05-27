#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mimo_s2s_gepa.skillopt import (  # noqa: E402
    run_openclaw_rollouts,
    run_skill_promotion,
    run_skill_validation,
    run_skillopt_gepa,
    run_trajectory_analysis,
)


STAGES: dict[str, tuple[Callable[[str], Path], str]] = {
    "collect": (run_openclaw_rollouts, "configs/collect_rollouts.yaml"),
    "analyze": (run_trajectory_analysis, "configs/analyze_trajectories.yaml"),
    "optimize": (run_skillopt_gepa, "configs/optimize_skill.yaml"),
    "validate": (run_skill_validation, "configs/validate_candidate.yaml"),
}


def run_stage(stage: str) -> Path:
    runner, config_path = STAGES[stage]
    print(f"stage={stage} config={config_path}", flush=True)
    return runner(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SkillOpt teaching flow. Default is `all`, which collects "
            "OpenClaw trajectories, analyzes them, runs DSPy GEPA, validates the "
            "candidate, and writes a review-only promotion package."
        )
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "collect", "analyze", "optimize", "validate", "promote"],
        help="SkillOpt stage to run. Defaults to all.",
    )
    args = parser.parse_args()

    if args.stage == "all":
        for stage in ["collect", "analyze", "optimize", "validate"]:
            run_stage(stage)
        print("stage=promote config=configs/promote_candidate.yaml review_only=true", flush=True)
        run_skill_promotion("configs/promote_candidate.yaml")
        return

    if args.stage == "promote":
        print("stage=promote config=configs/promote_candidate.yaml review_only=true", flush=True)
        run_skill_promotion("configs/promote_candidate.yaml")
        return

    run_stage(args.stage)


if __name__ == "__main__":
    main()

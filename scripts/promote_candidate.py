#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimo_s2s_gepa.skillopt import run_skill_promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Review or promote a validated OpenClaw skill candidate.")
    parser.add_argument("--config", default="configs/promote_candidate.yaml")
    parser.add_argument("--apply", action="store_true", help="Overwrite the live OpenClaw skill with the candidate.")
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help="Allow applying a candidate rejected by validation.",
    )
    args = parser.parse_args()

    overrides = {}
    if args.apply:
        overrides["apply"] = True
    if args.allow_rejected:
        overrides["allow_rejected"] = True

    run_skill_promotion(args.config, overrides=overrides)


if __name__ == "__main__":
    main()

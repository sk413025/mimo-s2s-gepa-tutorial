from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, load_config, resolve_path
from ..runner import make_run_dir, save_json
from .rollout import load_rollout_tasks, run_one_rollout
from .trajectory_analyzer import load_json

LOGGER = logging.getLogger(__name__)


def find_latest_candidate_dir(output_dir: str | Path) -> Path:
    candidates = sorted(Path(output_dir).glob("*_skill_candidate"))
    if not candidates:
        raise FileNotFoundError(f"No skill candidate runs found under {output_dir}")
    return candidates[-1]


def resolve_candidate_dir(config: dict[str, Any]) -> Path:
    configured = str(config.get("skill_candidate_dir") or "").strip()
    if configured:
        return resolve_path(configured)
    return find_latest_candidate_dir(config["output_dir"])


def make_rollout_config(config: dict[str, Any], skill_path: Path) -> dict[str, Any]:
    rollout_config = dict(config)
    rollout_config["openclaw_skill_path"] = str(skill_path)
    rollout_config["continue_on_validation_error"] = True
    return rollout_config


def run_rollout_set(config: dict[str, Any], run_dir: Path, skill_path: Path, label: str) -> dict[str, Any]:
    rollout_dir = run_dir / f"{label}_rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rollout_config = make_rollout_config(config, skill_path)
    tasks = load_rollout_tasks(rollout_config)
    summaries: list[dict[str, Any]] = []

    for task in tasks:
        LOGGER.info("running %s validation rollout task %s", label, task.task_id)
        summaries.append(run_one_rollout(rollout_config, rollout_dir, task))
        save_json(rollout_dir / "rollout_report.json", {"config": rollout_config, "tasks": summaries})

    report = {
        "label": label,
        "skill_path": str(skill_path),
        "rollout_dir": str(rollout_dir),
        "num_tasks": len(tasks),
        "num_passed": sum(1 for row in summaries if row.get("passed")),
        "num_clean_passed": sum(1 for row in summaries if row.get("passed") and not row.get("used_continuation")),
        "total_score": sum(float(row.get("score") or 0.0) for row in summaries),
        "tasks": summaries,
    }
    save_json(rollout_dir / "rollout_report.json", report)
    return report


def build_decision(
    *,
    candidate_dir: Path,
    candidate_skill_path: Path,
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, Any]:
    baseline_passed = int(baseline_report.get("num_passed", 0))
    candidate_passed = int(candidate_report.get("num_passed", 0))
    baseline_score = float(baseline_report.get("total_score", baseline_passed))
    candidate_score = float(candidate_report.get("total_score", candidate_passed))
    accepted = candidate_score > baseline_score
    if accepted:
        reason = "candidate validation score is higher than baseline"
    elif candidate_score == baseline_score:
        reason = "candidate tied baseline; reject because validation requires strict improvement"
    else:
        reason = "candidate validation score is lower than baseline"

    return {
        "accepted": accepted,
        "reason": reason,
        "baseline_passed": baseline_passed,
        "candidate_passed": candidate_passed,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "baseline_clean_passed": baseline_report.get("num_clean_passed"),
        "candidate_clean_passed": candidate_report.get("num_clean_passed"),
        "baseline_num_tasks": baseline_report.get("num_tasks"),
        "candidate_num_tasks": candidate_report.get("num_tasks"),
        "baseline_rollout_dir": baseline_report.get("rollout_dir"),
        "candidate_rollout_dir": candidate_report.get("rollout_dir"),
        "candidate_dir": str(candidate_dir),
        "candidate_skill_path": str(candidate_skill_path),
    }


def run_skill_validation(config_path: str) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(config_path)
    run_dir = make_run_dir(config, "skill_validation")
    candidate_dir = resolve_candidate_dir(config)
    proposal = load_json(candidate_dir / "proposal.json")

    baseline_skill_path = resolve_path(config["openclaw_live_skill_path"])
    candidate_skill_path = Path(proposal["candidate_skill_path"])
    if not candidate_skill_path.is_file():
        raise FileNotFoundError(f"Candidate skill does not exist: {candidate_skill_path}")

    LOGGER.info("validating baseline skill %s", baseline_skill_path)
    baseline_report = run_rollout_set(config, run_dir, baseline_skill_path, "baseline")
    save_json(run_dir / "baseline_report.json", baseline_report)

    LOGGER.info("validating candidate skill %s", candidate_skill_path)
    candidate_report = run_rollout_set(config, run_dir, candidate_skill_path, "candidate")
    save_json(run_dir / "candidate_report.json", candidate_report)

    decision = build_decision(
        candidate_dir=candidate_dir,
        candidate_skill_path=candidate_skill_path,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
    )
    save_json(run_dir / "decision.json", decision)

    summary = {
        "mode": "skill_validation",
        "project_root": str(PROJECT_ROOT),
        "config": config,
        "candidate_proposal": proposal,
        "baseline": baseline_report,
        "candidate": candidate_report,
        "decision": decision,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"saved_run_dir={run_dir}")
    return run_dir

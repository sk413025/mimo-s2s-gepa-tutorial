from __future__ import annotations

import difflib
import logging
import shutil
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, load_config, resolve_path
from ..runner import make_run_dir, save_json
from .registry import append_registry_record, build_promotion_record, file_sha256
from .trajectory_analyzer import load_json

LOGGER = logging.getLogger(__name__)


def find_latest_validation_dir(output_dir: str | Path) -> Path:
    candidates = sorted(Path(output_dir).glob("*_skill_validation"))
    if not candidates:
        raise FileNotFoundError(f"No skill validation runs found under {output_dir}")
    return candidates[-1]


def resolve_validation_dir(config: dict[str, Any]) -> Path:
    configured = str(config.get("skill_validation_dir") or "").strip()
    if configured:
        return resolve_path(configured)
    return find_latest_validation_dir(config["output_dir"])


def bool_config(config: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def write_skill_diff(live_skill_path: Path, candidate_skill_path: Path, diff_path: Path) -> str:
    live_text = live_skill_path.read_text(encoding="utf-8").splitlines(keepends=True)
    candidate_text = candidate_skill_path.read_text(encoding="utf-8").splitlines(keepends=True)
    diff_text = "".join(
        difflib.unified_diff(
            live_text,
            candidate_text,
            fromfile=str(live_skill_path),
            tofile=str(candidate_skill_path),
        )
    )
    diff_path.write_text(diff_text, encoding="utf-8")
    return diff_text


def load_promotion_inputs(config: dict[str, Any]) -> tuple[Path, dict[str, Any], Path]:
    validation_dir = resolve_validation_dir(config)
    decision_path = validation_dir / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(f"Validation decision does not exist: {decision_path}")
    decision = load_json(decision_path)

    candidate_skill_path = Path(decision.get("candidate_skill_path") or "")
    if not candidate_skill_path.is_file():
        raise FileNotFoundError(f"Candidate skill does not exist: {candidate_skill_path}")

    return validation_dir, decision, candidate_skill_path


def build_promotion_report(
    *,
    config: dict[str, Any],
    run_dir: Path,
    validation_dir: Path,
    decision: dict[str, Any],
    candidate_skill_path: Path,
    live_skill_path: Path,
    diff_path: Path,
    candidate_copy_path: Path,
    backup_live_skill_path: Path,
) -> dict[str, Any]:
    apply_requested = bool_config(config, "apply", False)
    allow_rejected = bool_config(config, "allow_rejected", False)
    accepted = bool(decision.get("accepted", False))
    promotion_allowed = accepted or allow_rejected
    status = "dry_run"
    blocked_reason = ""
    applied = False

    live_hash_before = file_sha256(live_skill_path)
    candidate_hash = file_sha256(candidate_skill_path)

    if apply_requested and not promotion_allowed:
        status = "blocked_rejected_candidate"
        blocked_reason = "candidate was rejected by validation; set allow_rejected: true to override"
    elif apply_requested:
        backup_live_skill_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live_skill_path, backup_live_skill_path)
        shutil.copy2(candidate_skill_path, live_skill_path)
        applied = True
        status = "promoted"

    live_hash_after = file_sha256(live_skill_path)

    return {
        "mode": "skill_promotion",
        "project_root": str(PROJECT_ROOT),
        "config": config,
        "status": status,
        "applied": applied,
        "apply_requested": apply_requested,
        "allow_rejected": allow_rejected,
        "promotion_allowed": promotion_allowed,
        "blocked_reason": blocked_reason,
        "validation_dir": str(validation_dir),
        "decision": decision,
        "accepted": accepted,
        "decision_reason": decision.get("reason", ""),
        "live_skill_path": str(live_skill_path),
        "candidate_skill_path": str(candidate_skill_path),
        "candidate_copy_path": str(candidate_copy_path),
        "diff_path": str(diff_path),
        "backup_live_skill_path": str(backup_live_skill_path),
        "live_hash_before": live_hash_before,
        "candidate_hash": candidate_hash,
        "live_hash_after": live_hash_after,
    }


def run_skill_promotion(config_path: str, overrides: dict[str, Any] | None = None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(config_path)
    if overrides:
        config.update(overrides)
    run_dir = make_run_dir(config, "skill_promotion")
    validation_dir, decision, candidate_skill_path = load_promotion_inputs(config)

    live_skill_path = resolve_path(config["openclaw_live_skill_path"])
    if not live_skill_path.is_file():
        raise FileNotFoundError(f"Live skill does not exist: {live_skill_path}")

    candidate_copy_dir = run_dir / "candidate"
    candidate_copy_dir.mkdir(parents=True, exist_ok=True)
    candidate_copy_path = candidate_copy_dir / "SKILL.md"
    shutil.copy2(candidate_skill_path, candidate_copy_path)

    live_before_dir = run_dir / "live_before"
    live_before_dir.mkdir(parents=True, exist_ok=True)
    backup_live_skill_path = live_before_dir / "SKILL.md"
    shutil.copy2(live_skill_path, backup_live_skill_path)

    diff_path = run_dir / "diff.md"
    write_skill_diff(live_skill_path, candidate_skill_path, diff_path)

    report = build_promotion_report(
        config=config,
        run_dir=run_dir,
        validation_dir=validation_dir,
        decision=decision,
        candidate_skill_path=candidate_skill_path,
        live_skill_path=live_skill_path,
        diff_path=diff_path,
        candidate_copy_path=candidate_copy_path,
        backup_live_skill_path=backup_live_skill_path,
    )

    registry_file = ""
    if report["applied"]:
        registry_file = str(
            append_registry_record(
                config,
                build_promotion_record(
                    source="skill_promotion",
                    run_dir=run_dir,
                    decision=decision,
                    report=report,
                ),
            )
        )
    report["candidate_registry_path"] = registry_file

    save_json(run_dir / "promotion_report.json", report)
    save_json(run_dir / "summary.json", report)
    LOGGER.info("saved promotion output to %s", run_dir)
    print(f"saved_run_dir={run_dir}")
    return run_dir

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def registry_path(config: dict[str, Any]) -> Path:
    configured = str(config.get("candidate_registry_path") or "").strip()
    if configured:
        return Path(configured)
    return Path(config["output_dir"]) / "candidate_registry.jsonl"


def append_registry_record(config: dict[str, Any], record: dict[str, Any]) -> Path:
    path = registry_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **record,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def diff_line_count(diff_text: str) -> int:
    return len(
        [
            line
            for line in diff_text.splitlines()
            if (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        ]
    )


def collect_audio_paths(report: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for task in report.get("tasks", []):
        generated_audio = task.get("generated_audio") or {}
        audio_path = generated_audio.get("audio_path_expanded") or generated_audio.get("audio_path")
        if audio_path:
            paths.append(str(audio_path))
    return paths


def collect_continuation_tasks(report: dict[str, Any]) -> list[str]:
    tasks: list[str] = []
    for task in report.get("tasks", []):
        if task.get("used_continuation"):
            tasks.append(str(task.get("task_id")))
    return tasks


def build_candidate_record(
    *,
    source: str,
    run_dir: Path,
    candidate_skill_path: Path,
    candidate_diff_path: Path | None,
    proposal: dict[str, Any],
    analysis_dir: str | None = None,
    metric_call_dir: Path | None = None,
) -> dict[str, Any]:
    diff_text = candidate_diff_path.read_text(encoding="utf-8") if candidate_diff_path and candidate_diff_path.is_file() else ""
    return {
        "event": "candidate_created",
        "source": source,
        "run_dir": str(run_dir),
        "metric_call_dir": str(metric_call_dir) if metric_call_dir else "",
        "analysis_dir": analysis_dir or str(proposal.get("analysis_dir") or ""),
        "candidate_skill_path": str(candidate_skill_path),
        "candidate_diff_path": str(candidate_diff_path or ""),
        "candidate_hash": file_sha256(candidate_skill_path),
        "diff_hash": text_sha256(diff_text) if diff_text else "",
        "diff_line_count": proposal.get("diff_line_count", diff_line_count(diff_text)),
        "patch_edits": proposal.get("patch_edits", []),
        "validation_errors": proposal.get("validation_errors", []),
        "proposal_notes": proposal.get("proposal_notes", {}),
    }

def build_validation_record(
    *,
    source: str,
    run_dir: Path,
    candidate_dir: Path,
    proposal: dict[str, Any],
    decision: dict[str, Any],
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, Any]:
    candidate_skill_path = Path(decision.get("candidate_skill_path") or proposal.get("candidate_skill_path") or "")
    return {
        "event": "candidate_validated",
        "source": source,
        "run_dir": str(run_dir),
        "candidate_dir": str(candidate_dir),
        "analysis_dir": str(proposal.get("analysis_dir") or ""),
        "candidate_skill_path": str(candidate_skill_path),
        "candidate_hash": file_sha256(candidate_skill_path) if candidate_skill_path.is_file() else "",
        "patch_edits": proposal.get("patch_edits", []),
        "accepted": decision.get("accepted", False),
        "decision_reason": decision.get("reason", ""),
        "baseline_passed": decision.get("baseline_passed"),
        "candidate_passed": decision.get("candidate_passed"),
        "baseline_score": decision.get("baseline_score"),
        "candidate_score": decision.get("candidate_score"),
        "baseline_clean_passed": decision.get("baseline_clean_passed"),
        "candidate_clean_passed": decision.get("candidate_clean_passed"),
        "baseline_audio_paths": collect_audio_paths(baseline_report),
        "candidate_audio_paths": collect_audio_paths(candidate_report),
        "baseline_continuation_tasks": collect_continuation_tasks(baseline_report),
        "candidate_continuation_tasks": collect_continuation_tasks(candidate_report),
    }

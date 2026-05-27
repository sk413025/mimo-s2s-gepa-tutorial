from __future__ import annotations

from pathlib import Path
from typing import Any

from .rollout_tasks import OpenClawRolloutTask


def validate_rollout_result(parsed: dict[str, Any], task: OpenClawRolloutTask) -> None:
    errors = find_validation_errors(parsed, task)
    if errors:
        raise RuntimeError("; ".join(errors))


def find_validation_errors(parsed: dict[str, Any], task: OpenClawRolloutTask) -> list[str]:
    errors: list[str] = []
    if parsed.get("final_status") != "success":
        errors.append(f"OpenClaw trajectory final status is not success: {parsed.get('final_status')}")

    tool_calls = parsed.get("tool_calls", [])
    tool_results = parsed.get("tool_results", [])
    tool_names = {str(call.get("name") or "") for call in tool_calls}
    evidence_text = "\n".join(
        [
            *(str(call.get("arguments") or "") for call in tool_calls),
            *(str(result.get("text") or "") for result in tool_results),
            *(str(text or "") for text in parsed.get("assistant_texts", [])),
        ]
    )

    for name in task.required_tool_names:
        if name not in tool_names:
            errors.append(f"required tool was not called: {name}")

    for needle in task.required_command_substrings:
        if needle not in evidence_text:
            errors.append(f"required command evidence was not found: {needle}")

    generated_audio = parsed.get("generated_audio") or {}
    if task.require_generated_audio and not parsed.get("generated_audio"):
        errors.append("OpenClaw rollout completed, but no generated audio was found in the trajectory")

    for field in task.required_audio_fields:
        if not generated_audio.get(field):
            errors.append(f"generated audio field is missing: {field}")

    if task.expected_backend and generated_audio.get("backend") != task.expected_backend:
        errors.append(f"generated audio backend mismatch: {generated_audio.get('backend')}")

    if task.require_audio_file_exists:
        audio_path = generated_audio.get("audio_path_expanded") or generated_audio.get("audio_path")
        if not audio_path or not Path(str(audio_path)).expanduser().is_file():
            errors.append(f"generated audio file does not exist: {audio_path}")

    return errors


def summarize_rollout(task: OpenClawRolloutTask, task_dir: Path, parsed: dict[str, Any]) -> dict[str, Any]:
    generated_audio = parsed.get("generated_audio") or {}
    validation_errors = find_validation_errors(parsed, task)
    turn_count = len(list(task_dir.glob("parsed_result_turn_*.json"))) or 1
    passed = not validation_errors
    score = 1.0 if passed and turn_count == 1 else (0.75 if passed else 0.0)
    return {
        "task_id": task.task_id,
        "task_dir": str(task_dir),
        "final_status": parsed.get("final_status"),
        "event_count": parsed.get("event_count"),
        "tool_call_count": len(parsed.get("tool_calls", [])),
        "tool_result_count": len(parsed.get("tool_results", [])),
        "turn_count": turn_count,
        "used_continuation": turn_count > 1,
        "generated_audio": generated_audio,
        "validation_errors": validation_errors,
        "passed": passed,
        "score": score,
    }

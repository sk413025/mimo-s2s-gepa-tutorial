from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import dspy

from ..config import PROJECT_ROOT, load_config, resolve_path
from ..runner import make_run_dir, save_json

LOGGER = logging.getLogger(__name__)


class AnalyzeOpenClawTrajectories(dspy.Signature):
    """Analyze OpenClaw rollout trajectories and produce SkillOpt feedback."""

    rollout_summary: str = dspy.InputField()
    trajectory_evidence: str = dspy.InputField()
    feedback_json: str = dspy.OutputField(
        desc=(
            "A valid JSON object with keys: success_patterns, failure_patterns, "
            "skill_issues, suggested_changes, evidence, next_validation_tasks. "
            "Each value must be an array of concise strings. Do not include markdown."
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_rollout_dir(output_dir: str | Path) -> Path:
    candidates = sorted(Path(output_dir).glob("*_openclaw_rollout"))
    if not candidates:
        raise FileNotFoundError(f"No OpenClaw rollout runs found under {output_dir}")
    return candidates[-1]


def resolve_rollout_dir(config: dict[str, Any]) -> Path:
    configured = str(config.get("rollout_run_dir") or "").strip()
    if configured:
        return resolve_path(configured)
    return find_latest_rollout_dir(config["output_dir"])


def short_text(value: Any, max_chars: int = 1200) -> str:
    text = str(value or "")
    return text if len(text) <= max_chars else text[:max_chars] + "...<truncated>"


def tool_result_brief(result: dict[str, Any]) -> dict[str, Any]:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    return {
        "tool_name": result.get("tool_name"),
        "is_error": result.get("is_error", False),
        "parsed_json": result.get("parsed_json"),
        "status": details.get("status"),
        "exit_code": details.get("exitCode"),
        "text": short_text(result.get("text"), 900),
    }


def compact_task_evidence(task_summary: dict[str, Any]) -> dict[str, Any]:
    task_dir = Path(task_summary["task_dir"])
    parsed = load_json(task_dir / "parsed_result.json")
    tool_calls = parsed.get("tool_calls", [])
    tool_results = parsed.get("tool_results", [])
    return {
        "task_id": task_summary.get("task_id"),
        "passed": task_summary.get("passed"),
        "final_status": parsed.get("final_status"),
        "generated_audio": parsed.get("generated_audio"),
        "assistant_texts": [short_text(text, 900) for text in parsed.get("assistant_texts", [])],
        "tool_calls": [
            {
                "name": call.get("name"),
                "arguments": call.get("arguments"),
            }
            for call in tool_calls
        ],
        "tool_results": [tool_result_brief(result) for result in tool_results],
    }


def build_evidence(rollout_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = load_json(rollout_dir / "rollout_report.json")
    tasks = [compact_task_evidence(task) for task in report.get("tasks", [])]
    return report, tasks


def build_analyzer_lm(config: dict[str, Any]) -> dspy.LM:
    return dspy.LM(
        config["gemma_model"],
        api_base=config["gemma_base_url"],
        api_key=config.get("gemma_api_key", "sk-local"),
        temperature=float(config.get("analyzer_temperature", 0.0)),
        max_tokens=int(config.get("analyzer_max_tokens", 2000)),
        cache=False,
    )


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_feedback_json(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Analyzer did not return a JSON object")

    keys = [
        "success_patterns",
        "failure_patterns",
        "skill_issues",
        "suggested_changes",
        "evidence",
        "next_validation_tasks",
    ]
    normalized: dict[str, Any] = {}
    for key in keys:
        value = parsed.get(key, [])
        normalized[key] = value if isinstance(value, list) else [str(value)]
    return normalized


def analyze_rollout_dir(config: dict[str, Any], rollout_dir: Path, run_dir: Path) -> dict[str, Any]:
    report, task_evidence = build_evidence(rollout_dir)
    rollout_summary = json.dumps(
        {
            "rollout_dir": str(rollout_dir),
            "num_tasks": report.get("num_tasks"),
            "num_passed": report.get("num_passed"),
            "tasks": report.get("tasks", []),
        },
        ensure_ascii=False,
        indent=2,
    )
    trajectory_evidence = json.dumps(task_evidence, ensure_ascii=False, indent=2)

    analyzer = dspy.Predict(AnalyzeOpenClawTrajectories)
    with dspy.context(lm=build_analyzer_lm(config)):
        prediction = analyzer(
            rollout_summary=rollout_summary,
            trajectory_evidence=trajectory_evidence,
        )

    raw_feedback = str(prediction.feedback_json)
    feedback = parse_feedback_json(raw_feedback)
    result = {
        "mode": "trajectory_analysis",
        "project_root": str(PROJECT_ROOT),
        "rollout_dir": str(rollout_dir),
        "feedback": feedback,
        "raw_feedback": raw_feedback,
    }
    save_json(run_dir / "trajectory_feedback.json", result)
    save_json(run_dir / "rollout_evidence.json", {"summary": report, "tasks": task_evidence})
    return result


def run_trajectory_analysis(config_path: str) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(config_path)
    rollout_dir = resolve_rollout_dir(config)
    run_dir = make_run_dir(config, "trajectory_analysis")
    LOGGER.info("analyzing OpenClaw rollout run %s", rollout_dir)
    result = analyze_rollout_dir(config, rollout_dir, run_dir)
    save_json(run_dir / "summary.json", result)
    print(f"saved_run_dir={run_dir}")
    return run_dir

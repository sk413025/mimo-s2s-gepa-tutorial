from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, load_config
from ..runner import make_run_dir, save_json
from .openclaw_cli import delete_agent, export_trajectory, run_agent_task, setup_agent
from .rollout_messages import build_continuation_message, build_rollout_message, build_skill_driven_rollout_message
from .rollout_tasks import OpenClawRolloutTask, load_rollout_tasks
from .rollout_validation import summarize_rollout, validate_rollout_result
from .trajectory_parser import parse_trajectory_bundle

LOGGER = logging.getLogger(__name__)


def build_task_record(
    *,
    task: OpenClawRolloutTask,
    agent_id: str,
    agent_message: str,
    config: dict[str, Any],
    paths,
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "message": agent_message,
        "user_message": task.message,
        "execution_mode": task.execution_mode,
        "require_generated_audio": task.require_generated_audio,
        "required_tool_names": list(task.required_tool_names),
        "required_command_substrings": list(task.required_command_substrings),
        "required_audio_fields": list(task.required_audio_fields),
        "expected_backend": task.expected_backend,
        "min_duration_sec": task.min_duration_sec,
        "require_audio_file_exists": task.require_audio_file_exists,
        "timeout_sec": task.timeout_sec,
        "max_turns": task.max_turns,
        "agent_id": agent_id,
        "workspace_dir": str(paths.workspace_dir),
        "skill_path": str(paths.skill_path),
        "source_skill_path": str(config.get("openclaw_skill_path") or config["openclaw_live_skill_path"]),
        "wrapper_path": str(paths.wrapper_path),
        "live_skill_path": str(config["openclaw_live_skill_path"]),
    }


def build_initial_message(task: OpenClawRolloutTask, paths) -> str:
    if task.execution_mode == "skill_driven":
        return build_skill_driven_rollout_message(task.message, paths)
    if task.execution_mode == "guided":
        return build_rollout_message(task.message, paths)
    raise ValueError(f"Unknown OpenClaw rollout execution_mode: {task.execution_mode}")


def run_one_rollout(config: dict[str, Any], run_dir: Path, task: OpenClawRolloutTask) -> dict[str, Any]:
    task_dir = run_dir / "rollouts" / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    agent_id: str | None = None
    try:
        agent_id, paths, add_result = setup_agent(config, task_dir, task.task_id)
        agent_message = build_initial_message(task, paths)
        save_json(
            task_dir / "task.json",
            build_task_record(
                task=task,
                agent_id=agent_id,
                agent_message=agent_message,
                config=config,
                paths=paths,
            ),
        )
        save_json(task_dir / "agent_setup.json", add_result)

        parsed: dict[str, Any] = {}
        for turn in range(1, task.max_turns + 1):
            task_result = run_agent_task(
                agent_id=agent_id,
                task_id=task.task_id,
                message=agent_message,
                timeout_sec=task.timeout_sec,
            )
            save_json(task_dir / f"openclaw_result_turn_{turn}.json", task_result)
            save_json(task_dir / "openclaw_result.json", task_result)
            if task_result["returncode"] != 0:
                raise RuntimeError(f"OpenClaw rollout failed: {task_result['stderr'] or task_result['stdout']}")

            export_result = export_trajectory(agent_id, task_result["session_key"], task_dir, f"{task.task_id}-turn-{turn}")
            save_json(task_dir / f"trajectory_export_turn_{turn}.json", export_result)
            save_json(task_dir / "trajectory_export.json", export_result)
            parsed = parse_trajectory_bundle(export_result["bundle_dir"])
            save_json(task_dir / f"parsed_result_turn_{turn}.json", parsed)
            save_json(task_dir / "parsed_result.json", parsed)
            try:
                validate_rollout_result(parsed, task)
                break
            except RuntimeError as exc:
                validation_error = str(exc)
                if turn == task.max_turns:
                    if config.get("continue_on_validation_error"):
                        break
                    raise
                agent_message = build_continuation_message(validation_error, paths)
        summary = summarize_rollout(task, task_dir, parsed)
        save_json(task_dir / "rollout_summary.json", summary)
        return summary
    finally:
        if agent_id:
            cleanup = delete_agent(agent_id)
            save_json(task_dir / "agent_cleanup.json", cleanup)
            if cleanup.get("returncode") != 0:
                LOGGER.warning("temporary OpenClaw agent cleanup failed: %s", cleanup.get("stderr") or cleanup.get("stdout"))
            shutil.rmtree(task_dir / "openclaw_workspace", ignore_errors=True)
            shutil.rmtree(task_dir / "openclaw_agent", ignore_errors=True)
            shutil.rmtree(task_dir / "trajectory_workspace", ignore_errors=True)


def run_openclaw_rollouts(config_path: str) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(config_path)
    run_dir = make_run_dir(config, "openclaw_rollout")
    tasks = load_rollout_tasks(config)
    LOGGER.info("starting OpenClaw rollout collection with %d task(s)", len(tasks))

    summaries: list[dict[str, Any]] = []
    for task in tasks:
        LOGGER.info("running OpenClaw rollout task %s", task.task_id)
        summaries.append(run_one_rollout(config, run_dir, task))
        save_json(run_dir / "rollout_report.json", {"config": config, "tasks": summaries})

    report = {
        "mode": "openclaw_rollout",
        "project_root": str(PROJECT_ROOT),
        "config": config,
        "num_tasks": len(tasks),
        "num_passed": sum(1 for row in summaries if row.get("passed")),
        "tasks": summaries,
    }
    save_json(run_dir / "rollout_report.json", report)
    save_json(run_dir / "summary.json", report)
    print(f"saved_run_dir={run_dir}")
    return run_dir

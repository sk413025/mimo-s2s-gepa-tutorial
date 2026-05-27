from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, load_config
from ..runner import make_run_dir, save_json
from .openclaw_skill import read_live_skill
from .trajectory_parser import parse_trajectory_bundle

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenClawRolloutPaths:
    run_dir: Path
    workspace_dir: Path
    agent_dir: Path
    skill_path: Path

    @property
    def wrapper_path(self) -> Path:
        return self.skill_path.parent / "scripts" / "mimo_audio_wrapper.py"


@dataclass(frozen=True)
class OpenClawRolloutTask:
    task_id: str
    message: str
    require_generated_audio: bool
    timeout_sec: int
    max_turns: int
    execution_mode: str
    required_tool_names: tuple[str, ...]
    required_command_substrings: tuple[str, ...]
    required_audio_fields: tuple[str, ...]
    expected_backend: str | None
    min_duration_sec: float | None
    require_audio_file_exists: bool


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "task"


def run_cli(args: list[str], timeout_sec: int | float = 120) -> subprocess.CompletedProcess[str]:
    LOGGER.info("running: %s", " ".join(args))
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )


def parse_last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            best = obj
    return best


def copy_skill_tree(skill_path: str | Path, live_skill_path: str | Path, workspace_dir: Path) -> Path:
    source_skill = Path(skill_path)
    live_skill = Path(live_skill_path)
    source_skill_dir = source_skill.parent
    live_skill_dir = live_skill.parent
    source_tree_dir = source_skill_dir if (source_skill_dir / "scripts").is_dir() else live_skill_dir
    target_dir = workspace_dir / "skills" / live_skill_dir.name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_tree_dir, target_dir)

    target_skill_path = target_dir / "SKILL.md"
    text = read_live_skill(source_skill)
    text = text.replace(str(source_skill_dir), str(target_dir))
    text = text.replace(str(live_skill_dir), str(target_dir))
    text = text.replace(str(live_skill_dir.parent.parent), str(workspace_dir))
    target_skill_path.write_text(text, encoding="utf-8")
    return target_skill_path


def write_workspace_guidance(paths: OpenClawRolloutPaths) -> None:
    guidance = f"""# SkillOpt OpenClaw Rollout Workspace

This temporary workspace is used to validate the workspace `mimo-audio` skill.

For any MiMo-Audio, LDV restoration, S2S, health, or smoke-test request:

- Read this exact workspace skill first: `{paths.skill_path}`.
- Treat that file as the source of truth.
- Do not read bundled skill paths under npm or global OpenClaw installs.
- Do not call `healthcheck`, `mimo-audio`, or `mimo_audio_s2s` as tools unless they are explicitly listed in the current tool set.
- Do not use `sessions_spawn` for this harness task.
- Use `exec` to run the wrapper documented by the skill.

Known wrapper commands:

```bash
python3 {paths.wrapper_path} health --model mimo_audio_s2s
python3 {paths.wrapper_path} s2s-smoke
```
"""
    paths.workspace_dir.mkdir(parents=True, exist_ok=True)
    (paths.workspace_dir / "AGENTS.md").write_text(guidance, encoding="utf-8")
    (paths.workspace_dir / "TOOLS.md").write_text(guidance, encoding="utf-8")


def setup_agent(config: dict[str, Any], run_dir: Path, task_id: str) -> tuple[str, OpenClawRolloutPaths, dict[str, Any]]:
    stamp = time.strftime("%Y%m%d%H%M%S")
    agent_id = f"skillopt-rollout-{stamp}-{slugify(task_id)[:24]}"
    workspace_dir = run_dir / "openclaw_workspace"
    agent_dir = run_dir / "openclaw_agent"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    add_result = run_cli(
        [
            "openclaw",
            "agents",
            "add",
            agent_id,
            "--workspace",
            str(workspace_dir),
            "--agent-dir",
            str(agent_dir),
            "--model",
            str(config.get("openclaw_agent_model", "vllm/google/gemma-4-E4B-it")),
            "--non-interactive",
            "--json",
        ],
        timeout_sec=120,
    )
    add_json = parse_last_json_object(add_result.stdout)
    if add_result.returncode != 0 or add_json is None:
        raise RuntimeError(f"failed to add OpenClaw agent: {add_result.stderr or add_result.stdout}")

    source_skill_path = config.get("openclaw_skill_path") or config["openclaw_live_skill_path"]
    skill_path = copy_skill_tree(source_skill_path, config["openclaw_live_skill_path"], workspace_dir)
    paths = OpenClawRolloutPaths(
        run_dir=run_dir,
        workspace_dir=workspace_dir,
        agent_dir=agent_dir,
        skill_path=skill_path,
    )
    write_workspace_guidance(paths)
    return agent_id, paths, {"stdout": add_result.stdout, "stderr": add_result.stderr, "json": add_json}


def delete_agent(agent_id: str) -> dict[str, Any]:
    result = run_cli(["openclaw", "agents", "delete", agent_id, "--force", "--json"], timeout_sec=120)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "json": parse_last_json_object(result.stdout),
    }


def run_agent_task(
    *,
    agent_id: str,
    task_id: str,
    message: str,
    timeout_sec: int,
) -> dict[str, Any]:
    session_key = f"agent:{agent_id}:{task_id}"
    result = run_cli(
        [
            "openclaw",
            "agent",
            "--agent",
            agent_id,
            "--session-key",
            session_key,
            "--message",
            message,
            "--json",
            "--timeout",
            str(timeout_sec),
        ],
        timeout_sec=timeout_sec + 60,
    )
    parsed = parse_last_json_object(result.stdout)
    return {
        "session_key": session_key,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "json": parsed,
    }


def build_rollout_message(message: str, paths: OpenClawRolloutPaths) -> str:
    return f"""Use tools now. Do not answer with a plan before tool calls.

Step 1: call the read tool on this exact workspace skill path:
{paths.skill_path}

Step 2: call the exec tool with this exact command and wait for completion:
python3 {paths.wrapper_path} health --model mimo_audio_s2s

Step 3: call the exec tool with this exact command and wait for completion:
python3 {paths.wrapper_path} s2s-smoke

If an exec call returns "Command still running", call the process tool to poll/log
the returned session until it completes. Do not stop after starting a background
process. Do not use sessions_spawn. Do not call healthcheck, mimo-audio, or
mimo_audio_s2s as tools. Do not read global/npm skill paths.

Final answer must report: resolved_request, error, backend, audio_path, audio_url,
duration_sec, text_channel, n_prompt_examples.

User task:
{message}
"""


def build_skill_driven_rollout_message(message: str, paths: OpenClawRolloutPaths) -> str:
    return f"""Use tools now. Do not answer with a plan before tool calls.

Step 1: call the read tool on this exact workspace skill path:
{paths.skill_path}

Step 2: follow that workspace skill to satisfy the user task below. Treat the
workspace skill as the source of truth for wrapper commands and reporting fields.
Do not read global/npm skill paths. Do not use sessions_spawn. Do not call
healthcheck, mimo-audio, or mimo_audio_s2s as tools unless they are explicitly
listed in the current tool set.

If an exec call returns "Command still running", call the process tool to
poll/log the returned session until it completes. Do not stop after starting a
background process.

Final answer must report: resolved_request, error, backend, audio_path,
audio_url, duration_sec, text_channel, n_prompt_examples.

User task:
{message}
"""


def build_continuation_message(last_error: str, paths: OpenClawRolloutPaths) -> str:
    return f"""Use tools now. The previous validation did not pass: {last_error}

Continue in this same session. Do not answer with a plan before tool calls.

Call exec with this exact command and wait for completion:
python3 {paths.wrapper_path} s2s-smoke

If exec returns "Command still running", call the process tool to poll/log the
returned session until it completes. Final answer must include audio_path,
audio_url, duration_sec, backend, text_channel, n_prompt_examples,
resolved_request, and error. Do not use sessions_spawn or global/npm skill paths.
"""


def export_trajectory(agent_id: str, session_key: str, run_dir: Path, output_name: str) -> dict[str, Any]:
    export_workspace = run_dir / "trajectory_workspace"
    export_workspace.mkdir(parents=True, exist_ok=True)
    result = run_cli(
        [
            "openclaw",
            "sessions",
            "export-trajectory",
            "--agent",
            agent_id,
            "--session-key",
            session_key,
            "--workspace",
            str(export_workspace),
            "--output",
            output_name,
            "--json",
        ],
        timeout_sec=120,
    )
    parsed = parse_last_json_object(result.stdout)
    if result.returncode != 0 or not parsed:
        raise RuntimeError(f"failed to export trajectory: {result.stderr or result.stdout}")

    output_dir = Path(parsed["outputDir"])
    target_dir = run_dir / "trajectory"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(output_dir, target_dir)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "json": parsed,
        "bundle_dir": str(target_dir),
    }


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

    if task.min_duration_sec is not None:
        duration = generated_audio.get("duration_sec")
        try:
            if float(duration) < task.min_duration_sec:
                errors.append(f"generated audio duration is too short: {duration}")
        except (TypeError, ValueError):
            errors.append(f"generated audio duration is not numeric: {duration}")

    if task.require_audio_file_exists:
        audio_path = generated_audio.get("audio_path_expanded") or generated_audio.get("audio_path")
        if not audio_path or not Path(str(audio_path)).expanduser().is_file():
            errors.append(f"generated audio file does not exist: {audio_path}")

    return errors


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_rollout_tasks(config: dict[str, Any]) -> list[OpenClawRolloutTask]:
    raw_tasks = load_jsonl(config["task_path"]) if config.get("task_path") else [config]
    tasks: list[OpenClawRolloutTask] = []
    for raw in raw_tasks:
        task_id = slugify(str(raw.get("task_id", f"task-{len(tasks) + 1}")))
        message = str(raw.get("task_message") or raw.get("message") or "").strip()
        if not message:
            raise ValueError(f"OpenClaw rollout task has no message: {task_id}")
        tasks.append(
            OpenClawRolloutTask(
                task_id=task_id,
                message=message,
                require_generated_audio=bool(raw.get("require_generated_audio", config.get("require_generated_audio", False))),
                timeout_sec=int(raw.get("openclaw_timeout_sec", config.get("openclaw_timeout_sec", 600))),
                max_turns=int(raw.get("openclaw_max_turns", config.get("openclaw_max_turns", 3))),
                execution_mode=str(raw.get("execution_mode", config.get("execution_mode", "guided"))),
                required_tool_names=tuple(raw.get("required_tool_names", config.get("required_tool_names", []))),
                required_command_substrings=tuple(
                    raw.get("required_command_substrings", config.get("required_command_substrings", []))
                ),
                required_audio_fields=tuple(raw.get("required_audio_fields", config.get("required_audio_fields", []))),
                expected_backend=raw.get("expected_backend", config.get("expected_backend")),
                min_duration_sec=(
                    float(raw["min_duration_sec"])
                    if "min_duration_sec" in raw
                    else (float(config["min_duration_sec"]) if "min_duration_sec" in config else None)
                ),
                require_audio_file_exists=bool(raw.get("require_audio_file_exists", config.get("require_audio_file_exists", False))),
            )
        )
    max_tasks = int(config.get("max_tasks", len(tasks)))
    return tasks[:max_tasks]


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


def run_one_rollout(config: dict[str, Any], run_dir: Path, task: OpenClawRolloutTask) -> dict[str, Any]:
    task_dir = run_dir / "rollouts" / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    agent_id: str | None = None
    cleanup: dict[str, Any] | None = None
    try:
        agent_id, paths, add_result = setup_agent(config, task_dir, task.task_id)
        if task.execution_mode == "skill_driven":
            agent_message = build_skill_driven_rollout_message(task.message, paths)
        elif task.execution_mode == "guided":
            agent_message = build_rollout_message(task.message, paths)
        else:
            raise ValueError(f"Unknown OpenClaw rollout execution_mode: {task.execution_mode}")
        task_record = {
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
        save_json(task_dir / "task.json", task_record)
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

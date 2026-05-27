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
    if parsed.get("final_status") != "success":
        raise RuntimeError(f"OpenClaw trajectory final status is not success: {parsed.get('final_status')}")
    if task.require_generated_audio and not parsed.get("generated_audio"):
        raise RuntimeError("OpenClaw rollout completed, but no generated audio was found in the trajectory")


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
            )
        )
    max_tasks = int(config.get("max_tasks", len(tasks)))
    return tasks[:max_tasks]


def summarize_rollout(task: OpenClawRolloutTask, task_dir: Path, parsed: dict[str, Any]) -> dict[str, Any]:
    generated_audio = parsed.get("generated_audio") or {}
    return {
        "task_id": task.task_id,
        "task_dir": str(task_dir),
        "final_status": parsed.get("final_status"),
        "event_count": parsed.get("event_count"),
        "tool_call_count": len(parsed.get("tool_calls", [])),
        "tool_result_count": len(parsed.get("tool_results", [])),
        "generated_audio": generated_audio,
        "passed": parsed.get("final_status") == "success" and (not task.require_generated_audio or bool(generated_audio)),
    }


def run_one_rollout(config: dict[str, Any], run_dir: Path, task: OpenClawRolloutTask) -> dict[str, Any]:
    task_dir = run_dir / "rollouts" / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    agent_id: str | None = None
    cleanup: dict[str, Any] | None = None
    try:
        agent_id, paths, add_result = setup_agent(config, task_dir, task.task_id)
        agent_message = build_rollout_message(task.message, paths)
        task_record = {
            "task_id": task.task_id,
            "message": agent_message,
            "user_message": task.message,
            "require_generated_audio": task.require_generated_audio,
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

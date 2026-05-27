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
from .trajectory import parse_trajectory_bundle

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenClawTaskPaths:
    run_dir: Path
    workspace_dir: Path
    agent_dir: Path
    skill_path: Path

    @property
    def wrapper_path(self) -> Path:
        return self.skill_path.parent / "scripts" / "mimo_audio_wrapper.py"


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


def copy_skill_tree(live_skill_path: str | Path, workspace_dir: Path) -> Path:
    live_skill = Path(live_skill_path)
    live_skill_dir = live_skill.parent
    target_dir = workspace_dir / "skills" / live_skill_dir.name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(live_skill_dir, target_dir)

    skill_path = target_dir / "SKILL.md"
    text = read_live_skill(skill_path)
    text = text.replace(str(live_skill_dir), str(target_dir))
    text = text.replace(str(live_skill_dir.parent.parent), str(workspace_dir))
    skill_path.write_text(text, encoding="utf-8")
    return skill_path


def write_workspace_guidance(paths: OpenClawTaskPaths) -> None:
    guidance = f"""# SkillOpt OpenClaw Task Workspace

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


def setup_agent(config: dict[str, Any], run_dir: Path) -> tuple[str, OpenClawTaskPaths, dict[str, Any]]:
    stamp = time.strftime("%Y%m%d%H%M%S")
    agent_id = f"skillopt-task-{stamp}"
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

    skill_path = copy_skill_tree(config["openclaw_live_skill_path"], workspace_dir)
    paths = OpenClawTaskPaths(
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


def build_task_message(message: str, paths: OpenClawTaskPaths) -> str:
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


def build_continuation_message(last_error: str, paths: OpenClawTaskPaths) -> str:
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


def validate_parsed_result(parsed: dict[str, Any], config: dict[str, Any]) -> None:
    if parsed.get("final_status") != "success":
        raise RuntimeError(f"OpenClaw trajectory final status is not success: {parsed.get('final_status')}")
    if config.get("require_generated_audio", False) and not parsed.get("generated_audio"):
        raise RuntimeError("OpenClaw task completed, but no generated audio was found in the trajectory")


def run_openclaw_task(config_path: str) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(config_path)
    run_dir = make_run_dir(config, "openclaw_task")
    task_id = slugify(str(config.get("task_id", "openclaw-task")))
    message = str(config["task_message"])
    timeout_sec = int(config.get("openclaw_timeout_sec", 600))
    max_turns = int(config.get("openclaw_max_turns", 3))

    agent_id: str | None = None
    cleanup: dict[str, Any] | None = None
    validation_error: str | None = None
    try:
        agent_id, paths, add_result = setup_agent(config, run_dir)
        agent_message = build_task_message(message, paths)
        task = {
            "task_id": task_id,
            "message": agent_message,
            "user_message": message,
            "agent_id": agent_id,
            "workspace_dir": str(paths.workspace_dir),
            "skill_path": str(paths.skill_path),
            "wrapper_path": str(paths.wrapper_path),
            "live_skill_path": str(config["openclaw_live_skill_path"]),
        }
        save_json(run_dir / "task.json", task)
        save_json(run_dir / "agent_setup.json", add_result)

        for turn in range(1, max_turns + 1):
            task_result = run_agent_task(
                agent_id=agent_id,
                task_id=task_id,
                message=agent_message,
                timeout_sec=timeout_sec,
            )
            save_json(run_dir / f"openclaw_result_turn_{turn}.json", task_result)
            save_json(run_dir / "openclaw_result.json", task_result)
            if task_result["returncode"] != 0:
                raise RuntimeError(f"OpenClaw task failed: {task_result['stderr'] or task_result['stdout']}")

            export_result = export_trajectory(agent_id, task_result["session_key"], run_dir, f"{task_id}-turn-{turn}")
            save_json(run_dir / f"trajectory_export_turn_{turn}.json", export_result)
            save_json(run_dir / "trajectory_export.json", export_result)
            parsed = parse_trajectory_bundle(export_result["bundle_dir"])
            save_json(run_dir / f"parsed_result_turn_{turn}.json", parsed)
            save_json(run_dir / "parsed_result.json", parsed)
            try:
                validate_parsed_result(parsed, config)
                validation_error = None
                break
            except RuntimeError as exc:
                validation_error = str(exc)
                if turn == max_turns:
                    raise
                agent_message = build_continuation_message(validation_error, paths)
    finally:
        if agent_id:
            cleanup = delete_agent(agent_id)
            save_json(run_dir / "agent_cleanup.json", cleanup)
            shutil.rmtree(run_dir / "openclaw_workspace", ignore_errors=True)
            shutil.rmtree(run_dir / "openclaw_agent", ignore_errors=True)
            shutil.rmtree(run_dir / "trajectory_workspace", ignore_errors=True)

    if cleanup and cleanup.get("returncode") != 0:
        LOGGER.warning("temporary OpenClaw agent cleanup failed: %s", cleanup.get("stderr") or cleanup.get("stdout"))

    print(f"saved_run_dir={run_dir}")
    return run_dir

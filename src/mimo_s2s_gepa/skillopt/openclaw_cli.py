from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from .openclaw_skill import read_live_skill
from .rollout_tasks import slugify

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

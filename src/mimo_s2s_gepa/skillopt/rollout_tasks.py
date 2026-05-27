from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    require_audio_file_exists: bool


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "task"


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
                require_audio_file_exists=bool(raw.get("require_audio_file_exists", config.get("require_audio_file_exists", False))),
            )
        )
    max_tasks = int(config.get("max_tasks", len(tasks)))
    return tasks[:max_tasks]

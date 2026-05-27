from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_json_text(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def expand_home_path(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("~/"):
        return str(Path(value).expanduser())
    return value


def extract_field(text: str, field: str) -> str | None:
    patterns = [
        rf'"{re.escape(field)}"\s*:\s*"([^"\n]+)"',
        rf"{re.escape(field)}\s*:\s*([^\n,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().strip('"')
    return None


def extract_generated_audio_from_text(text: str) -> dict[str, Any] | None:
    audio_path = extract_field(text, "audio_path")
    if not audio_path or ".wav" not in audio_path:
        return None

    duration_text = extract_field(text, "duration_sec")
    examples_text = extract_field(text, "n_prompt_examples")
    generated: dict[str, Any] = {
        "audio_path": audio_path,
        "audio_path_expanded": expand_home_path(audio_path),
        "audio_url": extract_field(text, "audio_url"),
        "backend": extract_field(text, "backend"),
        "duration_sec": None,
        "text_channel": extract_field(text, "text_channel"),
        "n_prompt_examples": None,
        "error": extract_field(text, "error"),
        "source": "text",
    }
    if duration_text:
        try:
            generated["duration_sec"] = float(duration_text)
        except ValueError:
            generated["duration_sec"] = duration_text
    if examples_text:
        try:
            generated["n_prompt_examples"] = int(examples_text)
        except ValueError:
            generated["n_prompt_examples"] = examples_text
    return {key: value for key, value in generated.items() if value is not None}


def tool_result_text(event: dict[str, Any]) -> str:
    message = event.get("data", {}).get("message", {})
    parts = message.get("content") or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "\n".join(text for text in texts if text)


def parse_trajectory_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir)
    manifest = load_json(root / "manifest.json")
    artifacts = load_json(root / "artifacts.json")
    events = iter_jsonl(root / "events.jsonl")

    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    parsed_tool_results: list[Any] = []

    for event in events:
        event_type = event.get("type")
        data = event.get("data", {})
        if event_type == "tool.call":
            tool_calls.append(
                {
                    "seq": event.get("seq"),
                    "tool_call_id": data.get("toolCallId"),
                    "name": data.get("name"),
                    "arguments": data.get("arguments"),
                }
            )
        elif event_type == "tool.result":
            text = tool_result_text(event)
            parsed = parse_json_text(text)
            if parsed is not None:
                parsed_tool_results.append(parsed)
            message = data.get("message", {})
            tool_results.append(
                {
                    "seq": event.get("seq"),
                    "tool_call_id": message.get("toolCallId"),
                    "tool_name": message.get("toolName"),
                    "is_error": message.get("isError", False),
                    "text": text,
                    "parsed_json": parsed,
                    "details": message.get("details", {}),
                }
            )

    generated_audio = next(
        (
            result
            for result in reversed(parsed_tool_results)
            if isinstance(result, dict) and result.get("audio_path")
        ),
        None,
    )
    if isinstance(generated_audio, dict):
        generated_audio = {
            **generated_audio,
            "audio_path_expanded": expand_home_path(str(generated_audio.get("audio_path", ""))),
            "source": "json",
        }

    assistant_texts = artifacts.get("assistantTexts", [])
    if not generated_audio:
        text_sources: list[str] = []
        for result in reversed(tool_results):
            text_sources.append(result.get("text", ""))
            details = result.get("details", {})
            if isinstance(details, dict):
                text_sources.append(str(details.get("aggregated", "")))
        text_sources.extend(reversed([text for text in assistant_texts if isinstance(text, str)]))
        generated_audio = next(
            (extracted for text in text_sources if (extracted := extract_generated_audio_from_text(text))),
            None,
        )

    return {
        "bundle_dir": str(root),
        "session_id": manifest.get("sessionId"),
        "session_key": manifest.get("sessionKey"),
        "event_count": manifest.get("eventCount"),
        "runtime_event_count": manifest.get("runtimeEventCount"),
        "transcript_event_count": manifest.get("transcriptEventCount"),
        "final_status": artifacts.get("finalStatus"),
        "assistant_texts": assistant_texts,
        "tool_metas": artifacts.get("toolMetas", []),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "generated_audio": generated_audio,
    }

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse, urlunparse

import dspy

from .counters import COUNTERS


def normalize_audio_url(result: dict[str, Any], api_base: str) -> dict[str, Any]:
    audio_url = result.get("audio_url")
    if not isinstance(audio_url, str):
        return result

    parsed_audio_url = urlparse(audio_url)
    if parsed_audio_url.hostname not in {"127.0.0.1", "localhost"}:
        return result

    parsed_api_base = urlparse(api_base)
    if not parsed_api_base.scheme or not parsed_api_base.netloc:
        return result

    normalized = urlunparse(
        (
            parsed_api_base.scheme,
            parsed_api_base.netloc,
            parsed_audio_url.path,
            parsed_audio_url.params,
            parsed_audio_url.query,
            parsed_audio_url.fragment,
        )
    )
    return {**result, "audio_url": normalized}


def call_mimo_s2s(
    *,
    model: str,
    api_base: str,
    api_key: str,
    instruction: str,
    audio_path: str,
    prompt_examples_json: str,
    decoding: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
    COUNTERS["mimo_s2s_calls"] += 1
    lm = dspy.LM(model, api_base=api_base, api_key=api_key, cache=False)
    outputs = lm(
        messages=[
            {
                "role": "user",
                "content": {
                    "instruction": instruction,
                    "audio_path": audio_path,
                    "prompt_examples_json": prompt_examples_json,
                    "return_diagnostics": True,
                    **decoding,
                },
            }
        ],
        cache=False,
        timeout=timeout_sec + 10,
        timeout_sec=timeout_sec,
    )
    content = outputs[0] if isinstance(outputs, list) else outputs
    result = json.loads(content) if isinstance(content, str) else content
    return normalize_audio_url(result, api_base) if isinstance(result, dict) else result

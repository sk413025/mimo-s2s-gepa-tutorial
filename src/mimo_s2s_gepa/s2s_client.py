from __future__ import annotations

import json
from typing import Any

import dspy

from .counters import COUNTERS


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
    return json.loads(content) if isinstance(content, str) else content


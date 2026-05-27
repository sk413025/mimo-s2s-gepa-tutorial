from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def mean_score(rows: list[dict[str, Any]]) -> float:
    scores = [float(row.get("score", 0.0)) for row in rows]
    return sum(scores) / len(scores) if scores else 0.0


def report_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_examples": len(rows),
        "mean_score": mean_score(rows),
        "example_ids": [row.get("id") for row in rows],
        "audio_paths": [row.get("audio_path") for row in rows if row.get("audio_path")],
    }

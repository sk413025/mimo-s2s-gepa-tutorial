from __future__ import annotations

from pathlib import Path


def read_live_skill(path: str | Path) -> str:
    skill_path = Path(path)
    if not skill_path.is_file():
        raise FileNotFoundError(f"OpenClaw skill file does not exist: {skill_path}")
    text = skill_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"OpenClaw skill file is empty: {skill_path}")
    return text

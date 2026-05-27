from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillSnapshot:
    live_path: Path
    initial_path: Path
    text: str


@dataclass(frozen=True)
class CandidateSkill:
    skill_path: Path
    diff_path: Path


def read_live_skill(path: str | Path) -> str:
    skill_path = Path(path)
    if not skill_path.is_file():
        raise FileNotFoundError(f"OpenClaw skill file does not exist: {skill_path}")
    text = skill_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"OpenClaw skill file is empty: {skill_path}")
    return text


def snapshot_live_skill(live_path: str | Path, run_dir: Path) -> SkillSnapshot:
    live = Path(live_path)
    text = read_live_skill(live)
    initial_dir = run_dir / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)
    initial_path = initial_dir / "SKILL.md"
    initial_path.write_text(text, encoding="utf-8")
    return SkillSnapshot(live_path=live, initial_path=initial_path, text=text)


def make_candidate(run_dir: Path, text: str, diff: str) -> CandidateSkill:
    candidate_dir = run_dir / "candidates" / "skill_v0001"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    skill_path = candidate_dir / "SKILL.md"
    diff_path = candidate_dir / "diff.md"
    skill_path.write_text(text, encoding="utf-8")
    diff_path.write_text(diff, encoding="utf-8")
    return CandidateSkill(skill_path=skill_path, diff_path=diff_path)

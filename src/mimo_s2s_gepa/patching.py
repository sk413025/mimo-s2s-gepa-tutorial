from __future__ import annotations

import difflib


FORBIDDEN_TERMS = (
    "tts",
    "text-to-speech",
    "mimo_audio_tts",
    "instruct_tts",
    "speech audio generation",
)


def clean_candidate_skill(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "---":
            return "\n".join(lines[index:]).strip() + "\n"
    return text + "\n"


def validate_candidate_skill(text: str) -> list[str]:
    errors: list[str] = []
    stripped = text.strip()
    lower = stripped.lower()

    if not stripped:
        errors.append("candidate skill is empty")
    if not stripped.startswith("---"):
        errors.append("candidate skill must start with YAML frontmatter")
    if "name: mimo-audio" not in lower:
        errors.append("candidate skill must keep name: mimo-audio")
    if "s2s" not in lower and "speech-to-speech" not in lower:
        errors.append("candidate skill must stay scoped to S2S")
    if "mimo_audio_s2s" not in lower:
        errors.append("candidate skill must keep the mimo_audio_s2s backend model reference")

    for term in FORBIDDEN_TERMS:
        if term in lower:
            errors.append(f"candidate skill contains forbidden non-S2S term: {term}")

    return errors


def build_unified_diff(before: str, after: str, before_name: str = "initial/SKILL.md", after_name: str = "candidate/SKILL.md") -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=before_name,
        tofile=after_name,
        lineterm="",
    )
    return "\n".join(diff) + "\n"

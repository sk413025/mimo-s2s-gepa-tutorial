from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy

FORBIDDEN_TERMS = (
    "tts",
    "text-to-speech",
    "mimo_audio_tts",
    "instruct_tts",
    "speech audio generation",
)


class ProposeSkillEdits(dspy.Signature):
    """Propose small structured edits to the OpenClaw SKILL.md from SkillOpt feedback."""

    current_skill_md: str = dspy.InputField()
    trajectory_feedback: str = dspy.InputField()
    constraints: str = dspy.InputField()
    edits_json: str = dspy.OutputField(
        desc=(
            "A valid JSON array of 1-3 edit objects. Allowed ops: "
            "append_after_heading, insert_after_text, replace_text, append_to_end. "
            "Keep edits small and grounded in trajectory feedback."
        )
    )
    proposal_notes_json: str = dspy.OutputField(
        desc=(
            "A valid JSON object with keys changed_sections, rationale, risks, "
            "validation_focus. Each value should be an array of concise strings."
        )
    )


def find_latest_analysis_dir(output_dir: str | Path) -> Path:
    candidates = sorted(Path(output_dir).glob("*_trajectory_analysis"))
    if not candidates:
        raise FileNotFoundError(f"No trajectory analysis runs found under {output_dir}")
    return candidates[-1]


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def clean_candidate_skill(raw_text: str) -> str:
    text = strip_code_fence(raw_text)
    lines = text.splitlines()
    if text.count("```") % 2 == 1 and lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
        text = "\n".join(lines)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "---":
            return "\n".join(lines[index:]).strip() + "\n"
    return text.strip() + "\n"


def parse_notes_json(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"raw": cleaned}
        parsed = json.loads(cleaned[start : end + 1])
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


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
    if stripped.count("```") % 2 != 0:
        errors.append("candidate skill has unbalanced markdown code fences")

    for term in FORBIDDEN_TERMS:
        if term in lower:
            errors.append(f"candidate skill contains forbidden non-S2S term: {term}")

    return errors


def build_proposer_lm(config: dict[str, Any]) -> dspy.LM:
    return dspy.LM(
        config["gemma_model"],
        api_base=config["gemma_base_url"],
        api_key=config.get("gemma_api_key", "sk-local"),
        temperature=float(config.get("proposer_temperature", 0.2)),
        max_tokens=int(config.get("proposer_max_tokens", 5000)),
        cache=False,
    )


def proposal_constraints() -> str:
    return "\n".join(
        [
            "Return only a JSON array of 1-3 edit objects for SKILL.md.",
            "Do not return a complete SKILL.md.",
            "Do not wrap the JSON in markdown code fences.",
            "Allowed edit ops:",
            "- append_after_heading: {\"op\":\"append_after_heading\",\"heading\":\"## Workflow\",\"text\":\"...\"}",
            "- insert_after_text: {\"op\":\"insert_after_text\",\"anchor\":\"exact existing text\",\"text\":\"...\"}",
            "- replace_text: {\"op\":\"replace_text\",\"old_text\":\"exact existing text\",\"new_text\":\"...\"}",
            "- append_to_end: {\"op\":\"append_to_end\",\"text\":\"...\"}",
            "Make at least one small real content edit grounded in the trajectory feedback.",
            "Prefer append_after_heading or insert_after_text over replacing large sections.",
            "Do not edit YAML frontmatter or change `name: mimo-audio`.",
            "Keep the skill focused only on MiMo-Audio S2S / LDV restoration.",
            "Do not add TTS, generic audio generation, chat, or unrelated tool workflows.",
            "Keep wrapper commands executable with the existing absolute skill path.",
            "Keep `mimo_audio_s2s`, health check, s2s, and s2s-smoke guidance.",
            "Prefer small targeted edits grounded in trajectory feedback.",
            "If feedback is mostly successful, clarify robustness and validation guidance instead of rewriting everything.",
        ]
    )

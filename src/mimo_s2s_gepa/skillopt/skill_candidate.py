from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Any

import dspy

from ..config import PROJECT_ROOT, load_config, resolve_path
from ..runner import make_run_dir, save_json
from .openclaw_skill import read_live_skill
from .trajectory_analyzer import load_json

LOGGER = logging.getLogger(__name__)

FORBIDDEN_TERMS = (
    "tts",
    "text-to-speech",
    "mimo_audio_tts",
    "instruct_tts",
    "speech audio generation",
)


class ProposeSkillCandidate(dspy.Signature):
    """Propose a complete OpenClaw SKILL.md candidate from SkillOpt feedback."""

    current_skill_md: str = dspy.InputField()
    trajectory_feedback: str = dspy.InputField()
    constraints: str = dspy.InputField()
    candidate_skill_md: str = dspy.OutputField(
        desc=(
            "A complete replacement SKILL.md. Preserve YAML frontmatter and "
            "`name: mimo-audio`. Keep it scoped to MiMo-Audio S2S / LDV "
            "restoration. Do not introduce TTS or unrelated workflows."
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


def resolve_analysis_dir(config: dict[str, Any]) -> Path:
    configured = str(config.get("trajectory_analysis_dir") or "").strip()
    if configured:
        return resolve_path(configured)
    return find_latest_analysis_dir(config["output_dir"])


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


def build_unified_diff(before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="initial/SKILL.md",
        tofile="candidates/skill_v0001/SKILL.md",
        lineterm="",
    )
    return "\n".join(diff) + "\n"


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
            "Return a complete SKILL.md, not a patch.",
            "Do not wrap the SKILL.md in markdown code fences.",
            "Make at least one small real content edit grounded in the trajectory feedback.",
            "Preserve YAML frontmatter and keep `name: mimo-audio`.",
            "Keep the skill focused only on MiMo-Audio S2S / LDV restoration.",
            "Do not add TTS, generic audio generation, chat, or unrelated tool workflows.",
            "Keep wrapper commands executable with the existing absolute skill path.",
            "Keep `mimo_audio_s2s`, health check, s2s, and s2s-smoke guidance.",
            "Prefer small targeted edits grounded in trajectory feedback.",
            "If feedback is mostly successful, clarify robustness and validation guidance instead of rewriting everything.",
        ]
    )


def write_candidate_outputs(
    *,
    run_dir: Path,
    initial_skill: str,
    candidate_skill: str,
    proposal: dict[str, Any],
) -> None:
    initial_dir = run_dir / "initial"
    candidate_dir = run_dir / "candidates" / "skill_v0001"
    initial_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    (initial_dir / "SKILL.md").write_text(initial_skill, encoding="utf-8")
    (candidate_dir / "SKILL.md").write_text(candidate_skill, encoding="utf-8")
    (candidate_dir / "diff.md").write_text(build_unified_diff(initial_skill, candidate_skill), encoding="utf-8")
    save_json(run_dir / "proposal.json", proposal)
    save_json(run_dir / "summary.json", proposal)


def propose_candidate(config: dict[str, Any], analysis_dir: Path, run_dir: Path) -> dict[str, Any]:
    skill_path = resolve_path(config["openclaw_live_skill_path"])
    initial_skill = read_live_skill(skill_path)
    feedback = load_json(analysis_dir / "trajectory_feedback.json")
    feedback_text = json.dumps(feedback, ensure_ascii=False, indent=2)

    proposer = dspy.Predict(ProposeSkillCandidate)
    with dspy.context(lm=build_proposer_lm(config)):
        prediction = proposer(
            current_skill_md=initial_skill,
            trajectory_feedback=feedback_text,
            constraints=proposal_constraints(),
        )

    candidate_skill = clean_candidate_skill(str(prediction.candidate_skill_md))
    validation_errors = validate_candidate_skill(candidate_skill)
    if initial_skill.strip() == candidate_skill.strip():
        validation_errors.append("candidate skill must contain at least one real content change")
    proposal_notes = parse_notes_json(str(prediction.proposal_notes_json))
    diff_text = build_unified_diff(initial_skill, candidate_skill)

    proposal = {
        "mode": "skill_candidate",
        "project_root": str(PROJECT_ROOT),
        "analysis_dir": str(analysis_dir),
        "live_skill_path": str(skill_path),
        "initial_skill_path": str(run_dir / "initial" / "SKILL.md"),
        "candidate_skill_path": str(run_dir / "candidates" / "skill_v0001" / "SKILL.md"),
        "candidate_diff_path": str(run_dir / "candidates" / "skill_v0001" / "diff.md"),
        "validation_errors": validation_errors,
        "changed": initial_skill.strip() != candidate_skill.strip(),
        "diff_line_count": len(
            [
                line
                for line in diff_text.splitlines()
                if (line.startswith("+") and not line.startswith("+++"))
                or (line.startswith("-") and not line.startswith("---"))
            ]
        ),
        "proposal_notes": proposal_notes,
        "raw_candidate_skill_md": str(prediction.candidate_skill_md),
        "raw_proposal_notes_json": str(prediction.proposal_notes_json),
    }
    write_candidate_outputs(
        run_dir=run_dir,
        initial_skill=initial_skill,
        candidate_skill=candidate_skill,
        proposal=proposal,
    )

    if validation_errors:
        raise RuntimeError(f"candidate skill failed validation: {validation_errors}")
    return proposal


def run_skill_candidate_proposal(config_path: str) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(config_path)
    analysis_dir = resolve_analysis_dir(config)
    run_dir = make_run_dir(config, "skill_candidate")
    LOGGER.info("proposing skill candidate from %s", analysis_dir)
    propose_candidate(config, analysis_dir, run_dir)
    print(f"saved_run_dir={run_dir}")
    return run_dir

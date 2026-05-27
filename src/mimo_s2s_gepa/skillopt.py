from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import dspy

from .config import PROJECT_ROOT, load_config
from .data import load_examples
from .metrics import build_gemma_metric
from .openclaw_skill import CandidateSkill, make_candidate, promote_candidate, snapshot_live_skill
from .patching import build_unified_diff, clean_candidate_skill, validate_candidate_skill
from .program import MiMoS2SProgram
from .reports import mean_score, report_summary, save_json
from .runner import configure_logging, configure_task_lm, make_run_dir, prediction_to_dict

LOGGER = logging.getLogger(__name__)


class ProposeSkillRewrite(dspy.Signature):
    """Rewrite an OpenClaw SKILL.md for MiMo-Audio S2S based on validation feedback."""

    current_skill_md: str = dspy.InputField()
    task_context: str = dspy.InputField()
    run_feedback: str = dspy.InputField()
    candidate_skill_md: str = dspy.OutputField(
        desc=(
            "A complete replacement SKILL.md. Preserve YAML frontmatter, keep name: mimo-audio, "
            "stay scoped to MiMo-Audio S2S / LDV restoration, and do not add unrelated workflows."
        )
    )


def split_train_val(examples: list[dspy.Example], train_size: int, val_size: int) -> tuple[list[dspy.Example], list[dspy.Example]]:
    train_count = max(1, train_size)
    val_count = max(1, val_size)
    trainset = examples[:train_count]
    valset = examples[train_count : train_count + val_count]
    if not valset:
        valset = examples[:val_count]
    return trainset, valset


def build_proposer_lm(config: dict[str, Any]) -> dspy.LM:
    return dspy.LM(
        config["gemma_model"],
        api_base=config["gemma_base_url"],
        api_key=config.get("gemma_api_key", "sk-local"),
        temperature=float(config.get("proposer_temperature", 0.3)),
        max_tokens=int(config.get("proposer_max_tokens", 3200)),
        cache=False,
    )


def task_context_from_examples(examples: list[dspy.Example]) -> str:
    lines = [
        "Optimize the OpenClaw skill for these MiMo-Audio S2S LDV restoration examples.",
        "The skill should help write concise MiMo native instructions for the S2S wrapper.",
    ]
    for example in examples:
        lines.append(
            f"- id={example.id}; expected_transcript={example.expected_transcript}; "
            f"input_note={example.input_note}"
        )
    return "\n".join(lines)


def feedback_from_report(label: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"{label} mean_score={mean_score(rows):.3f}"]
    for row in rows:
        lines.extend(
            [
                f"sample={row.get('id')}",
                f"instruction={row.get('instruction')}",
                f"score={row.get('score')}",
                f"feedback={row.get('feedback')}",
            ]
        )
    return "\n".join(lines)


def evaluate_skill(
    *,
    config: dict[str, Any],
    examples: list[dspy.Example],
    skill_text: str,
    metric_fn,
) -> list[dict[str, Any]]:
    program = MiMoS2SProgram(config)
    rows: list[dict[str, Any]] = []
    for example in examples:
        LOGGER.info("validating sample %s with skill context", example.id)
        prediction = program(**example.inputs(), skill_context=skill_text)
        rows.append(prediction_to_dict(example, prediction, metric_fn))
    return rows


def propose_candidate_skill(
    *,
    config: dict[str, Any],
    current_skill_md: str,
    task_context: str,
    run_feedback: str,
) -> str:
    proposer = dspy.Predict(ProposeSkillRewrite)
    with dspy.context(lm=build_proposer_lm(config)):
        proposal = proposer(
            current_skill_md=current_skill_md,
            task_context=task_context,
            run_feedback=run_feedback,
        )
    return clean_candidate_skill(str(proposal.candidate_skill_md))


def write_candidate_or_reject(
    *,
    run_dir: Path,
    initial_skill: str,
    candidate_skill: str,
) -> tuple[CandidateSkill | None, list[str]]:
    validation_errors = validate_candidate_skill(candidate_skill)
    if validation_errors:
        return None, validation_errors

    diff = build_unified_diff(initial_skill, candidate_skill)
    return make_candidate(run_dir, candidate_skill, diff), []


def run_skillopt(config_path: str) -> Path:
    configure_logging()
    config = load_config(config_path)
    LOGGER.info("starting skillopt run with %s", config["_config_path"])
    run_dir = make_run_dir(config, "skillopt")

    snapshot = snapshot_live_skill(config["openclaw_skill_path"], run_dir)
    examples = load_examples(config["data_path"])
    trainset, valset = split_train_val(
        examples,
        train_size=int(config.get("train_size", 1)),
        val_size=int(config.get("val_size", 1)),
    )

    configure_task_lm(config)
    metric_fn = build_gemma_metric(config)

    training_rows = evaluate_skill(config=config, examples=trainset, skill_text=snapshot.text, metric_fn=metric_fn)
    save_json(run_dir / "training_report.json", {"summary": report_summary(training_rows), "outputs": training_rows})

    baseline_rows = evaluate_skill(config=config, examples=valset, skill_text=snapshot.text, metric_fn=metric_fn)
    save_json(
        run_dir / "baseline_report.json",
        {
            "summary": report_summary(baseline_rows),
            "outputs": baseline_rows,
            "note": "Baseline validation score for the live OpenClaw skill snapshot.",
        },
    )

    candidate_text = propose_candidate_skill(
        config=config,
        current_skill_md=snapshot.text,
        task_context=task_context_from_examples(trainset),
        run_feedback=feedback_from_report("training", training_rows),
    )
    candidate, validation_errors = write_candidate_or_reject(
        run_dir=run_dir,
        initial_skill=snapshot.text,
        candidate_skill=candidate_text,
    )

    candidate_rows: list[dict[str, Any]] = []
    if candidate:
        candidate_rows = evaluate_skill(config=config, examples=valset, skill_text=candidate_text, metric_fn=metric_fn)
        save_json(
            run_dir / "candidate_report.json",
            {
                "summary": report_summary(candidate_rows),
                "outputs": candidate_rows,
                "candidate_skill_path": str(candidate.skill_path),
                "candidate_diff_path": str(candidate.diff_path),
            },
        )
    else:
        rejected_dir = run_dir / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        (rejected_dir / "candidate_raw.md").write_text(candidate_text, encoding="utf-8")
        save_json(
            run_dir / "candidate_report.json",
            {"summary": {"num_examples": 0, "mean_score": 0.0}, "outputs": [], "validation_errors": validation_errors},
        )

    baseline_score = mean_score(baseline_rows)
    candidate_score = mean_score(candidate_rows)
    accepted = bool(candidate and candidate_score > baseline_score)
    promoted = False
    if accepted and bool(config.get("allow_live_promote", False)) and candidate:
        promote_candidate(candidate, snapshot.live_path)
        promoted = True

    decision = {
        "accepted": accepted,
        "promoted_to_live_skill": promoted,
        "reason": (
            "candidate score improved over baseline"
            if accepted
            else "candidate rejected because validation failed or score did not improve"
        ),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "validation_errors": validation_errors,
        "live_skill_path": str(snapshot.live_path),
        "initial_skill_path": str(snapshot.initial_path),
        "candidate_skill_path": str(candidate.skill_path) if candidate else None,
    }
    save_json(run_dir / "decision.json", decision)

    summary = {
        "mode": "skillopt",
        "project_root": str(PROJECT_ROOT),
        "config": config,
        "train_size": len(trainset),
        "val_size": len(valset),
        "training": report_summary(training_rows),
        "baseline": report_summary(baseline_rows),
        "candidate": report_summary(candidate_rows),
        "decision": decision,
    }
    save_json(run_dir / "summary.json", summary)
    LOGGER.info("saved skillopt output to %s", run_dir)
    print(f"saved_run_dir={run_dir}")
    return run_dir

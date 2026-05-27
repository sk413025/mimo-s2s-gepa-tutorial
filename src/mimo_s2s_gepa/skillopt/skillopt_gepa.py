from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from ..config import PROJECT_ROOT, load_config, resolve_path
from ..runner import make_run_dir, save_json
from .openclaw_skill import read_live_skill
from .skill_candidate import (
    ProposeSkillEdits,
    build_proposer_lm,
    clean_candidate_skill,
    find_latest_analysis_dir,
    parse_notes_json,
    proposal_constraints,
    validate_candidate_skill,
)
from .patching import apply_skill_edits, parse_edits_json, validate_edits
from .registry import append_registry_record, build_candidate_record, build_validation_record
from .trajectory_analyzer import load_json
from .validation_gate import build_decision, run_rollout_set

LOGGER = logging.getLogger(__name__)


class SkillCandidateProgram(dspy.Module):
    """DSPy program GEPA can optimize for SkillOpt candidate proposal."""

    def __init__(self):
        super().__init__()
        self.proposer = dspy.Predict(ProposeSkillEdits)

    def forward(
        self,
        current_skill_md: str,
        trajectory_feedback: str,
        constraints: str,
    ) -> dspy.Prediction:
        prediction = self.proposer(
            current_skill_md=current_skill_md,
            trajectory_feedback=trajectory_feedback,
            constraints=constraints,
        )
        return dspy.Prediction(
            edits_json=str(prediction.edits_json),
            proposal_notes_json=str(prediction.proposal_notes_json),
        )


def resolve_analysis_dir(config: dict[str, Any]) -> Path:
    configured = str(config.get("trajectory_analysis_dir") or "").strip()
    if configured:
        return resolve_path(configured)
    return find_latest_analysis_dir(config["output_dir"])


def build_skillopt_example(config: dict[str, Any], analysis_dir: Path) -> dspy.Example:
    skill_path = resolve_path(config["openclaw_live_skill_path"])
    feedback = load_json(analysis_dir / "trajectory_feedback.json")
    return dspy.Example(
        id="current-skill",
        current_skill_md=read_live_skill(skill_path),
        trajectory_feedback=json.dumps(feedback, ensure_ascii=False, indent=2),
        constraints=proposal_constraints(),
    ).with_inputs("current_skill_md", "trajectory_feedback", "constraints")


def write_candidate_artifacts(
    *,
    call_dir: Path,
    initial_skill: str,
    raw_prediction: dspy.Prediction,
    validation_errors: list[str],
) -> tuple[Path, str, list[dict[str, Any]]]:
    candidate_dir = call_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    try:
        edits = parse_edits_json(str(raw_prediction.edits_json))
        validation_errors.extend(validate_edits(edits))
        candidate_skill = clean_candidate_skill(apply_skill_edits(initial_skill, edits))
    except Exception as exc:
        edits = []
        candidate_skill = initial_skill
        validation_errors.append(f"failed to apply patch edits: {exc}")

    candidate_path = candidate_dir / "SKILL.md"
    candidate_path.write_text(candidate_skill, encoding="utf-8")
    notes = parse_notes_json(str(raw_prediction.proposal_notes_json))
    save_json(candidate_dir / "edits.json", edits)
    save_json(
        call_dir / "proposal.json",
        {
            "candidate_skill_path": str(candidate_path),
            "validation_errors": validation_errors,
            "patch_edits": edits,
            "proposal_notes": notes,
            "raw_edits_json": str(raw_prediction.edits_json),
            "raw_proposal_notes_json": str(raw_prediction.proposal_notes_json),
        },
    )
    return candidate_path, candidate_skill, edits


def candidate_audio_paths(report: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for task in report.get("tasks", []):
        generated_audio = task.get("generated_audio") or {}
        audio_path = generated_audio.get("audio_path")
        if audio_path:
            paths.append(str(audio_path))
    return paths


class SkillOptGepaMetric:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        run_dir: Path,
        baseline_report: dict[str, Any],
    ):
        self.config = config
        self.run_dir = run_dir
        self.baseline_report = baseline_report
        self.call_index = 0

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any | None = None,
        pred_name: str | None = None,
        pred_trace: Any | None = None,
    ) -> ScoreWithFeedback:
        del trace, pred_name, pred_trace
        self.call_index += 1
        call_dir = self.run_dir / "metric_calls" / f"call_{self.call_index:03d}"
        call_dir.mkdir(parents=True, exist_ok=True)

        try:
            validation_errors: list[str] = []
            candidate_path, candidate_skill, _ = write_candidate_artifacts(
                call_dir=call_dir,
                initial_skill=str(gold.current_skill_md),
                raw_prediction=pred,
                validation_errors=validation_errors,
            )
            validation_errors.extend(validate_candidate_skill(candidate_skill))
            proposal = load_json(call_dir / "proposal.json")
            append_registry_record(
                self.config,
                build_candidate_record(
                    source="skillopt_gepa_metric",
                    run_dir=self.run_dir,
                    metric_call_dir=call_dir,
                    candidate_skill_path=candidate_path,
                    candidate_diff_path=None,
                    proposal=proposal,
                    analysis_dir="",
                ),
            )

            if validation_errors:
                feedback = "Candidate SKILL.md failed static validation: " + "; ".join(validation_errors)
                save_json(call_dir / "metric_result.json", {"score": 0.0, "feedback": feedback})
                return ScoreWithFeedback(score=0.0, feedback=feedback)

            candidate_report = run_rollout_set(self.config, call_dir, candidate_path, "candidate")
            save_json(call_dir / "candidate_report.json", candidate_report)
            decision = build_decision(
                candidate_dir=call_dir,
                candidate_skill_path=candidate_path,
                baseline_report=self.baseline_report,
                candidate_report=candidate_report,
            )
            save_json(call_dir / "decision.json", decision)
            append_registry_record(
                self.config,
                build_validation_record(
                    source="skillopt_gepa_metric",
                    run_dir=self.run_dir,
                    candidate_dir=call_dir,
                    proposal=proposal,
                    decision=decision,
                    baseline_report=self.baseline_report,
                    candidate_report=candidate_report,
                ),
            )

            baseline_passed = int(decision["baseline_passed"])
            candidate_passed = int(decision["candidate_passed"])
            if decision["accepted"]:
                score = 1.0
            elif decision.get("candidate_score") == decision.get("baseline_score") and candidate_passed > 0:
                score = float(self.config.get("tie_score", 0.5))
            else:
                score = 0.0

            feedback = "\n".join(
                [
                    f"Validation decision: {decision['reason']}",
                    (
                        f"Baseline passed {baseline_passed}/{decision['baseline_num_tasks']} "
                        f"with score {decision.get('baseline_score')}."
                    ),
                    (
                        f"Candidate passed {candidate_passed}/{decision['candidate_num_tasks']} "
                        f"with score {decision.get('candidate_score')}."
                    ),
                    f"Candidate audio paths: {candidate_audio_paths(candidate_report)}",
                    "Improve the skill only with small S2S-scoped edits grounded in trajectory evidence.",
                ]
            )
            save_json(
                call_dir / "metric_result.json",
                {"score": score, "feedback": feedback, "decision": decision},
            )
            return ScoreWithFeedback(score=score, feedback=feedback)
        except Exception as exc:
            feedback = f"SkillOpt GEPA metric failed during candidate validation: {exc}"
            save_json(call_dir / "metric_result.json", {"score": 0.0, "feedback": feedback})
            return ScoreWithFeedback(score=0.0, feedback=feedback)


def run_baseline_validation(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    baseline_skill_path = resolve_path(config["openclaw_live_skill_path"])
    baseline_dir = run_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("running fresh SkillOpt GEPA baseline rollout with %s", baseline_skill_path)
    report = run_rollout_set(config, baseline_dir, baseline_skill_path, "baseline")
    save_json(run_dir / "baseline_report.json", report)
    return report


def save_final_candidate(
    *,
    config: dict[str, Any],
    run_dir: Path,
    program: SkillCandidateProgram,
    example: dspy.Example,
) -> dict[str, Any]:
    with dspy.context(lm=build_proposer_lm(config)):
        prediction = program(**example.inputs())
    validation_errors: list[str] = []
    try:
        edits = parse_edits_json(str(prediction.edits_json))
        validation_errors.extend(validate_edits(edits))
        candidate_skill = clean_candidate_skill(apply_skill_edits(str(example.current_skill_md), edits))
    except Exception as exc:
        edits = []
        candidate_skill = str(example.current_skill_md)
        validation_errors.append(f"failed to apply patch edits: {exc}")
    validation_errors.extend(validate_candidate_skill(candidate_skill))
    final_dir = run_dir / "final_candidate"
    final_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = final_dir / "SKILL.md"
    candidate_path.write_text(candidate_skill, encoding="utf-8")
    save_json(final_dir / "edits.json", edits)
    result = {
        "candidate_skill_path": str(candidate_path),
        "validation_errors": validation_errors,
        "patch_edits": edits,
        "proposal_notes": parse_notes_json(str(prediction.proposal_notes_json)),
        "raw_edits_json": str(prediction.edits_json),
        "raw_proposal_notes_json": str(prediction.proposal_notes_json),
    }
    save_json(final_dir / "proposal.json", result)
    append_registry_record(
        config,
        build_candidate_record(
            source="skillopt_gepa_final",
            run_dir=run_dir,
            candidate_skill_path=candidate_path,
            candidate_diff_path=None,
            proposal=result,
            analysis_dir="",
        ),
    )
    return result


def collect_metric_stats(run_dir: Path) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for metric_path in sorted((run_dir / "metric_calls").glob("call_*/metric_result.json")):
        result = load_json(metric_path)
        decision_path = metric_path.parent / "decision.json"
        decision = load_json(decision_path) if decision_path.is_file() else {}
        calls.append(
            {
                "call_dir": str(metric_path.parent),
                "score": result.get("score"),
                "accepted": decision.get("accepted", False),
                "reason": decision.get("reason", result.get("feedback", "")),
                "baseline_passed": decision.get("baseline_passed"),
                "candidate_passed": decision.get("candidate_passed"),
                "baseline_score": decision.get("baseline_score"),
                "candidate_score": decision.get("candidate_score"),
                "baseline_clean_passed": decision.get("baseline_clean_passed"),
                "candidate_clean_passed": decision.get("candidate_clean_passed"),
            }
        )
    accepted = [row for row in calls if row.get("accepted")]
    best_score = max((float(row.get("score") or 0.0) for row in calls), default=0.0)
    return {
        "metric_call_count": len(calls),
        "accepted_count": len(accepted),
        "best_score": best_score,
        "calls": calls,
    }


def run_skillopt_gepa(config_path: str) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(config_path)
    run_dir = make_run_dir(config, "skillopt_gepa")
    analysis_dir = resolve_analysis_dir(config)
    example = build_skillopt_example(config, analysis_dir)

    baseline_report = run_baseline_validation(config, run_dir)
    metric = SkillOptGepaMetric(config=config, run_dir=run_dir, baseline_report=baseline_report)
    program = SkillCandidateProgram()

    LOGGER.info("compiling SkillOpt candidate proposer with DSPy GEPA")
    with dspy.context(lm=build_proposer_lm(config)):
        optimizer = dspy.GEPA(
            metric=metric,
            reflection_lm=build_proposer_lm(config),
            max_metric_calls=int(config.get("max_metric_calls", 2)),
            reflection_minibatch_size=int(config.get("reflection_minibatch_size", 1)),
            num_threads=int(config.get("num_threads", 1)),
            track_stats=True,
            seed=int(config.get("seed", 0)),
            skip_perfect_score=bool(config.get("skip_perfect_score", False)),
        )
        optimized_program = optimizer.compile(program, trainset=[example], valset=[example])

    final_candidate = save_final_candidate(
        config=config,
        run_dir=run_dir,
        program=optimized_program,
        example=example,
    )
    metric_stats = collect_metric_stats(run_dir)
    save_json(run_dir / "stats.json", metric_stats)
    summary = {
        "mode": "skillopt_gepa",
        "project_root": str(PROJECT_ROOT),
        "config": config,
        "analysis_dir": str(analysis_dir),
        "baseline_report": baseline_report,
        "final_candidate": final_candidate,
        "metric_stats": metric_stats,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"saved_run_dir={run_dir}")
    return run_dir

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Callable
import wave

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from .lm import CountingLM


class EvaluateGeneratedAudio(dspy.Signature):
    """Evaluate generated speech audio and return a score plus feedback."""

    expected_transcript: str = dspy.InputField()
    candidate_instruction: str = dspy.InputField()
    run_evidence: str = dspy.InputField()
    generated_audio: dspy.Audio = dspy.InputField()
    score: float = dspy.OutputField(desc="A score from 0.0 to 1.0.")
    feedback: str = dspy.OutputField(desc="Concise feedback for improving the MiMo S2S instruction.")


def wav_duration(path: str) -> float | None:
    if not path or not Path(path).is_file():
        return None
    try:
        with wave.open(path, "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except wave.Error:
        return None


def transcript_similarity(expected: str, actual: str) -> float:
    expected = "".join(str(expected or "").split())
    actual = "".join(str(actual or "").split())
    if not expected or not actual:
        return 0.0
    return difflib.SequenceMatcher(a=expected, b=actual).ratio()


def rule_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
) -> ScoreWithFeedback:
    del trace, pred_name, pred_trace
    if pred.error:
        return ScoreWithFeedback(score=0.0, feedback=f"MiMo S2S returned error: {pred.error}")

    output_exists = bool(pred.audio_path and Path(pred.audio_path).is_file())
    output_duration = float(pred.duration_sec or 0)
    ref_duration = wav_duration(gold.reference_audio_path)
    duration_ratio = 0.0
    if ref_duration and output_duration:
        duration_ratio = min(ref_duration, output_duration) / max(ref_duration, output_duration)

    text_score = transcript_similarity(gold.expected_transcript, pred.text_channel)
    instruction = str(pred.instruction)
    has_preservation = any(
        term in instruction.lower()
        for term in ["preserve", "保留", "speaker", "words", "language", "timing", "不要"]
    )

    score = 0.0
    score += 0.25 if output_exists else 0.0
    score += 0.20 if output_duration > 0.1 else 0.0
    score += 0.20 * duration_ratio
    score += 0.30 * text_score
    score += 0.05 if has_preservation else 0.0
    score = max(0.0, min(1.0, score))

    feedback = "\n".join(
        [
            f"Score={score:.3f}",
            f"Candidate instruction: {instruction}",
            f"- output_audio_exists={output_exists}: {pred.audio_path}",
            f"- output_duration={output_duration:.2f}s",
            f"- reference_duration={ref_duration:.2f}s" if ref_duration else "- reference_duration=missing",
            f"- duration_ratio={duration_ratio:.3f}",
            f"- transcript_similarity={text_score:.3f}",
            f"- expected_transcript={gold.expected_transcript}",
            f"- text_channel={pred.text_channel}",
            f"- mentions_preservation={has_preservation}",
        ]
    )
    return ScoreWithFeedback(score=score, feedback=feedback)


def build_evaluator_lm(config: dict[str, Any]) -> CountingLM:
    return CountingLM(
        config["gemma_model"],
        api_base=config["gemma_base_url"],
        api_key=config.get("gemma_api_key", "sk-local"),
        temperature=float(config.get("evaluator_temperature", 0.0)),
        max_tokens=int(config.get("evaluator_max_tokens", 800)),
        cache=False,
        counter_key="evaluator_lm_calls",
    )


def gemma_metric(config: dict[str, Any]) -> Callable[..., ScoreWithFeedback]:
    evaluator_lm = build_evaluator_lm(config)
    audio_evaluator = dspy.Predict(EvaluateGeneratedAudio)

    def evaluate(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: Any | None = None,
        pred_name: str | None = None,
        pred_trace: Any | None = None,
    ) -> ScoreWithFeedback:
        del trace, pred_name, pred_trace
        rule_score = rule_metric(gold, pred)
        if pred.error:
            return rule_score

        run_evidence = (
            f"Expected transcript: {gold.expected_transcript}\n"
            f"Reference audio path: {gold.reference_audio_path}\n"
            f"Output audio path: {pred.audio_path}\n"
            f"Output audio URL: {pred.audio_url}\n"
            f"Output duration seconds: {pred.duration_sec}\n"
            f"Text channel: {pred.text_channel}\n"
            f"Backend: {pred.backend}\n"
            f"Rule score and evidence:\n{rule_score.feedback}\n"
        )

        try:
            if not pred.audio_path or not Path(pred.audio_path).is_file():
                return ScoreWithFeedback(
                    score=rule_score.score,
                    feedback=f"{rule_score.feedback}\n- gemma_evaluator_skipped=missing_audio_file",
                )

            with dspy.context(lm=evaluator_lm):
                judgment = audio_evaluator(
                    expected_transcript=gold.expected_transcript,
                    candidate_instruction=pred.instruction,
                    run_evidence=run_evidence,
                    generated_audio=dspy.Audio.from_file(pred.audio_path),
                )

            score = max(0.0, min(1.0, float(judgment.score)))
            feedback = str(judgment.feedback).strip()
            if not feedback:
                feedback = "Gemma evaluator returned an empty feedback field."
        except Exception as exc:
            return ScoreWithFeedback(
                score=rule_score.score,
                feedback=f"{rule_score.feedback}\n- gemma_evaluator_error={exc}",
            )

        combined_feedback = "\n".join(
            [
                f"Gemma evaluator score={score:.3f}",
                feedback,
                "",
                "Rule-based evidence:",
                rule_score.feedback,
            ]
        )
        return ScoreWithFeedback(score=score, feedback=combined_feedback)

    return evaluate


def build_metric(config: dict[str, Any]) -> Callable[..., ScoreWithFeedback]:
    evaluator = str(config.get("evaluator", "rule")).strip().lower()
    if evaluator == "gemma":
        return gemma_metric(config)
    if evaluator == "rule":
        return rule_metric
    raise ValueError(f"Unknown evaluator: {evaluator}")

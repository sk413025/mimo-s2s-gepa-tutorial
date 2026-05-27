from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
        if pred.error:
            return ScoreWithFeedback(score=0.0, feedback=f"MiMo S2S returned error: {pred.error}")
        if not pred.audio_path or not Path(pred.audio_path).is_file():
            return ScoreWithFeedback(score=0.0, feedback=f"Generated audio file is missing: {pred.audio_path}")

        run_evidence = (
            f"Expected transcript: {gold.expected_transcript}\n"
            f"Reference audio path: {gold.reference_audio_path}\n"
            f"Output audio path: {pred.audio_path}\n"
            f"Output audio URL: {pred.audio_url}\n"
            f"Output duration seconds: {pred.duration_sec}\n"
            f"Text channel: {pred.text_channel}\n"
            f"Backend: {pred.backend}\n"
        )

        try:
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
            return ScoreWithFeedback(score=0.0, feedback=f"Gemma evaluator error: {exc}")

        return ScoreWithFeedback(score=score, feedback=f"Gemma evaluator score={score:.3f}\n{feedback}")

    return evaluate


def build_metric(config: dict[str, Any]) -> Callable[..., ScoreWithFeedback]:
    return gemma_metric(config)

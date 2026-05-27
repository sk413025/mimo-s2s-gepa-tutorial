from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any
import wave

import dspy
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback


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


def metric(
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


from __future__ import annotations

import json
from pathlib import Path

import dspy


def load_examples(path: str | Path) -> list[dspy.Example]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    examples: list[dspy.Example] = []
    for record in records:
        sample_id = record["id"]
        transcript = record.get("transcript", "")
        speaker = record.get("speaker", "")
        material = record.get("material", "")
        examples.append(
            dspy.Example(
                id=sample_id,
                restoration_goal=(
                    "Convert calibrated LDV/vibration-domain speech into clean microphone-like Mandarin speech. "
                    "Preserve the target words, speaker identity, language, and approximate timing."
                ),
                input_note=(
                    f"sample_id={sample_id}; speaker={speaker}; material={material}; "
                    f"target_transcript={transcript}"
                ),
                audio_path=record["input_audio"],
                prompt_examples_json=record["prompt_examples_json"],
                reference_audio_path=record["reference_audio"],
                expected_transcript=transcript,
                speaker=speaker,
                material=material,
            ).with_inputs(
                "restoration_goal",
                "input_note",
                "audio_path",
                "prompt_examples_json",
            )
        )
    return examples


def split_examples(
    examples: list[dspy.Example],
    train_size: int,
    val_size: int,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    train = examples[: max(1, train_size)]
    val = examples[: max(1, val_size)]
    return train, val


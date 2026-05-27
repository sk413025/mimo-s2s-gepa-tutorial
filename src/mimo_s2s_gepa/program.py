from __future__ import annotations

from typing import Any

import dspy

from .s2s_client import call_mimo_s2s


class WriteInstruction(dspy.Signature):
    """Write one MiMo S2S instruction for restoring distorted LDV speech."""

    skill_context: str = dspy.InputField(
        desc="Relevant OpenClaw SKILL.md guidance. Use it as policy context when present."
    )
    restoration_goal: str = dspy.InputField()
    input_note: str = dspy.InputField()
    instruction: str = dspy.OutputField(
        desc="One concise instruction. Preserve words, speaker identity, language, and approximate timing."
    )


class MiMoS2SProgram(dspy.Module):
    """DSPy program: write an instruction, then evaluate it through MiMo S2S."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.instruction_writer = dspy.Predict(WriteInstruction)

    def forward(
        self,
        restoration_goal: str,
        input_note: str,
        audio_path: str,
        prompt_examples_json: str,
        skill_context: str = "",
    ) -> dspy.Prediction:
        written = self.instruction_writer(
            skill_context=skill_context or "No external skill guidance was provided.",
            restoration_goal=restoration_goal,
            input_note=input_note,
        )
        instruction = " ".join(str(written.instruction).split())

        result = call_mimo_s2s(
            model=self.config["mimo_model"],
            api_base=self.config["mimo_base_url"],
            api_key=self.config.get("mimo_api_key", "sk-local"),
            instruction=instruction,
            audio_path=audio_path,
            prompt_examples_json=prompt_examples_json,
            decoding={
                "max_new_tokens": int(self.config.get("max_new_tokens", 32)),
                "min_new_tokens": int(self.config.get("min_new_tokens", 0)),
                "repetition_penalty": float(self.config.get("repetition_penalty", 1.35)),
                "decode_preset": self.config.get("decode_preset", "deterministic"),
            },
            timeout_sec=float(self.config.get("timeout_sec", 240)),
        )
        return dspy.Prediction(
            instruction=instruction,
            audio_url=result.get("audio_url", ""),
            audio_path=result.get("audio_path", ""),
            text_channel=result.get("text_channel", ""),
            duration_sec=float(result.get("duration_sec") or 0),
            backend=result.get("backend", ""),
            error=result.get("error", ""),
        )

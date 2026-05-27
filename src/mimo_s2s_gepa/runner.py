from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import dspy

from .config import PROJECT_ROOT, load_config
from .counters import reset_counters, snapshot_counters
from .data import load_examples, split_examples
from .lm import CountingLM
from .metrics import build_metric
from .program import MiMoS2SProgram


def configure_task_lm(config: dict[str, Any]) -> None:
    task_lm = CountingLM(
        config["gemma_model"],
        api_base=config["gemma_base_url"],
        api_key=config.get("gemma_api_key", "sk-local"),
        temperature=float(config.get("gemma_temperature", 0.2)),
        max_tokens=int(config.get("gemma_max_tokens", 800)),
        cache=False,
        counter_key="task_lm_calls",
    )
    dspy.settings.configure(lm=task_lm)


def build_reflection_lm(config: dict[str, Any]) -> CountingLM:
    return CountingLM(
        config["gemma_model"],
        api_base=config["gemma_base_url"],
        api_key=config.get("gemma_api_key", "sk-local"),
        temperature=float(config.get("reflection_temperature", 1.0)),
        max_tokens=int(config.get("reflection_max_tokens", 2400)),
        cache=False,
        counter_key="reflection_lm_calls",
    )


def make_run_dir(config: dict[str, Any], mode: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["output_dir"]) / f"{stamp}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prediction_to_dict(
    example: dspy.Example,
    prediction: dspy.Prediction,
    metric_fn,
) -> dict[str, Any]:
    scored = metric_fn(example, prediction)
    return {
        "id": example.id,
        "instruction": prediction.instruction,
        "audio_url": prediction.audio_url,
        "audio_path": prediction.audio_path,
        "duration_sec": prediction.duration_sec,
        "text_channel": prediction.text_channel,
        "backend": prediction.backend,
        "error": prediction.error,
        "score": scored.score,
        "feedback": scored.feedback,
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(config_path: str, mode: str) -> Path:
    config = load_config(config_path)
    examples = load_examples(config["data_path"])
    trainset, valset = split_examples(
        examples,
        train_size=int(config.get("train_size", 1)),
        val_size=int(config.get("val_size", 1)),
    )

    reset_counters()
    run_dir = make_run_dir(config, mode)

    if mode != "smoke":
        configure_task_lm(config)

    fixed_instruction = config.get("fixed_instruction", "") if mode == "smoke" else ""
    program = MiMoS2SProgram(config, fixed_instruction=fixed_instruction)
    metric_fn = build_metric(config)

    if mode == "gepa":
        optimizer = dspy.GEPA(
            metric=metric_fn,
            reflection_lm=build_reflection_lm(config),
            max_metric_calls=int(config.get("max_metric_calls", 4)),
            reflection_minibatch_size=int(config.get("reflection_minibatch_size", 1)),
            num_threads=int(config.get("num_threads", 1)),
            track_stats=True,
            seed=int(config.get("seed", 0)),
        )
        program = optimizer.compile(program, trainset=trainset, valset=valset)

    outputs = []
    for example in valset:
        prediction = program(**example.inputs())
        outputs.append(prediction_to_dict(example, prediction, metric_fn))
        print(json.dumps(outputs[-1], ensure_ascii=False, indent=2))

    summary = {
        "mode": mode,
        "project_root": str(PROJECT_ROOT),
        "config": config,
        "train_size": len(trainset),
        "val_size": len(valset),
        "counters": snapshot_counters(),
        "outputs": outputs,
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "predictions.json", outputs)
    print(f"saved_run_dir={run_dir}")
    return run_dir

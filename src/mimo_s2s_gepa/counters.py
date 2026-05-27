COUNTERS = {
    "task_lm_calls": 0,
    "reflection_lm_calls": 0,
    "mimo_s2s_calls": 0,
}


def reset_counters() -> None:
    for key in COUNTERS:
        COUNTERS[key] = 0


def snapshot_counters() -> dict[str, int]:
    return dict(COUNTERS)


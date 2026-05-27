from __future__ import annotations

import dspy

from .counters import COUNTERS


class CountingLM(dspy.LM):
    """DSPy LM that counts logical calls made by this tutorial."""

    def __init__(self, *args, counter_key: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter_key = counter_key

    def __call__(self, *args, **kwargs):
        COUNTERS[self.counter_key] += 1
        return super().__call__(*args, **kwargs)


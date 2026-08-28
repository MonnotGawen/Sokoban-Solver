"""Select predictor implementation from environment / explicit choice."""

from __future__ import annotations

import os
from typing import Literal

from track1.llm.openai_predictor import OpenAIPredictor, PromptStyle
from track1.llm.predictor import MockPredictor, OneStepPredictor

PredictorKind = Literal["auto", "openai", "mock"]


def make_predictor(
    kind: PredictorKind = "auto",
    *,
    mock_seed: int | None = 0,
    prompt_style: PromptStyle = "plain",
    samples: int = 1,
    enable_cache: bool = True,
) -> OneStepPredictor:
    """Return an OpenAI predictor when ``OPENAI_API_KEY`` is available, else the mock."""
    if kind == "mock":
        return MockPredictor(seed=mock_seed)
    use_openai = kind == "openai" or (
        kind == "auto" and os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if use_openai:
        return OpenAIPredictor(
            prompt_style=prompt_style,
            samples=samples,
            enable_cache=enable_cache,
        )
    return MockPredictor(seed=mock_seed)

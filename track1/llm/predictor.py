from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from track1.sokoban.model import Action, Level, State


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    action: Action
    score: float
    reasoning: str | None = None


class OneStepPredictor(ABC):
    """Ranks the legal actions at a single state (higher score = more preferred)."""

    @abstractmethod
    async def predict_ranked(
        self,
        level: Level,
        state: State,
        legal_actions: Sequence[Action],
        ascii_board: str,
    ) -> list[ActionPrediction]: ...


class MockPredictor(OneStepPredictor):
    """Offline control predictor: returns a uniform random permutation of the legal moves."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def predict_ranked(
        self,
        level: Level,
        state: State,
        legal_actions: Sequence[Action],
        ascii_board: str,
    ) -> list[ActionPrediction]:
        acts = list(legal_actions)
        self._rng.shuffle(acts)
        return [
            ActionPrediction(action=a, score=float(len(acts) - i))
            for i, a in enumerate(acts)
        ]

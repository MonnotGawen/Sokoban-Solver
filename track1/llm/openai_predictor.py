"""One-step action predictor backed by an OpenAI-compatible chat API.

Each call sends a single board state and asks the model to rank its legal actions,
so we stay within the take-home's "one-step predictor" constraint.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from typing import Any, Literal, Sequence

from openai import AsyncOpenAI

from track1.llm.predictor import ActionPrediction, OneStepPredictor
from track1.sokoban.model import Action, Level, State

PromptStyle = Literal["plain", "reason_act"]

_SYSTEM_PROMPT = (
    "You are a Sokoban one-step planner. "
    "Output strictly one JSON object and nothing else."
)


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _parse_ranking(content: str, legal: set[Action]) -> list[Action]:
    """Parse a ``{"ranking": ["up","left",...]}`` object into a de-duplicated list of legal actions."""
    obj = _extract_json_object(content)
    if not obj:
        return []
    raw = obj.get("ranking")
    if not isinstance(raw, list):
        return []
    seen: set[Action] = set()
    out: list[Action] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        try:
            a = Action(item.strip().lower())
        except ValueError:
            continue
        if a in legal and a not in seen:
            out.append(a)
            seen.add(a)
    return out


def _borda_count(contents: list[str], legal: set[Action]) -> list[Action]:
    """Aggregate several rankings by Borda count (best gets len(r) points, next len(r)-1, ...)."""
    scores: Counter[Action] = Counter()
    for c in contents:
        for i, a in enumerate(_parse_ranking(c, legal)):
            scores[a] += len(legal) - i
    return [a for a, _ in scores.most_common()]


class OpenAIPredictor(OneStepPredictor):
    """Rank the legal actions at one state via a chat completion that returns a JSON ranking.

    ``samples > 1`` triggers self-consistency (Borda count at T=0.5); caching is
    only active at ``samples=1``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        prompt_style: PromptStyle = "plain",
        samples: int = 1,
        enable_cache: bool = True,
        timeout: float = 30.0,
        max_tokens: int = 128,
    ) -> None:
        key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIPredictor")

        self._model = (model or os.environ.get("OPENAI_MODEL") or "llama-3.1-8b-instruct").strip()
        self._timeout = timeout
        self._max_tokens = max_tokens

        client_kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout, "max_retries": 2}
        base = (base_url or os.environ.get("OPENAI_BASE_URL") or "").strip()
        if base:
            client_kwargs["base_url"] = base.rstrip("/")
        self._client = AsyncOpenAI(**client_kwargs)

        self._prompt_style: PromptStyle = prompt_style
        self._samples = max(1, int(samples))
        self._cache_enabled = enable_cache and self._samples == 1
        self._cache: dict[tuple, tuple[Action, ...]] = {}
        self.cache_hits = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompt_style(self) -> PromptStyle:
        return self._prompt_style

    @property
    def samples(self) -> int:
        return self._samples

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.close()

    def _build_user_prompt(
        self,
        level: Level,
        state: State,
        legal_words: str,
        ascii_board: str,
    ) -> str:
        legend = (
            "Symbols: # wall, @ player, $ box, . goal, * box on goal, + player on goal, space floor."
        )
        header = (
            f"{legend}\n\nCurrent state:\n{ascii_board}\n\n"
            f"Grid: {level.height} rows x {level.width} cols | boxes={len(state.boxes)} | goals={len(level.goals)}.\n"
            f"Legal moves (choose from): {legal_words}.\n\n"
            "Rank EVERY legal move from best to worst under these priorities:\n"
            "1) Progress toward solving (boxes on goals),\n"
            "2) Avoid immediate undo/backtracking when another legal move is comparable,\n"
            "3) Avoid pushes that likely deadlock boxes (non-goal corners/walls)."
        )
        if self._prompt_style == "reason_act":
            return (
                header
                + "\n\n"
                + 'Return exactly one JSON object: '
                '{"reason": "<=25 words", "ranking": ["up","left","down","right"]}.\n'
                "The ranking must list each legal move exactly once, "
                "first = most preferred. No markdown, no prose outside JSON."
            )
        return (
            header
            + "\n\n"
            + 'Return exactly one JSON object: '
            '{"ranking": ["up","left","down","right"]}.\n'
            "The ranking must list each legal move exactly once, "
            "first = most preferred. No markdown, no prose."
        )

    async def _one_completion(self, user: str, temperature: float) -> str:
        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        return completion.choices[0].message.content or ""

    async def predict_ranked(
        self,
        level: Level,
        state: State,
        legal_actions: Sequence[Action],
        ascii_board: str,
    ) -> list[ActionPrediction]:
        legal = list(legal_actions)
        if not legal:
            return []
        legal_set = set(legal)
        legal_words = ", ".join(a.value for a in legal)

        cache_key = (
            state.key(),
            tuple(a.value for a in legal),
            self._prompt_style,
            ascii_board,
        )
        if self._cache_enabled and cache_key in self._cache:
            self.cache_hits += 1
            return self._ranking_to_predictions(list(self._cache[cache_key]), legal)

        user = self._build_user_prompt(level, state, legal_words, ascii_board)

        try:
            if self._samples == 1:
                content = await self._one_completion(user, temperature=0.0)
                ranking = _parse_ranking(content, legal_set)
            else:
                contents = await asyncio.gather(
                    *[self._one_completion(user, temperature=0.5) for _ in range(self._samples)]
                )
                ranking = _borda_count(contents, legal_set)
        except Exception as e:
            return self._ranking_to_predictions([], legal, reasoning=f"error:{type(e).__name__}")

        for a in legal:
            if a not in ranking:
                ranking.append(a)

        if self._cache_enabled:
            self._cache[cache_key] = tuple(ranking)

        return self._ranking_to_predictions(ranking, legal)

    @staticmethod
    def _ranking_to_predictions(
        ranking: list[Action],
        legal: list[Action],
        *,
        reasoning: str | None = None,
    ) -> list[ActionPrediction]:
        ordered = list(ranking) if ranking else list(legal)
        for a in legal:
            if a not in ordered:
                ordered.append(a)
        n = len(ordered)
        return [
            ActionPrediction(action=a, score=float(n - i), reasoning=reasoning)
            for i, a in enumerate(ordered)
        ]

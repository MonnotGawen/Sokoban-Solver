"""Beam search over states, with the LLM ranking legal actions at each expanded node.

One ``predict_ranked`` call per node returns an ordering over every legal action,
so the per-layer cost is ``beam_width`` calls (independent of the branching
factor). Children are deduplicated by state key within a layer, keeping the best
cumulative score; ``visited`` is updated only after top-B selection so that
dropped candidates don't block future exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Hashable

from track1.sokoban.model import Action, Level, State
from track1.sokoban.simulator import Simulator

if TYPE_CHECKING:
    from track1.llm.predictor import OneStepPredictor


@dataclass(frozen=True, slots=True)
class BeamResult:
    success: bool
    actions: tuple[Action, ...]
    nodes_expanded: int
    llm_calls: int
    message: str = ""


async def beam_search_llm_guided_async(
    level: Level,
    predictor: "OneStepPredictor",
    board_fn: Callable[[Level, State], str],
    *,
    beam_width: int = 3,
    max_depth: int = 30,
    max_llm_calls: int = 200,
) -> BeamResult:
    sim = Simulator(level)
    start = sim.initial_state()
    if sim.is_solved(start):
        return BeamResult(True, (), 0, 0, "already solved")

    # Beam entries: (state, actions_so_far, cumulative_score).
    beam: list[tuple[State, tuple[Action, ...], float]] = [(start, (), 0.0)]
    visited: set[Hashable] = {start.key()}
    llm_calls = 0
    nodes_expanded = 0

    for _ in range(max_depth):
        # Per-layer candidate table keyed on child state; keep best cumulative score.
        candidates: dict[Hashable, tuple[State, tuple[Action, ...], float]] = {}

        for state, path, cum in beam:
            if llm_calls >= max_llm_calls:
                return BeamResult(False, (), nodes_expanded, llm_calls, "max_llm_calls")

            legal = sim.legal_actions(state)
            if not legal:
                continue

            board_text = board_fn(level, state)
            preds = await predictor.predict_ranked(level, state, legal, board_text)
            llm_calls += 1
            nodes_expanded += 1
            if not preds:
                continue

            for p in preds:
                if p.action not in legal:
                    continue
                res = sim.try_step(state, p.action)
                if res is None:
                    continue
                child = res.state
                child_key = child.key()
                if child_key in visited:
                    continue
                new_path = path + (p.action,)
                new_cum = cum + float(p.score)
                if sim.is_solved(child):
                    return BeamResult(True, new_path, nodes_expanded, llm_calls, "")
                prev = candidates.get(child_key)
                if prev is None or prev[2] < new_cum:
                    candidates[child_key] = (child, new_path, new_cum)

        if not candidates:
            return BeamResult(False, (), nodes_expanded, llm_calls, "frontier exhausted")

        # Keep top-B by cumulative LLM score and *only then* mark visited.
        ranked = sorted(candidates.values(), key=lambda x: x[2], reverse=True)
        beam = ranked[:beam_width]
        for child, _, _ in beam:
            visited.add(child.key())

    return BeamResult(False, (), nodes_expanded, llm_calls, "max_depth")

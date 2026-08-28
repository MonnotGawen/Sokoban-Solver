"""LLM-only rollout: greedy one-step chain (spec-compliant one-step predictor)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Hashable

from track1.sokoban.model import Action, Level, State
from track1.sokoban.simulator import Simulator

if TYPE_CHECKING:
    from track1.llm.predictor import OneStepPredictor


@dataclass(frozen=True, slots=True)
class RolloutResult:
    success: bool
    actions: tuple[Action, ...]
    llm_calls: int
    steps_executed: int
    message: str = ""


def _reverse_action(a: Action) -> Action:
    if a == Action.UP:
        return Action.DOWN
    if a == Action.DOWN:
        return Action.UP
    if a == Action.LEFT:
        return Action.RIGHT
    return Action.LEFT


async def greedy_single_action_rollout_async(
    level: Level,
    predictor: OneStepPredictor,
    board_fn: Callable[[Level, State], str],
    *,
    max_steps: int = 200,
    max_llm_calls: int = 500,
) -> RolloutResult:
    """
    Repeatedly ask the one-step predictor for its top move and apply it (greedy walk).

    Stops on solve, illegal move, cycle, or caps. Each step = one ``predict_ranked`` call.
    """
    sim = Simulator(level)
    state = sim.initial_state()
    if sim.is_solved(state):
        return RolloutResult(True, (), 0, 0, "already solved")

    visited: set[Hashable] = {state.key()}
    actions: list[Action] = []
    llm_calls = 0

    for _ in range(max_steps):
        if sim.is_solved(state):
            return RolloutResult(True, tuple(actions), llm_calls, len(actions), "")
        legal = sim.legal_actions(state)
        if not legal:
            return RolloutResult(False, tuple(actions), llm_calls, len(actions), "no legal moves")
        if llm_calls >= max_llm_calls:
            return RolloutResult(
                sim.is_solved(state),
                tuple(actions),
                llm_calls,
                len(actions),
                "max_llm_calls",
            )

        board_text = board_fn(level, state)
        recent = ",".join(a.value for a in actions[-6:]) if actions else "none"
        avoid_reverse = _reverse_action(actions[-1]).value if actions else "none"
        # Context hint for the LLM predictor: keep short-term memory to reduce oscillations.
        board_with_context = (
            f"{board_text}\n\n"
            "Planner context:\n"
            f"- step_index={len(actions)}\n"
            f"- recent_actions={recent}\n"
            f"- avoid_immediate_reverse={avoid_reverse}\n"
        )
        preds = await predictor.predict_ranked(level, state, legal, board_with_context)
        llm_calls += 1
        if not preds:
            return RolloutResult(False, tuple(actions), llm_calls, len(actions), "empty prediction")
        a = preds[0].action
        if a not in legal:
            return RolloutResult(False, tuple(actions), llm_calls, len(actions), "illegal top action")
        res = sim.try_step(state, a)
        if res is None:
            return RolloutResult(False, tuple(actions), llm_calls, len(actions), "simulator rejected")
        state = res.state
        actions.append(a)
        k = state.key()
        if k in visited:
            return RolloutResult(False, tuple(actions), llm_calls, len(actions), "state cycle")
        visited.add(k)

    return RolloutResult(sim.is_solved(state), tuple(actions), llm_calls, len(actions), "max_steps")

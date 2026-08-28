from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable

from track1.sokoban.model import Action, Level, State
from track1.sokoban.simulator import Simulator


@dataclass(frozen=True, slots=True)
class SearchResult:
    success: bool
    actions: tuple[Action, ...]
    nodes_expanded: int
    message: str = ""
    llm_calls: int = 0


def breadth_first_search(level: Level, max_nodes: int = 500_000) -> SearchResult:
    """Exact baseline (no LLM)."""
    sim = Simulator(level)
    start = sim.initial_state()
    if sim.is_solved(start):
        return SearchResult(True, (), 1, llm_calls=0)

    start_key = start.key()
    queue: deque[tuple[State, tuple[Action, ...]]] = deque([(start, ())])
    visited: set[Hashable] = {start_key}
    expanded = 0

    while queue and expanded < max_nodes:
        state, path = queue.popleft()
        expanded += 1
        for a in sim.legal_actions(state):
            ns = sim.step(state, a)
            k = ns.key()
            if k in visited:
                continue
            visited.add(k)
            npath = path + (a,)
            if sim.is_solved(ns):
                return SearchResult(True, npath, expanded, llm_calls=0)
            queue.append((ns, npath))

    return SearchResult(False, (), expanded, message="BFS exhausted or node cap", llm_calls=0)

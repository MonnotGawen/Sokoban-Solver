from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from track1.sokoban.model import ACTION_DELTA, Action, Level, State


@dataclass(frozen=True, slots=True)
class TransitionResult:
    state: State
    moved_box: bool


def _cell_char(level: Level, state: State, r: int, c: int) -> str:
    coord = (r, c)
    if coord in level.walls:
        return "#"
    is_goal = coord in level.goals
    is_box = coord in state.boxes
    is_player = coord == state.player
    if is_player:
        return "+" if is_goal else "@"
    if is_box:
        return "*" if is_goal else "$"
    return "." if is_goal else " "


def render_ascii(level: Level, state: State) -> str:
    lines: list[str] = []
    for r in range(level.height):
        row_chars: list[str] = []
        for c in range(level.width):
            row_chars.append(_cell_char(level, state, r, c))
        lines.append("".join(row_chars))
    return "\n".join(lines)


class Simulator:
    """Legal moves + terminal check."""

    def __init__(self, level: Level) -> None:
        self.level = level

    def initial_state(self) -> State:
        return State(player=self.level.initial_player, boxes=self.level.initial_boxes)

    def is_solved(self, state: State) -> bool:
        return state.is_solved(self.level.goals)

    def legal_actions(self, state: State) -> list[Action]:
        out: list[Action] = []
        for a in Action:
            if self.try_step(state, a) is not None:
                out.append(a)
        return out

    def try_step(self, state: State, action: Action) -> TransitionResult | None:
        dr, dc = ACTION_DELTA[action]
        pr, pc = state.player
        tr, tc = pr + dr, pc + dc
        if not self.level.in_bounds(tr, tc):
            return None
        if (tr, tc) in self.level.walls:
            return None

        boxes = state.boxes
        if (tr, tc) in boxes:
            br, bc = tr + dr, tc + dc
            if not self.level.in_bounds(br, bc):
                return None
            if (br, bc) in self.level.walls or (br, bc) in boxes:
                return None
            new_boxes = frozenset((set(boxes) - {(tr, tc)}) | {(br, bc)})
            return TransitionResult(state=State(player=(tr, tc), boxes=new_boxes), moved_box=True)

        return TransitionResult(state=State(player=(tr, tc), boxes=boxes), moved_box=False)

    def step(self, state: State, action: Action) -> State:
        res = self.try_step(state, action)
        if res is None:
            raise ValueError(f"Illegal action {action} from state")
        return res.state

    def apply_path(self, state: State, actions: Iterable[Action]) -> State:
        s = state
        for a in actions:
            s = self.step(s, a)
        return s

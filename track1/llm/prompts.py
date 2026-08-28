"""State encodings for LLM prompts (ablations / experiments)."""

from __future__ import annotations

from typing import Literal

from track1.sokoban.model import Coord, Level, State
from track1.sokoban.simulator import render_ascii

StateEncoding = Literal["ascii", "labeled_rows", "tuple_fields", "ascii_features"]


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _is_corner_deadlock(level: Level, box: Coord) -> bool:
    """Box is in a non-goal cell and blocked by two perpendicular walls."""
    if box in level.goals:
        return False
    r, c = box
    up = (r - 1, c) in level.walls or not level.in_bounds(r - 1, c)
    down = (r + 1, c) in level.walls or not level.in_bounds(r + 1, c)
    left = (r, c - 1) in level.walls or not level.in_bounds(r, c - 1)
    right = (r, c + 1) in level.walls or not level.in_bounds(r, c + 1)
    return (up or down) and (left or right)


def _features_block(level: Level, state: State) -> str:
    goals = list(level.goals)
    lines: list[str] = []
    solved_boxes = 0
    for box in sorted(state.boxes):
        on_goal = box in level.goals
        if on_goal:
            solved_boxes += 1
        if goals:
            nearest = min(goals, key=lambda g: _manhattan(box, g))
            dist = _manhattan(box, nearest)
        else:
            nearest = (-1, -1)
            dist = -1
        corner = _is_corner_deadlock(level, box)
        tag = "on_goal" if on_goal else ("corner_deadlock" if corner else "free")
        lines.append(
            f"- box{tuple(box)} -> nearest_goal{tuple(nearest)} dist={dist} status={tag}"
        )
    goals_left = max(len(level.goals) - solved_boxes, 0)
    header = (
        f"progress: boxes_on_goals={solved_boxes}/{len(level.goals)} "
        f"remaining_goals={goals_left}"
    )
    return header + "\n" + "\n".join(lines) if lines else header


def encode_board_for_llm(level: Level, state: State, style: StateEncoding) -> str:
    """Render the current (level, state) in one of four text formats used in the ablations."""
    if style == "ascii":
        return render_ascii(level, state)
    if style == "labeled_rows":
        lines = render_ascii(level, state).splitlines()
        return "\n".join(f"row{r:02d}: {ln}" for r, ln in enumerate(lines))
    if style == "tuple_fields":
        goals = sorted(level.goals)
        boxes = sorted(state.boxes)
        pr, pc = state.player
        return (
            f"grid_height={level.height} grid_width={level.width}\n"
            f"player=(row={pr}, col={pc})\n"
            f"goals={[list(g) for g in goals]}\n"
            f"boxes={[list(b) for b in boxes]}\n"
            f"num_walls={len(level.walls)}\n"
        )
    if style == "ascii_features":
        board = render_ascii(level, state)
        feats = _features_block(level, state)
        return f"{board}\n\nderived_features:\n{feats}"
    raise ValueError(f"Unknown encoding: {style}")

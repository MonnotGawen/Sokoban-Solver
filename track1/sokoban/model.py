from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

Coord = Tuple[int, int]


class Action(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


ACTION_DELTA: dict[Action, Coord] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True, slots=True)
class Level:
    """Static puzzle definition: walls, initial boxes, player, goals."""

    height: int
    width: int
    walls: frozenset[Coord]
    goals: frozenset[Coord]
    initial_boxes: frozenset[Coord]
    initial_player: Coord
    title: str | None = None

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width


@dataclass(frozen=True, slots=True)
class State:
    """Dynamic state: player + box positions (goals and walls come from Level)."""

    player: Coord
    boxes: frozenset[Coord]

    def key(self) -> tuple[Coord, frozenset[Coord]]:
        return (self.player, self.boxes)

    def is_solved(self, goals: frozenset[Coord]) -> bool:
        return goals <= self.boxes and self.boxes <= goals

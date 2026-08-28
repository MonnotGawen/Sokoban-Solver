from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from track1.sokoban.model import Level, State
from track1.sokoban.simulator import Simulator


@dataclass
class GameSession:
    session_id: str
    level: Level
    level_index: int
    state: State
    move_count: int = 0
    undo_stack: list[tuple[State, int]] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, GameSession] = {}

    def create(self, level: Level, level_index: int) -> GameSession:
        sid = str(uuid.uuid4())
        sim = Simulator(level)
        sess = GameSession(
            session_id=sid,
            level=level,
            level_index=level_index,
            state=sim.initial_state(),
        )
        self._sessions[sid] = sess
        return sess

    def get(self, session_id: str) -> GameSession | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from track1.llm.prompts import StateEncoding
from track1.sokoban.model import Action


class LevelSummary(BaseModel):
    index: int
    title: str | None = None
    height: int
    width: int
    num_boxes: int


class BoardResponse(BaseModel):
    rows: list[str]
    solved: bool
    legal_actions: list[Action]


class SessionCreate(BaseModel):
    level_index: int = Field(ge=0)


class SessionInfo(BaseModel):
    session_id: str
    level_index: int
    board: BoardResponse
    move_count: int
    can_undo: bool = False


class MoveRequest(BaseModel):
    action: Action


class MoveResponse(BaseModel):
    board: BoardResponse
    move_count: int
    illegal: bool = False
    can_undo: bool = False


class SolveRequest(BaseModel):
    method: Literal["bfs", "llm_greedy", "beam"] = "bfs"
    max_nodes: int = Field(default=500_000, ge=100, le=5_000_000)
    predictor: Literal["auto", "openai", "mock"] = "auto"
    state_encoding: StateEncoding = "ascii"
    greedy_max_steps: int = Field(default=200, ge=5, le=5000)
    greedy_max_llm_calls: int = Field(default=500, ge=5, le=20_000)
    beam_width: int = Field(default=3, ge=1, le=32)
    beam_max_depth: int = Field(default=30, ge=2, le=500)
    beam_max_llm_calls: int = Field(default=150, ge=5, le=20_000)


class SolveResponse(BaseModel):
    success: bool
    actions: list[Action]
    nodes_expanded: int
    message: str = ""
    llm_calls: int = 0
    state_encoding: str = "ascii"
    search_mode: str = ""


class ApplyPlanRequest(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    stop_on_illegal: bool = True


class ApplyPlanResponse(BaseModel):
    board: BoardResponse
    move_count: int
    applied_steps: int
    completed: bool
    illegal_at_step: int | None = None
    can_undo: bool = False


class LlmPreviewResponse(BaseModel):
    encoding: str
    text: str

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from track1.api.levels_loader import load_levels
from track1.api.schemas import (
    ApplyPlanRequest,
    ApplyPlanResponse,
    BoardResponse,
    LevelSummary,
    LlmPreviewResponse,
    MoveRequest,
    MoveResponse,
    SessionCreate,
    SessionInfo,
    SolveRequest,
    SolveResponse,
)
from track1.api.store import GameSession, SessionStore
from track1.config import load_app_env, openai_configured
from track1.llm.beam import beam_search_llm_guided_async
from track1.llm.factory import make_predictor
from track1.llm.openai_predictor import OpenAIPredictor
from track1.llm.prompts import StateEncoding, encode_board_for_llm
from track1.llm.rollout import greedy_single_action_rollout_async
from track1.sokoban.search import breadth_first_search
from track1.sokoban.simulator import Simulator, render_ascii

LEVELS: list = []
STORE = SessionStore()


def _board_payload(level, state) -> BoardResponse:
    sim = Simulator(level)
    rows = render_ascii(level, state).splitlines()
    return BoardResponse(
        rows=rows,
        solved=sim.is_solved(state),
        legal_actions=sim.legal_actions(state),
    )


def _session_info(sess: GameSession) -> SessionInfo:
    return SessionInfo(
        session_id=sess.session_id,
        level_index=sess.level_index,
        board=_board_payload(sess.level, sess.state),
        move_count=sess.move_count,
        can_undo=bool(sess.undo_stack),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global LEVELS
    load_app_env()
    env_path = os.environ.get("TRACK1_LEVELS_PATH")
    LEVELS = load_levels(Path(env_path) if env_path else None)
    yield


app = FastAPI(title="Track1 Sokoban API", version="0.1.0", lifespan=lifespan)

_cors = os.environ.get("TRACK1_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "levels_loaded": len(LEVELS),
        "openai_configured": openai_configured(),
        "openai_model": os.environ.get("OPENAI_MODEL", "llama-3.1-8b-instruct") or "llama-3.1-8b-instruct",
    }


@app.get("/api/levels", response_model=list[LevelSummary])
def list_levels():
    out: list[LevelSummary] = []
    for i, lvl in enumerate(LEVELS):
        out.append(
            LevelSummary(
                index=i,
                title=lvl.title,
                height=lvl.height,
                width=lvl.width,
                num_boxes=len(lvl.initial_boxes),
            )
        )
    return out


@app.post("/api/session", response_model=SessionInfo)
def create_session(body: SessionCreate):
    if body.level_index >= len(LEVELS):
        raise HTTPException(404, "level_index out of range")
    level = LEVELS[body.level_index]
    sess = STORE.create(level, body.level_index)
    return _session_info(sess)


@app.get("/api/session/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    sess = STORE.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    return _session_info(sess)


@app.post("/api/session/{session_id}/reset", response_model=SessionInfo)
def reset_session(session_id: str):
    sess = STORE.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    sim = Simulator(sess.level)
    sess.state = sim.initial_state()
    sess.move_count = 0
    sess.undo_stack.clear()
    return _session_info(sess)


@app.post("/api/session/{session_id}/move", response_model=MoveResponse)
def move(session_id: str, body: MoveRequest):
    sess = STORE.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    sim = Simulator(sess.level)
    res = sim.try_step(sess.state, body.action)
    if res is None:
        return MoveResponse(
            board=_board_payload(sess.level, sess.state),
            move_count=sess.move_count,
            illegal=True,
            can_undo=bool(sess.undo_stack),
        )
    sess.undo_stack.append((sess.state, sess.move_count))
    sess.state = res.state
    sess.move_count += 1
    return MoveResponse(
        board=_board_payload(sess.level, sess.state),
        move_count=sess.move_count,
        illegal=False,
        can_undo=True,
    )


@app.post("/api/session/{session_id}/undo", response_model=MoveResponse)
def undo(session_id: str):
    sess = STORE.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    if not sess.undo_stack:
        raise HTTPException(400, "nothing to undo")
    prev_state, prev_mc = sess.undo_stack.pop()
    sess.state = prev_state
    sess.move_count = prev_mc
    return MoveResponse(
        board=_board_payload(sess.level, sess.state),
        move_count=sess.move_count,
        illegal=False,
        can_undo=bool(sess.undo_stack),
    )


@app.get("/api/session/{session_id}/llm_preview", response_model=LlmPreviewResponse)
def llm_preview(
    session_id: str,
    encoding: StateEncoding = Query(default="ascii"),
):
    sess = STORE.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    text = encode_board_for_llm(sess.level, sess.state, encoding)
    return LlmPreviewResponse(encoding=encoding, text=text)


@app.post("/api/session/{session_id}/apply_plan", response_model=ApplyPlanResponse)
def apply_plan(session_id: str, body: ApplyPlanRequest):
    """Apply a list of legal moves in one shot (for instant replay / verification)."""
    sess = STORE.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    sess.undo_stack.clear()
    sim = Simulator(sess.level)
    applied = 0
    for i, a in enumerate(body.actions):
        res = sim.try_step(sess.state, a)
        if res is None:
            if body.stop_on_illegal:
                return ApplyPlanResponse(
                    board=_board_payload(sess.level, sess.state),
                    move_count=sess.move_count,
                    applied_steps=applied,
                    completed=False,
                    illegal_at_step=i,
                    can_undo=False,
                )
            break
        sess.state = res.state
        sess.move_count += 1
        applied += 1
        if sim.is_solved(sess.state):
            break
    return ApplyPlanResponse(
        board=_board_payload(sess.level, sess.state),
        move_count=sess.move_count,
        applied_steps=applied,
        completed=sim.is_solved(sess.state),
        illegal_at_step=None,
        can_undo=False,
    )


@app.post("/api/session/{session_id}/solve", response_model=SolveResponse)
async def solve(session_id: str, body: SolveRequest):
    sess = STORE.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    level = sess.level
    if body.method == "bfs":
        r = breadth_first_search(level, max_nodes=body.max_nodes)
        return SolveResponse(
            success=r.success,
            actions=list(r.actions),
            nodes_expanded=r.nodes_expanded,
            message=r.message,
            llm_calls=r.llm_calls,
            state_encoding="bfs",
            search_mode="bfs",
        )
    if body.predictor == "openai" and not openai_configured():
        raise HTTPException(
            400,
            "predictor=openai requires OPENAI_API_KEY in track_1/.env (see .env.example)",
        )
    try:
        predictor = make_predictor(
            body.predictor,
            mock_seed=0,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    enc = body.state_encoding

    def board_fn(lvl, st) -> str:
        return encode_board_for_llm(lvl, st, enc)

    try:
        if body.method == "llm_greedy":
            rr = await greedy_single_action_rollout_async(
                level,
                predictor,
                board_fn,
                max_steps=body.greedy_max_steps,
                max_llm_calls=body.greedy_max_llm_calls,
            )
            return SolveResponse(
                success=rr.success,
                actions=list(rr.actions),
                nodes_expanded=rr.steps_executed,
                message=rr.message,
                llm_calls=rr.llm_calls,
                state_encoding=enc,
                search_mode="llm_greedy",
            )

        br = await beam_search_llm_guided_async(
            level,
            predictor,
            board_fn,
            beam_width=body.beam_width,
            max_depth=body.beam_max_depth,
            max_llm_calls=body.beam_max_llm_calls,
        )
        return SolveResponse(
            success=br.success,
            actions=list(br.actions),
            nodes_expanded=br.nodes_expanded,
            message=br.message,
            llm_calls=br.llm_calls,
            state_encoding=enc,
            search_mode="beam",
        )
    finally:
        if isinstance(predictor, OpenAIPredictor):
            await predictor.close()

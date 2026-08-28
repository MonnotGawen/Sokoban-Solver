#!/usr/bin/env python3
"""
Compare planning strategies on the same levels: BFS, greedy one-step chain, beam (tree search).

Writes a CSV suitable for the report (success, plan length, nodes/steps, LLM calls, wall time).
OpenAI calls are each wrapped in try/except so a single API hiccup does not abort the run.

Examples:
  cd track_1 && python scripts/run_llm_comparison.py -n 5 -o research/output/llm_compare.csv
  OPENAI_API_KEY=... python scripts/run_llm_comparison.py -n 8 \
      --greedy-steps 60 --greedy-calls 60 \
      --beam-width 3 --beam-depth 30 --beam-calls 60
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from track1.api.levels_loader import load_levels
from track1.config import load_app_env, openai_configured
from track1.llm.beam import beam_search_llm_guided_async
from track1.llm.factory import make_predictor
from track1.llm.openai_predictor import OpenAIPredictor
from track1.llm.predictor import MockPredictor
from track1.llm.prompts import encode_board_for_llm
from track1.llm.rollout import greedy_single_action_rollout_async
from track1.sokoban.search import breadth_first_search

FIELDS = [
    "experiment",
    "level_index",
    "title",
    "height",
    "width",
    "num_boxes",
    "success",
    "plan_len",
    "nodes_expanded",
    "llm_calls",
    "wall_ms",
    "message",
]


def _meta(lvl, i: int) -> dict:
    title = (lvl.title or f"level_{i}").replace(",", ";")
    return {
        "level_index": i,
        "title": title,
        "height": lvl.height,
        "width": lvl.width,
        "num_boxes": len(lvl.initial_boxes),
    }


def _row_from_result(exp: str, meta: dict, wall_ms: float, result) -> dict:
    nodes = result.nodes_expanded if hasattr(result, "nodes_expanded") else result.steps_executed
    return {
        "experiment": exp,
        **meta,
        "success": int(result.success),
        "plan_len": len(result.actions),
        "nodes_expanded": nodes,
        "llm_calls": result.llm_calls,
        "wall_ms": round(wall_ms, 2),
        "message": getattr(result, "message", ""),
    }


def _error_row(exp: str, meta: dict, wall_ms: float, err: Exception) -> dict:
    return {
        "experiment": exp,
        **meta,
        "success": 0,
        "plan_len": 0,
        "nodes_expanded": 0,
        "llm_calls": 0,
        "wall_ms": round(wall_ms, 2),
        "message": f"error:{type(err).__name__}:{str(err)[:120]}",
    }


def main() -> None:
    load_app_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels-path", type=Path, default=None)
    ap.add_argument("-n", "--num-levels", type=int, default=8)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "research" / "output" / "llm_compare.csv",
    )
    ap.add_argument("--greedy-steps", type=int, default=80)
    ap.add_argument("--greedy-calls", type=int, default=80)
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--beam-depth", type=int, default=30)
    ap.add_argument("--beam-calls", type=int, default=60)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    quiet = args.quiet

    def log(msg: str) -> None:
        if not quiet:
            print(msg, file=sys.stderr, flush=True)

    levels = load_levels(args.levels_path)
    n = min(args.num_levels, len(levels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    oai = openai_configured()

    log(
        f"[run_llm_comparison] first {n}/{len(levels)} levels → {args.output} | "
        f"OPENAI={oai} | greedy=(steps={args.greedy_steps}, calls={args.greedy_calls}) | "
        f"beam=(width={args.beam_width}, depth={args.beam_depth}, calls={args.beam_calls})"
    )

    persistent_loop: asyncio.AbstractEventLoop | None = None
    if oai:
        persistent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(persistent_loop)

    def run_async(coro):
        if persistent_loop is not None:
            return persistent_loop.run_until_complete(coro)
        return asyncio.run(coro)

    try:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()

            def write(row: dict) -> None:
                w.writerow(row)
                f.flush()

            for i in range(n):
                lvl = levels[i]
                meta = _meta(lvl, i)
                log(f"\n[run_llm_comparison] === Level {i + 1}/{n}: {meta['title']!r} ===")

                def board_ascii(l, s):
                    return encode_board_for_llm(l, s, "ascii")

                # BFS baseline
                t = time.perf_counter()
                bfs = breadth_first_search(lvl)
                write(_row_from_result("bfs", meta, (time.perf_counter() - t) * 1000, bfs))
                log(
                    f"  BFS | success={bfs.success} | plan_len={len(bfs.actions)} | "
                    f"nodes={bfs.nodes_expanded}"
                )

                # Mock greedy
                t = time.perf_counter()
                gr_m = run_async(
                    greedy_single_action_rollout_async(
                        lvl,
                        MockPredictor(seed=i + 1),
                        board_ascii,
                        max_steps=args.greedy_steps,
                        max_llm_calls=args.greedy_calls,
                    )
                )
                write(_row_from_result("greedy_mock_ascii", meta, (time.perf_counter() - t) * 1000, gr_m))
                log(f"  greedy_mock | success={gr_m.success} | steps={gr_m.steps_executed}")

                # Mock beam
                t = time.perf_counter()
                br_m = run_async(
                    beam_search_llm_guided_async(
                        lvl,
                        MockPredictor(seed=i + 1),
                        board_ascii,
                        beam_width=args.beam_width,
                        max_depth=args.beam_depth,
                        max_llm_calls=args.beam_calls,
                    )
                )
                write(_row_from_result("beam_mock_ascii", meta, (time.perf_counter() - t) * 1000, br_m))
                log(
                    f"  beam_mock | success={br_m.success} | nodes={br_m.nodes_expanded} | "
                    f"{br_m.message}"
                )

                if not oai:
                    continue

                # OpenAI greedy
                async def _openai_greedy():
                    pred = make_predictor("auto", mock_seed=0)
                    if not isinstance(pred, OpenAIPredictor):
                        raise TypeError("expected OpenAIPredictor")
                    try:
                        return await greedy_single_action_rollout_async(
                            lvl,
                            pred,
                            board_ascii,
                            max_steps=args.greedy_steps,
                            max_llm_calls=args.greedy_calls,
                        )
                    finally:
                        await pred.close()

                t = time.perf_counter()
                try:
                    gr_o = run_async(_openai_greedy())
                    dt = (time.perf_counter() - t) * 1000
                    write(_row_from_result("greedy_openai_ascii", meta, dt, gr_o))
                    log(
                        f"  greedy_openai | success={gr_o.success} | "
                        f"llm_calls={gr_o.llm_calls} | {dt / 1000:.1f}s"
                    )
                except Exception as e:  # noqa: BLE001
                    dt = (time.perf_counter() - t) * 1000
                    write(_error_row("greedy_openai_ascii", meta, dt, e))
                    log(f"  greedy_openai ERROR: {type(e).__name__}: {e}")

                # OpenAI beam
                async def _openai_beam():
                    pred = make_predictor("auto", mock_seed=0)
                    if not isinstance(pred, OpenAIPredictor):
                        raise TypeError("expected OpenAIPredictor")
                    try:
                        return await beam_search_llm_guided_async(
                            lvl,
                            pred,
                            board_ascii,
                            beam_width=args.beam_width,
                            max_depth=args.beam_depth,
                            max_llm_calls=args.beam_calls,
                        )
                    finally:
                        await pred.close()

                t = time.perf_counter()
                try:
                    br_o = run_async(_openai_beam())
                    dt = (time.perf_counter() - t) * 1000
                    write(_row_from_result("beam_openai_ascii", meta, dt, br_o))
                    log(
                        f"  beam_openai | success={br_o.success} | nodes={br_o.nodes_expanded} | "
                        f"llm_calls={br_o.llm_calls} | {dt / 1000:.1f}s | {br_o.message}"
                    )
                except Exception as e:  # noqa: BLE001
                    dt = (time.perf_counter() - t) * 1000
                    write(_error_row("beam_openai_ascii", meta, dt, e))
                    log(f"  beam_openai ERROR: {type(e).__name__}: {e}")

        print(f"Wrote {args.output}")
    finally:
        if persistent_loop is not None:
            try:
                pending = [t for t in asyncio.all_tasks(persistent_loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    persistent_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            try:
                persistent_loop.run_until_complete(persistent_loop.shutdown_asyncgens())
            except Exception:
                pass
            persistent_loop.close()
            asyncio.set_event_loop(None)


if __name__ == "__main__":
    main()

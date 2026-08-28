#!/usr/bin/env python3
"""
Run a small experiment matrix: BFS + LLM predictor rollouts (mock / optional OpenAI).

Writes a CSV to ``research/output/`` for the written report.

Example::

    cd track_1
    python scripts/run_experiments.py -n 12
    python scripts/run_experiments.py -n 2 --openai-rollouts
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
from track1.llm.prompts import StateEncoding, encode_board_for_llm
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
]


def main() -> None:
    load_app_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels-path", type=Path, default=None)
    ap.add_argument("-n", "--num-levels", type=int, default=12)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "research" / "output" / "latest.csv",
    )
    ap.add_argument(
        "--encodings",
        nargs="*",
        default=["ascii", "labeled_rows", "tuple_fields"],
        choices=["ascii", "labeled_rows", "tuple_fields"],
    )
    ap.add_argument(
        "--openai-rollouts",
        action="store_true",
        help="Also run greedy rollout with predictor=auto (OpenAI if key set); only ascii encoding to limit cost",
    )
    ap.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Disable progress lines (still prints final path or errors).",
    )
    args = ap.parse_args()
    quiet = args.quiet

    def log(msg: str) -> None:
        if not quiet:
            print(msg, file=sys.stderr, flush=True)

    levels = load_levels(args.levels_path)
    n = min(args.num_levels, len(levels))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    encodings: list[StateEncoding] = list(args.encodings)  # type: ignore[arg-type]
    oai_ok = openai_configured()

    log(
        f"[run_experiments] levels_file={args.levels_path or 'default'} | "
        f"running first {n}/{len(levels)} levels | encodings={list(encodings)} | "
        f"--openai-rollouts={args.openai_rollouts} | OPENAI ready={oai_ok}"
    )
    if args.openai_rollouts and not oai_ok:
        log("[run_experiments] WARN: --openai-rollouts set but no API key → OpenAI rows skipped.")

    # One loop for the whole run when OpenAI is used: repeated asyncio.run() closes the loop
    # while httpx/OpenAI cleanup still targets it → "Event loop is closed" noise and races.
    use_shared_loop = args.openai_rollouts and oai_ok
    persistent_loop: asyncio.AbstractEventLoop | None = None
    if use_shared_loop:
        persistent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(persistent_loop)
        log("[run_experiments] Using one persistent asyncio loop for OpenAI + mock async steps.")

    def run_async(coro):
        if persistent_loop is not None:
            return persistent_loop.run_until_complete(coro)
        return asyncio.run(coro)

    try:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for i in range(n):
                lvl = levels[i]
                title = (lvl.title or f"level_{i}").replace(",", ";")
                meta = {
                    "level_index": i,
                    "title": title,
                    "height": lvl.height,
                    "width": lvl.width,
                    "num_boxes": len(lvl.initial_boxes),
                }

                log(f"\n[run_experiments] === Level {i + 1}/{n}: {title!r} ({meta['width']}×{meta['height']}) ===")

                log("  BFS …")
                t0 = time.perf_counter()
                bfs = breadth_first_search(lvl)
                dt = (time.perf_counter() - t0) * 1000
                w.writerow(
                    {
                        "experiment": "bfs",
                        **meta,
                        "success": int(bfs.success),
                        "plan_len": len(bfs.actions),
                        "nodes_expanded": bfs.nodes_expanded,
                        "llm_calls": bfs.llm_calls,
                        "wall_ms": round(dt, 2),
                    }
                )
                log(
                    f"  BFS done in {dt:.0f} ms | success={bfs.success} | "
                    f"plan_len={len(bfs.actions)} | nodes={bfs.nodes_expanded}"
                )

                for enc in encodings:

                    def board_fn(l, s, e=enc):
                        return encode_board_for_llm(l, s, e)

                    async def _greedy(level, encoding: StateEncoding, seed: int):
                        mp = MockPredictor(seed=seed)

                        def bf(l, s, e=encoding):
                            return encode_board_for_llm(l, s, e)

                        return await greedy_single_action_rollout_async(
                            level,
                            mp,
                            bf,
                            max_steps=120,
                            max_llm_calls=400,
                        )

                    log(f"    greedy_mock …")
                    t_g = time.perf_counter()
                    gr = run_async(_greedy(lvl, enc, i))
                    dt_g = (time.perf_counter() - t_g) * 1000
                    w.writerow(
                        {
                            "experiment": f"greedy_mock_{enc}",
                            **meta,
                            "success": int(gr.success),
                            "plan_len": len(gr.actions),
                            "nodes_expanded": gr.steps_executed,
                            "llm_calls": gr.llm_calls,
                            "wall_ms": round(dt_g, 2),
                        }
                    )
                    log(
                        f"    greedy_mock done in {dt_g:.0f} ms | success={gr.success} | "
                        f"steps={gr.steps_executed} | llm_calls={gr.llm_calls} (mock)"
                    )

                    async def _beam(level, encoding: StateEncoding, seed: int):
                        mp = MockPredictor(seed=seed)

                        def bf(l, s, e=encoding):
                            return encode_board_for_llm(l, s, e)

                        return await beam_search_llm_guided_async(
                            level,
                            mp,
                            bf,
                            beam_width=3,
                            max_depth=30,
                            max_llm_calls=150,
                        )

                    log("    beam_mock …")
                    t_b = time.perf_counter()
                    br = run_async(_beam(lvl, enc, i))
                    dt_b = (time.perf_counter() - t_b) * 1000
                    w.writerow(
                        {
                            "experiment": f"beam_mock_{enc}",
                            **meta,
                            "success": int(br.success),
                            "plan_len": len(br.actions),
                            "nodes_expanded": br.nodes_expanded,
                            "llm_calls": br.llm_calls,
                            "wall_ms": round(dt_b, 2),
                        }
                    )
                    log(
                        f"    beam_mock done in {dt_b:.0f} ms | success={br.success} | "
                        f"nodes={br.nodes_expanded} | llm_calls={br.llm_calls} (mock) | {br.message}"
                    )

                if args.openai_rollouts and oai_ok:
                    log(
                        "  OpenAI block: greedy rollout only — one API call per predicted step."
                    )

                    async def _openai_block_for_level(level):
                        def bf(l, s):
                            return encode_board_for_llm(l, s, "ascii")

                        rows: list[tuple[str, float, object]] = []
                        log("    [OpenAI] greedy_auto_ascii starting (1 API call per greedy step)…")
                        pred_g = make_predictor("auto", mock_seed=0)
                        if not isinstance(pred_g, OpenAIPredictor):
                            raise TypeError("openai greedy requires OpenAIPredictor")
                        t4 = time.perf_counter()
                        try:
                            gr_o = await greedy_single_action_rollout_async(
                                level,
                                pred_g,
                                bf,
                                max_steps=150,
                                max_llm_calls=400,
                            )
                        finally:
                            await pred_g.close()
                        dt4 = (time.perf_counter() - t4) * 1000
                        rows.append(("greedy_auto_ascii", dt4, gr_o))
                        log(
                            f"    [OpenAI] greedy_auto_ascii done in {dt4 / 1000:.1f}s | "
                            f"success={gr_o.success} | steps={gr_o.steps_executed} | "
                            f"llm_calls={gr_o.llm_calls} (API calls)"
                        )

                        log("    [OpenAI] beam_auto_ascii starting (1 API call per beam node)…")
                        pred_b = make_predictor("auto", mock_seed=0)
                        if not isinstance(pred_b, OpenAIPredictor):
                            raise TypeError("openai beam requires OpenAIPredictor")
                        t5 = time.perf_counter()
                        try:
                            br_o = await beam_search_llm_guided_async(
                                level,
                                pred_b,
                                bf,
                                beam_width=3,
                                max_depth=30,
                                max_llm_calls=150,
                            )
                        finally:
                            await pred_b.close()
                        dt5 = (time.perf_counter() - t5) * 1000
                        rows.append(("beam_auto_ascii", dt5, br_o))
                        log(
                            f"    [OpenAI] beam_auto_ascii done in {dt5 / 1000:.1f}s | "
                            f"success={br_o.success} | nodes={br_o.nodes_expanded} | "
                            f"llm_calls={br_o.llm_calls} (API calls) | {br_o.message}"
                        )
                        return rows

                    for exp_name, dt_ms, result in run_async(_openai_block_for_level(lvl)):
                        nodes = (
                            result.nodes_expanded
                            if hasattr(result, "nodes_expanded")
                            else result.steps_executed
                        )
                        w.writerow(
                            {
                                "experiment": exp_name,
                                **meta,
                                "success": int(result.success),
                                "plan_len": len(result.actions),
                                "nodes_expanded": nodes,
                                "llm_calls": result.llm_calls,
                                "wall_ms": round(dt_ms, 2),
                            }
                        )

        print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
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

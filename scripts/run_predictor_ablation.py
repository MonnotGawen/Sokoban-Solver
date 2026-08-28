#!/usr/bin/env python3
"""
Run structured ablations for the LLM predictor behavior.

Focus:
- BFS baseline once per level
- Greedy rollout sensitivity to max_steps / max_llm_calls / encoding (mock)
- Beam search (tree search guided by the LLM) mock sweep over beam_width
- Optional OpenAI factorial grid: encoding x prompt_style x samples (greedy + beam)
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
from track1.llm.openai_predictor import OpenAIPredictor, PromptStyle
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
    "method",
    "predictor",
    "encoding",
    "prompt_style",
    "samples",
    "beam_width",
    "max_steps",
    "max_llm_calls",
    "success",
    "plan_len",
    "nodes_expanded",
    "llm_calls",
    "cache_hits",
    "wall_ms",
    "message",
]

ENCODING_CHOICES: list[str] = ["ascii", "labeled_rows", "tuple_fields", "ascii_features"]


def main() -> None:
    load_app_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels-path", type=Path, default=None)
    ap.add_argument("-n", "--num-levels", type=int, default=10)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "research" / "output" / "predictor_ablation.csv",
    )
    ap.add_argument(
        "--encodings",
        nargs="*",
        default=["ascii", "labeled_rows", "tuple_fields", "ascii_features"],
        choices=ENCODING_CHOICES,
    )
    ap.add_argument("--greedy-steps", nargs="*", type=int, default=[40, 120, 240])
    ap.add_argument("--greedy-calls", nargs="*", type=int, default=[40, 120, 240])
    ap.add_argument(
        "--beam-widths",
        nargs="*",
        type=int,
        default=[2, 3, 5],
        help="Grid for beam_width (mock + OpenAI). Set to 0 to disable beam rows.",
    )
    ap.add_argument(
        "--beam-max-depth",
        type=int,
        default=30,
        help="Beam search depth cap (shared across widths).",
    )
    ap.add_argument(
        "--beam-max-llm-calls",
        type=int,
        default=150,
        help="Beam search LLM-call budget per run.",
    )

    # OpenAI factorial grid (budget fixed).
    ap.add_argument(
        "--with-openai",
        action="store_true",
        help="Run an OpenAI factorial grid (encoding x prompt x samples) at fixed budget.",
    )
    ap.add_argument("--openai-budget-steps", type=int, default=80)
    ap.add_argument("--openai-budget-calls", type=int, default=80)
    ap.add_argument(
        "--openai-encodings",
        nargs="*",
        default=["ascii", "ascii_features"],
        choices=ENCODING_CHOICES,
        help="Encodings used for OpenAI rows.",
    )
    ap.add_argument(
        "--prompt-styles",
        nargs="*",
        default=["plain", "reason_act"],
        choices=["plain", "reason_act"],
    )
    ap.add_argument(
        "--samples",
        nargs="*",
        type=int,
        default=[1],
        help="Self-consistency sample counts (e.g. 1 3 5).",
    )
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
    encodings: list[StateEncoding] = list(args.encodings)  # type: ignore[arg-type]
    oai_encodings: list[StateEncoding] = list(args.openai_encodings)  # type: ignore[arg-type]
    greedy_steps: list[int] = sorted({x for x in args.greedy_steps if x > 0})
    greedy_calls: list[int] = sorted({x for x in args.greedy_calls if x > 0})
    beam_widths: list[int] = sorted({x for x in args.beam_widths if x > 0})
    samples_grid: list[int] = sorted({max(1, x) for x in args.samples})
    prompt_styles: list[PromptStyle] = list(args.prompt_styles)  # type: ignore[arg-type]

    log(
        f"[run_predictor_ablation] levels={n}/{len(levels)} | encodings={encodings} | "
        f"steps={greedy_steps} | calls={greedy_calls} | beams={beam_widths} | "
        f"with_openai={args.with_openai} | OPENAI={oai}"
    )
    if args.with_openai:
        log(
            f"[run_predictor_ablation] OpenAI grid: encodings={oai_encodings} "
            f"prompt_styles={prompt_styles} samples={samples_grid} "
            f"budget(steps={args.openai_budget_steps}, calls={args.openai_budget_calls})"
        )
    if args.with_openai and not oai:
        log("[run_predictor_ablation] WARN: --with-openai ignored (no OPENAI_API_KEY).")

    persistent_loop: asyncio.AbstractEventLoop | None = None
    if args.with_openai and oai:
        persistent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(persistent_loop)
        log("[run_predictor_ablation] Using one persistent asyncio loop.")

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
                log(f"\n[run_predictor_ablation] === Level {i + 1}/{n}: {title!r} ===")

                # ---------- BFS baseline ----------
                t_bfs = time.perf_counter()
                bfs = breadth_first_search(lvl)
                w.writerow(
                    {
                        "experiment": "bfs",
                        **meta,
                        "method": "bfs",
                        "predictor": "none",
                        "encoding": "n/a",
                        "prompt_style": "n/a",
                        "samples": 0,
                        "beam_width": 0,
                        "max_steps": 0,
                        "max_llm_calls": 0,
                        "success": int(bfs.success),
                        "plan_len": len(bfs.actions),
                        "nodes_expanded": bfs.nodes_expanded,
                        "llm_calls": bfs.llm_calls,
                        "cache_hits": 0,
                        "wall_ms": round((time.perf_counter() - t_bfs) * 1000, 2),
                        "message": bfs.message,
                    }
                )

                # ---------- Mock greedy sweep (budget sensitivity x encoding) ----------
                for enc in encodings:

                    def board_fn(l, s, e=enc):
                        return encode_board_for_llm(l, s, e)

                    for ms in greedy_steps:
                        for mc in greedy_calls:
                            t0 = time.perf_counter()
                            gr = run_async(
                                greedy_single_action_rollout_async(
                                    lvl,
                                    MockPredictor(seed=(i * 1009 + ms * 17 + mc * 19) % 100000),
                                    board_fn,
                                    max_steps=ms,
                                    max_llm_calls=mc,
                                )
                            )
                            w.writerow(
                                {
                                    "experiment": f"greedy_mock_{enc}_s{ms}_c{mc}",
                                    **meta,
                                    "method": "llm_greedy",
                                    "predictor": "mock",
                                    "encoding": enc,
                                    "prompt_style": "n/a",
                                    "samples": 0,
                                    "beam_width": 0,
                                    "max_steps": ms,
                                    "max_llm_calls": mc,
                                    "success": int(gr.success),
                                    "plan_len": len(gr.actions),
                                    "nodes_expanded": gr.steps_executed,
                                    "llm_calls": gr.llm_calls,
                                    "cache_hits": 0,
                                    "wall_ms": round((time.perf_counter() - t0) * 1000, 2),
                                    "message": gr.message,
                                }
                            )

                # ---------- Mock beam sweep (tree search width sensitivity x encoding) ----------
                for enc in encodings:

                    def board_fn(l, s, e=enc):
                        return encode_board_for_llm(l, s, e)

                    for bw in beam_widths:
                        t0 = time.perf_counter()
                        br = run_async(
                            beam_search_llm_guided_async(
                                lvl,
                                MockPredictor(seed=(i * 2017 + bw * 31) % 100000),
                                board_fn,
                                beam_width=bw,
                                max_depth=args.beam_max_depth,
                                max_llm_calls=args.beam_max_llm_calls,
                            )
                        )
                        w.writerow(
                            {
                                "experiment": f"beam_mock_{enc}_w{bw}",
                                **meta,
                                "method": "beam",
                                "predictor": "mock",
                                "encoding": enc,
                                "prompt_style": "n/a",
                                "samples": 0,
                                "beam_width": bw,
                                "max_steps": args.beam_max_depth,
                                "max_llm_calls": args.beam_max_llm_calls,
                                "success": int(br.success),
                                "plan_len": len(br.actions),
                                "nodes_expanded": br.nodes_expanded,
                                "llm_calls": br.llm_calls,
                                "cache_hits": 0,
                                "wall_ms": round((time.perf_counter() - t0) * 1000, 2),
                                "message": br.message,
                            }
                        )

                # ---------- OpenAI factorial grid ----------
                if args.with_openai and oai:
                    budget_steps = args.openai_budget_steps
                    budget_calls = args.openai_budget_calls

                    async def _run_openai_greedy(enc_, ps_, k_):
                        pred = OpenAIPredictor(
                            prompt_style=ps_,  # type: ignore[arg-type]
                            samples=k_,
                            enable_cache=True,
                        )
                        try:
                            return (
                                await greedy_single_action_rollout_async(
                                    lvl,
                                    pred,
                                    lambda l, s, e=enc_: encode_board_for_llm(l, s, e),
                                    max_steps=budget_steps,
                                    max_llm_calls=budget_calls,
                                ),
                                pred.cache_hits,
                            )
                        finally:
                            await pred.close()

                    async def _run_openai_beam(enc_, ps_, k_, bw_):
                        pred = OpenAIPredictor(
                            prompt_style=ps_,  # type: ignore[arg-type]
                            samples=k_,
                            enable_cache=True,
                        )
                        try:
                            return (
                                await beam_search_llm_guided_async(
                                    lvl,
                                    pred,
                                    lambda l, s, e=enc_: encode_board_for_llm(l, s, e),
                                    beam_width=bw_,
                                    max_depth=args.beam_max_depth,
                                    max_llm_calls=args.beam_max_llm_calls,
                                ),
                                pred.cache_hits,
                            )
                        finally:
                            await pred.close()

                    for enc in oai_encodings:
                        for ps in prompt_styles:
                            for k in samples_grid:
                                t1 = time.perf_counter()
                                try:
                                    gr_o, hits = run_async(_run_openai_greedy(enc, ps, k))
                                    row = {
                                        "experiment": f"greedy_openai_{enc}_{ps}_k{k}",
                                        **meta,
                                        "method": "llm_greedy",
                                        "predictor": "openai",
                                        "encoding": enc,
                                        "prompt_style": ps,
                                        "samples": k,
                                        "beam_width": 0,
                                        "max_steps": budget_steps,
                                        "max_llm_calls": budget_calls,
                                        "success": int(gr_o.success),
                                        "plan_len": len(gr_o.actions),
                                        "nodes_expanded": gr_o.steps_executed,
                                        "llm_calls": gr_o.llm_calls,
                                        "cache_hits": hits,
                                        "wall_ms": round((time.perf_counter() - t1) * 1000, 2),
                                        "message": gr_o.message,
                                    }
                                    log(
                                        f"  greedy_openai[{enc}/{ps}/k{k}] success={gr_o.success} "
                                        f"calls={gr_o.llm_calls} hits={hits}"
                                    )
                                except Exception as e:  # noqa: BLE001
                                    row = {
                                        "experiment": f"greedy_openai_{enc}_{ps}_k{k}",
                                        **meta,
                                        "method": "llm_greedy",
                                        "predictor": "openai",
                                        "encoding": enc,
                                        "prompt_style": ps,
                                        "samples": k,
                                        "beam_width": 0,
                                        "max_steps": budget_steps,
                                        "max_llm_calls": budget_calls,
                                        "success": 0,
                                        "plan_len": 0,
                                        "nodes_expanded": 0,
                                        "llm_calls": 0,
                                        "cache_hits": 0,
                                        "wall_ms": round((time.perf_counter() - t1) * 1000, 2),
                                        "message": f"error:{type(e).__name__}:{str(e)[:120]}",
                                    }
                                    log(f"  greedy_openai[{enc}/{ps}/k{k}] ERROR {type(e).__name__}: {e}")
                                w.writerow(row)
                                f.flush()

                                for bw in beam_widths:
                                    t_b = time.perf_counter()
                                    try:
                                        br_o, hits_b = run_async(_run_openai_beam(enc, ps, k, bw))
                                        row = {
                                            "experiment": f"beam_openai_{enc}_{ps}_k{k}_w{bw}",
                                            **meta,
                                            "method": "beam",
                                            "predictor": "openai",
                                            "encoding": enc,
                                            "prompt_style": ps,
                                            "samples": k,
                                            "beam_width": bw,
                                            "max_steps": args.beam_max_depth,
                                            "max_llm_calls": args.beam_max_llm_calls,
                                            "success": int(br_o.success),
                                            "plan_len": len(br_o.actions),
                                            "nodes_expanded": br_o.nodes_expanded,
                                            "llm_calls": br_o.llm_calls,
                                            "cache_hits": hits_b,
                                            "wall_ms": round((time.perf_counter() - t_b) * 1000, 2),
                                            "message": br_o.message,
                                        }
                                        log(
                                            f"  beam_openai[{enc}/{ps}/k{k}/w{bw}] "
                                            f"success={br_o.success} calls={br_o.llm_calls} "
                                            f"nodes={br_o.nodes_expanded} hits={hits_b}"
                                        )
                                    except Exception as e:  # noqa: BLE001
                                        row = {
                                            "experiment": f"beam_openai_{enc}_{ps}_k{k}_w{bw}",
                                            **meta,
                                            "method": "beam",
                                            "predictor": "openai",
                                            "encoding": enc,
                                            "prompt_style": ps,
                                            "samples": k,
                                            "beam_width": bw,
                                            "max_steps": args.beam_max_depth,
                                            "max_llm_calls": args.beam_max_llm_calls,
                                            "success": 0,
                                            "plan_len": 0,
                                            "nodes_expanded": 0,
                                            "llm_calls": 0,
                                            "cache_hits": 0,
                                            "wall_ms": round((time.perf_counter() - t_b) * 1000, 2),
                                            "message": f"error:{type(e).__name__}:{str(e)[:120]}",
                                        }
                                        log(f"  beam_openai[{enc}/{ps}/k{k}/w{bw}] ERROR {type(e).__name__}: {e}")
                                    w.writerow(row)
                                    f.flush()

        print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    finally:
        if persistent_loop is not None:
            try:
                pending = [t for t in asyncio.all_tasks(persistent_loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    persistent_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
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

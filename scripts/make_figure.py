"""Produce the headline figure for the report: success rate and median wall-time per method.

Reads ``research/output/llm_compare.csv`` and writes a two-panel PNG to
``research/figures/success_vs_walltime.png``. Matplotlib is only needed for this
script (``pip install matplotlib``).
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


METHOD_LABELS = {
    "bfs": "BFS (oracle)",
    "greedy_mock_ascii": "Greedy\n(mock)",
    "beam_mock_ascii": "Beam w=3\n(mock)",
    "greedy_openai_ascii": "Greedy\n(llama-3.1-8b)",
    "beam_openai_ascii": "Beam w=3\n(llama-3.1-8b)",
}
ORDER = list(METHOD_LABELS.keys())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="research/output/llm_compare.csv")
    ap.add_argument("-o", "--output", default="research/figures/success_vs_walltime.png")
    args = ap.parse_args()

    rows = defaultdict(list)
    with open(args.input) as f:
        for r in csv.DictReader(f):
            rows[r["experiment"]].append(r)

    methods = [m for m in ORDER if m in rows]
    success = [sum(1 for r in rows[m] if r["success"] == "1") / len(rows[m]) for m in methods]
    wall_med = [st.median(float(r["wall_ms"]) for r in rows[m]) for m in methods]

    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_s.bar(range(len(methods)), success, color="#3b82f6")
    ax_s.set_ylim(0, 1.05)
    ax_s.set_ylabel("Success rate")
    ax_s.set_title("Success rate on Microban 1–10")
    ax_s.set_xticks(range(len(methods)))
    ax_s.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
    for i, v in enumerate(success):
        ax_s.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    ax_w.bar(range(len(methods)), wall_med, color="#f59e0b")
    ax_w.set_yscale("log")
    ax_w.set_ylabel("Wall-time per level (ms, log)")
    ax_w.set_title("Median wall-time per level")
    ax_w.set_xticks(range(len(methods)))
    ax_w.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
    for i, v in enumerate(wall_med):
        ax_w.text(i, v * 1.15, f"{v:.0f}", ha="center", fontsize=9)

    fig.suptitle("BFS solves 9/10, all LLM-guided variants solve 0/10", y=1.02, fontsize=11)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

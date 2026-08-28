#!/usr/bin/env python3
"""Download David Skinner's Microban collection into track_1/data/microban.txt."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

DEFAULT_URL = (
    "http://www.abelmartin.com/rj/sokobanJS/Skinner/"
    "David%20W.%20Skinner%20-%20Sokoban_files/Microban.txt"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "microban.txt",
    )
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        r = client.get(args.url)
        r.raise_for_status()
        args.output.write_bytes(r.content)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

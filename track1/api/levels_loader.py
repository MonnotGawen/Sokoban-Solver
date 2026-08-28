from __future__ import annotations

import warnings
from pathlib import Path

from track1.sokoban.model import Level
from track1.sokoban.parser import parse_levels_from_text

SAMPLE = """
; Tiny 1
####
# .#
# $#
# @#
####

; Tiny 2
#####
#   #
#.$@#
#####
"""


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "microban.txt"


def load_levels(path: Path | None = None) -> list[Level]:
    p = path or default_data_path()
    if p.exists():
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            levels = parse_levels_from_text(text)
            if levels:
                return levels
        except ValueError as e:
            warnings.warn(f"Failed to parse {p}: {e}. Using bundled SAMPLE.", stacklevel=2)
    return parse_levels_from_text(SAMPLE)

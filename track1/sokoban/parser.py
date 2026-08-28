from __future__ import annotations

import re
from track1.sokoban.model import Level

# Microban / XSB-style characters
_VALID_CHARS = set("#@+$*. ")


def _looks_like_grid_row(line: str) -> bool:
    """True if the line can be part of a Sokoban map (skips HTML/preamble text)."""
    if not line.strip():
        return False
    return all(ch in _VALID_CHARS for ch in line.rstrip("\n"))


def _strip_trailing_spaces(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    max_len = max(len(line.rstrip(" ")) for line in lines)
    return [line.rstrip(" ").ljust(max_len) for line in lines]


def parse_level_grid(lines: list[str], title: str | None = None) -> Level:
    """Parse a rectangular ASCII grid into a Level."""
    lines = [line.expandtabs() for line in lines]
    lines = _strip_trailing_spaces(lines)
    if not lines:
        raise ValueError("Empty level")

    height = len(lines)
    width = max(len(row) for row in lines)
    rows = [row.ljust(width) for row in lines]

    walls: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    boxes: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None

    for r, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"Non-rectangular row {r}: {row!r}")
        for c, ch in enumerate(row):
            if ch not in _VALID_CHARS:
                raise ValueError(f"Invalid character {ch!r} at ({r},{c})")
            if ch == "#":
                walls.add((r, c))
            elif ch == ".":
                goals.add((r, c))
            elif ch == "$":
                boxes.add((r, c))
            elif ch == "*":
                boxes.add((r, c))
                goals.add((r, c))
            elif ch == "@":
                if player is not None:
                    raise ValueError("Multiple players")
                player = (r, c)
            elif ch == "+":
                if player is not None:
                    raise ValueError("Multiple players")
                player = (r, c)
                goals.add((r, c))
            elif ch == " ":
                pass

    if player is None:
        raise ValueError("No player (@ or +) in level")

    return Level(
        height=height,
        width=width,
        walls=frozenset(walls),
        goals=frozenset(goals),
        initial_boxes=frozenset(boxes),
        initial_player=player,
        title=title,
    )


def parse_levels_from_text(text: str) -> list[Level]:
    """
    Split Microban-style text into levels.

    Supports:
    - comment lines starting with ';' (optional title like '; 1')
    - blank line separators
    - consecutive grid blocks
    """
    raw_lines = text.splitlines()
    levels: list[Level] = []
    buf: list[str] = []
    pending_title: str | None = None

    def flush_buffer() -> None:
        nonlocal buf, pending_title
        if not buf:
            return
        levels.append(parse_level_grid(buf, title=pending_title))
        buf = []
        pending_title = None

    title_re = re.compile(r"^;\s*(.+?)\s*$")

    for line in raw_lines:
        if line.strip() == "":
            flush_buffer()
            continue
        if line.lstrip().startswith(";"):
            m = title_re.match(line.strip())
            flush_buffer()
            pending_title = m.group(1) if m else None
            continue
        raw = line.rstrip("\n")
        if not _looks_like_grid_row(raw):
            flush_buffer()
            continue
        buf.append(raw)

    flush_buffer()
    return levels

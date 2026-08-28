from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

TRACK1_ROOT: Path = Path(__file__).resolve().parents[1]


def load_app_env() -> None:
    env_file = TRACK1_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def openai_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())

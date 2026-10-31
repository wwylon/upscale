"""Shared server state accessors for APIRouter modules.

Routers import ``D`` and read attributes; values always come from the live
``upscaler.server`` module so test monkeypatches keep working.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    import upscaler.server as server

    try:
        return getattr(server, name)
    except AttributeError as exc:
        raise AttributeError(name) from exc

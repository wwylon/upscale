"""Optional paper / native algorithm plugins.

``load_plugins()`` imports every non-private submodule in this package so each
module can call ``register_algorithm``. Failures in optional deps (e.g. missing
``vtracer``) are skipped; other import errors are logged once.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Iterable

logger = logging.getLogger(__name__)

_LOADED = False

# Skip docs / native stubs
_SKIP = frozenset({"native"})


def _iter_plugin_modules() -> Iterable[str]:
    import upscaler.algorithms.plugins as pkg

    for mod in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        short = mod.name.rsplit(".", 1)[-1]
        if short.startswith("_") or short in _SKIP:
            continue
        yield mod.name


def load_plugins() -> list[str]:
    """Import plugin modules. Returns list of successfully imported module names."""
    global _LOADED
    if _LOADED:
        return []
    _LOADED = True
    loaded: list[str] = []
    for name in _iter_plugin_modules():
        try:
            importlib.import_module(name)
            loaded.append(name)
        except ImportError as exc:
            # Optional dependency missing (vectorize, etc.)
            logger.debug("Skip plugin %s: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plugin %s failed to load: %s", name, exc)
    return loaded


__all__ = ["load_plugins"]

"""asyncio compatibility shim for nest_asyncio + modern uvicorn.

Why this exists:
  * app.py relies on nest_asyncio (nested loops in Gradio callbacks).
  * nest_asyncio's patched ``asyncio.run`` predates the ``loop_factory``
    keyword that Python 3.12+/uvicorn>=0.35 now forwards
    (`asyncio_run(..., loop_factory=...)`).
  * With uvicorn >= 0.35 the server thread dies with
    ``TypeError: ... unexpected keyword argument 'loop_factory'`` and on
    single-port hosts (HF Spaces) gradio then reports
    ``Cannot find empty port`` - reproduced locally with uvicorn 0.54.

Fix: apply nest_asyncio first, then re-wrap ``asyncio.run`` to swallow the
``loop_factory`` kwarg while keeping every other keyword. Idempotent, safe to
call from any module before the app boots.
"""
from __future__ import annotations

import asyncio
import logging

import nest_asyncio

logger = logging.getLogger(__name__)


def ensure_utf8_stdio() -> None:
    """Never fail on emoji in console output (gradio prints 🔨 banners).

    Windows cp1252 consoles kill the whole app on a UnicodeEncodeError from a
    single print(); Linux/HF already run UTF-8, so this is a no-op there.
    """
    import sys

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def apply() -> None:
    """Patch asyncio once: nest_asyncio (nested loops) + loop_factory shim."""
    nest_asyncio.apply()
    if getattr(asyncio.run, "_loop_factory_safe", False):
        return  # already patched

    _patched_run = asyncio.run

    def _safe_run(main, *, debug=None, **rest):
        rest.pop("loop_factory", None)
        return _patched_run(main, debug=debug, **rest)

    _safe_run._loop_factory_safe = True  # type: ignore[attr-defined]
    asyncio.run = _safe_run
    logger.info("asyncio shim active: nest_asyncio + loop_factory tolerance")

"""FastAPI route modules (included from server.app)."""

from __future__ import annotations

from fastapi import FastAPI


def include_routers(app: FastAPI) -> None:
    from .compare_routes import router as compare_router
    from .jobs import router as jobs_router
    from .models_routes import router as models_router
    from .system_routes import router as system_router

    app.include_router(jobs_router)
    app.include_router(compare_router)
    app.include_router(models_router)
    app.include_router(system_router)

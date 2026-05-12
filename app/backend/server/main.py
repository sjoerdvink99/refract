from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routers import analysis, counterfactuals, graph, metrics, nodes, runs
from server.settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Refract API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/api")
app.include_router(nodes.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(graph.router, prefix="/api")
app.include_router(counterfactuals.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

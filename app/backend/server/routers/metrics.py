from fastapi import APIRouter, HTTPException

from server.dependencies import ArtifactReaderDep
from schemas.metrics import RunMetrics

router = APIRouter(tags=["metrics"])


@router.get("/runs/{run_id}/metrics", response_model=RunMetrics)
async def get_metrics(run_id: str, reader: ArtifactReaderDep) -> RunMetrics:
    metrics = reader.read_metrics(run_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Metrics for run '{run_id}' not found")
    return metrics

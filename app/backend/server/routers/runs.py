from fastapi import APIRouter, HTTPException

from server.dependencies import ArtifactReaderDep
from schemas.projection import ProjectionData
from schemas.run import RunManifest, RunSummary

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(reader: ArtifactReaderDep) -> list[RunSummary]:
    return reader.list_runs()


@router.get("/runs/{run_id}/manifest", response_model=RunManifest)
async def get_manifest(run_id: str, reader: ArtifactReaderDep) -> RunManifest:
    manifest = reader.read_manifest(run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return manifest


@router.get("/runs/{run_id}/projection", response_model=ProjectionData)
async def get_projection(run_id: str, reader: ArtifactReaderDep) -> ProjectionData:
    data = reader.read_projection(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Projection for run '{run_id}' not found")
    return data

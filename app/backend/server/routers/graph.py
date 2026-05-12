from fastapi import APIRouter, HTTPException, Query

from server.dependencies import ArtifactReaderDep
from schemas.graph import EgoGraph, Subgraph

router = APIRouter(tags=["graph"])


@router.get("/runs/{run_id}/graph/ego/{node_id}", response_model=EgoGraph)
async def get_ego_graph(
    run_id: str,
    node_id: str,
    reader: ArtifactReaderDep,
    radius: int = Query(default=1, ge=1, le=2),
) -> EgoGraph:
    ego = reader.read_ego_graph(run_id, node_id, radius)
    if ego is None:
        raise HTTPException(
            status_code=404, detail=f"Node '{node_id}' not found in run '{run_id}'"
        )
    return ego


@router.get("/runs/{run_id}/graph/subgraph", response_model=Subgraph)
async def get_subgraph(
    run_id: str,
    reader: ArtifactReaderDep,
    node_ids: str = Query(..., description="Comma-separated node IDs"),
) -> Subgraph:
    ids = [n.strip() for n in node_ids.split(",") if n.strip()]
    if not ids:
        raise HTTPException(status_code=422, detail="node_ids must not be empty")
    return reader.read_subgraph(run_id, ids)

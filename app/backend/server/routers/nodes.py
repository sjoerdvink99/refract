from fastapi import APIRouter, HTTPException

from server.dependencies import ArtifactReaderDep
from schemas.node import NodeDetail

router = APIRouter(tags=["nodes"])


@router.get("/runs/{run_id}/nodes/{node_id}", response_model=NodeDetail)
async def get_node(run_id: str, node_id: str, reader: ArtifactReaderDep) -> NodeDetail:
    detail = reader.read_node_detail(run_id, node_id)
    if detail is None:
        raise HTTPException(
            status_code=404, detail=f"Node '{node_id}' not found in run '{run_id}'"
        )
    return detail

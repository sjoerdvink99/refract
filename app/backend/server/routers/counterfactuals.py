from fastapi import APIRouter, HTTPException

from server.dependencies import ArtifactReaderDep
from schemas.counterfactual import NodeCounterfactuals

router = APIRouter(tags=["counterfactuals"])


@router.get("/runs/{run_id}/counterfactuals/{node_id}", response_model=NodeCounterfactuals)
async def get_counterfactuals(
    run_id: str, node_id: str, reader: ArtifactReaderDep
) -> NodeCounterfactuals:
    data = reader.read_counterfactuals(run_id, node_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Counterfactuals for node '{node_id}' not found in run '{run_id}'",
        )
    return data

from fastapi import APIRouter, HTTPException

from server.dependencies import ArtifactReaderDep

router = APIRouter(tags=["analysis"])


@router.get("/runs/{run_id}/expert-profile")
async def get_expert_profile(run_id: str, reader: ArtifactReaderDep) -> dict:
    profile = reader.read_expert_profile(run_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No nodes data for run '{run_id}'")
    return profile


@router.get("/runs/{run_id}/training/history")
async def get_training_history(run_id: str, reader: ArtifactReaderDep) -> list[dict]:
    history = reader.read_training_history(run_id)
    if history is None:
        raise HTTPException(
            status_code=404, detail=f"No training history for run '{run_id}'"
        )
    return history

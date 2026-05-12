import json
from pathlib import Path

from schemas.run import RunManifest


def write_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.model_dump(), f, indent=2)


def read_manifest(path: Path) -> RunManifest | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return RunManifest.model_validate(data)

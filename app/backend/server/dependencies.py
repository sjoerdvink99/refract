from typing import Annotated

from fastapi import Depends

from server.settings import settings
from refract.artifacts.reader import ArtifactReader


def get_artifact_reader() -> ArtifactReader:
    return ArtifactReader(settings.artifacts_dir)


ArtifactReaderDep = Annotated[ArtifactReader, Depends(get_artifact_reader)]

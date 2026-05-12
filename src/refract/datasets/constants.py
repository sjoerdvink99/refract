from __future__ import annotations

BINARY_DATASETS: frozenset[str] = frozenset({"minesweeper", "tolokers", "questions"})
IMBALANCED_DATASETS: frozenset[str] = frozenset({"questions", "tolokers", "minesweeper"})
USE_ROC_AUC_STOPPING: frozenset[str] = frozenset({"minesweeper", "tolokers", "questions"})

ALL_DATASETS: tuple[str, ...] = (
    "cora", "citeseer", "pubmed",
    "texas", "cornell", "wisconsin",
    "chameleon", "squirrel", "actor",
    "roman-empire", "amazon-ratings",
    "minesweeper", "tolokers", "questions",
)

NUM_SPLITS: dict[str, int] = {ds: 10 for ds in ALL_DATASETS}

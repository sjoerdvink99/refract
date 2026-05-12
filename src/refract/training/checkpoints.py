from pathlib import Path

import torch


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model

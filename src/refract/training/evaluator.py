from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    roc_auc: float | None
    predictions: np.ndarray
    probabilities: np.ndarray
    true_labels: np.ndarray


def _safe_roc_auc(probs: np.ndarray, true: np.ndarray) -> float | None:
    try:
        if probs.shape[1] == 2:
            return float(roc_auc_score(true, probs[:, 1]))
        return float(roc_auc_score(true, probs, multi_class="ovr", average="macro"))
    except Exception:
        return None


def evaluate_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    regime: torch.Tensor | None = None,
    device: torch.device | None = None,
    compute_roc_auc: bool = False,
) -> EvalResult:
    if device is not None:
        model = model.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        y = y.to(device)
        mask = mask.to(device)
        if regime is not None:
            regime = regime.to(device)

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, regime) if regime is not None else model(x, edge_index)

        probs = out["probs"][mask].cpu().numpy()
        preds = probs.argmax(axis=1)
        true = y[mask].cpu().numpy()

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="macro", zero_division=0)
    roc_auc = _safe_roc_auc(probs, true) if compute_roc_auc else None

    return EvalResult(
        accuracy=float(acc),
        macro_f1=float(f1),
        roc_auc=roc_auc,
        predictions=preds,
        probabilities=probs,
        true_labels=true,
    )

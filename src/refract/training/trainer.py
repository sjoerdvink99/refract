import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.utils import dropout_edge

from refract.datasets.base import GraphDataset
from refract.models.losses import total_loss
from refract.training.checkpoints import load_checkpoint, save_checkpoint
from refract.training.evaluator import EvalResult, evaluate_model


@dataclass
class TrainingConfig:
    epochs: int = 300
    patience: int = 50
    lr: float = 1e-3
    weight_decay: float = 5e-4
    lambda_align: float = 0.05
    lambda_smooth: float = 0.01
    lambda_expert_aux: float = 0.05
    aux_floor: float = 0.2
    lambda_diversity: float = 0.0
    lambda_load: float = 0.0
    lambda_entropy: float = 0.0
    lambda_z_loss: float = 0.005
    T_gate_start: float = 2.0
    T_gate_end: float = 1.0
    T_gate_anneal: int = 100
    device: str = "cpu"
    class_weighted: bool = False
    drop_edge_p: float = 0.0
    use_roc_auc_stopping: bool = False
    grad_clip: float = 1.0
    warmup_epochs: int = 10


@dataclass
class GateSnapshot:
    epoch: int
    gate_attribute_mean: float
    gate_attribute_std: float
    gate_topology_mean: float
    gate_topology_std: float


@dataclass
class TrainingResult:
    best_val_acc: float
    test_result: EvalResult
    train_history: list[dict] = field(default_factory=list)
    gate_history: list[GateSnapshot] = field(default_factory=list)
    best_epoch: int = 0


def _make_scheduler(
    optimizer: optim.Optimizer, config: TrainingConfig
) -> SequentialLR | CosineAnnealingLR:
    remaining = max(1, config.epochs - config.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
    if config.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])
    return cosine


def train_routed_model(
    model: torch.nn.Module,
    dataset: GraphDataset,
    regime: torch.Tensor,
    config: TrainingConfig,
    checkpoint_path: Path | None = None,
) -> TrainingResult:
    device = torch.device(config.device)
    model = model.to(device)

    x = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    y = dataset.y.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    regime_d = regime.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = _make_scheduler(optimizer, config)

    class_weights: torch.Tensor | None = None
    if config.class_weighted:
        counts = torch.bincount(dataset.y[dataset.train_mask])
        class_weights = (counts.sum() / (len(counts) * counts.float())).to(device)

    best_val_score = 0.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    gate_history: list[GateSnapshot] = []

    use_tmp = checkpoint_path is None
    if use_tmp:
        tmp_fd, tmp_str = tempfile.mkstemp(suffix=".pt", prefix="refract_")
        os.close(tmp_fd)
        tmp_path: Path = Path(tmp_str)
    else:
        tmp_path = checkpoint_path

    try:
        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()

            progress = min(1.0, epoch / max(1, config.T_gate_anneal))
            temperature = config.T_gate_start * (1.0 - progress) + config.T_gate_end * progress

            train_ei = edge_index
            if config.drop_edge_p > 0.0:
                train_ei, _ = dropout_edge(edge_index, p=config.drop_edge_p, training=True)

            out = model(x, train_ei, regime_d, temperature=temperature)
            expert_hidden = out["expert_hidden"]
            loss, components = total_loss(
                out["logits"],
                y,
                train_mask,
                out["gate"],
                regime_d[:, 0],
                regime_d[:, 1],
                config.lambda_align,
                config.lambda_smooth,
                config.lambda_diversity,
                config.lambda_load,
                edge_index,
                regime_d,
                expert_hidden[:, 0, :],
                expert_hidden[:, 1, :],
                class_weights=class_weights,
                lambda_entropy=config.lambda_entropy,
                gate_logits=out.get("gate_logits"),
                lambda_z_loss=config.lambda_z_loss,
            )
            if config.lambda_expert_aux > 0:
                expert_logits = out["expert_logits"]
                decay = max(0.0, 1.0 - epoch / max(1, config.T_gate_anneal))
                for i in range(expert_logits.shape[1]):
                    w = config.lambda_expert_aux * (decay + config.aux_floor)
                    if w > 0:
                        loss = loss + w * F.cross_entropy(
                            expert_logits[:, i, :][train_mask], y[train_mask], weight=class_weights
                        )

            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            val_result = evaluate_model(
                model, x, edge_index, y, val_mask, regime_d,
                compute_roc_auc=config.use_roc_auc_stopping,
            )
            val_score = (
                val_result.roc_auc if config.use_roc_auc_stopping and val_result.roc_auc is not None
                else val_result.accuracy
            )
            history.append({
                "epoch": epoch,
                "loss": loss.item(),
                **components,
                "val_acc": val_result.accuracy,
                "val_roc": val_result.roc_auc,
            })

            gate_np = out["gate"].detach().cpu().numpy()
            gate_history.append(GateSnapshot(
                epoch=epoch,
                gate_attribute_mean=float(gate_np[:, 0].mean()),
                gate_attribute_std=float(gate_np[:, 0].std()),
                gate_topology_mean=float(gate_np[:, 1].mean()),
                gate_topology_std=float(gate_np[:, 1].std()),
            ))

            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(model, tmp_path)
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                break

        if tmp_path.exists():
            load_checkpoint(model, tmp_path)
    finally:
        if use_tmp and tmp_path.exists():
            tmp_path.unlink()

    test_result = evaluate_model(
        model, x, edge_index, y, dataset.test_mask.to(device), regime_d,
        compute_roc_auc=config.use_roc_auc_stopping,
    )

    return TrainingResult(
        best_val_acc=best_val_score,
        test_result=test_result,
        train_history=history,
        gate_history=gate_history,
        best_epoch=best_epoch,
    )


def train_baseline_model(
    model: torch.nn.Module,
    dataset: GraphDataset,
    config: TrainingConfig,
    checkpoint_path: Path | None = None,
) -> TrainingResult:
    device = torch.device(config.device)
    model = model.to(device)

    x = dataset.x.to(device)
    edge_index = dataset.edge_index.to(device)
    y = dataset.y.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = _make_scheduler(optimizer, config)

    bline_class_weights: torch.Tensor | None = None
    if config.class_weighted:
        counts = torch.bincount(dataset.y[dataset.train_mask])
        bline_class_weights = (counts.sum() / (len(counts) * counts.float())).to(device)

    best_val_score = 0.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []

    use_tmp = checkpoint_path is None
    if use_tmp:
        tmp_fd, tmp_str = tempfile.mkstemp(suffix=".pt", prefix="refract_")
        os.close(tmp_fd)
        tmp_path: Path = Path(tmp_str)
    else:
        tmp_path = checkpoint_path

    try:
        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()

            train_ei = edge_index
            if config.drop_edge_p > 0.0:
                train_ei, _ = dropout_edge(edge_index, p=config.drop_edge_p, training=True)
            out = model(x, train_ei)
            loss = F.cross_entropy(out["logits"][train_mask], y[train_mask], weight=bline_class_weights)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            val_result = evaluate_model(
                model, x, edge_index, y, val_mask,
                compute_roc_auc=config.use_roc_auc_stopping,
            )
            val_score = (
                val_result.roc_auc if config.use_roc_auc_stopping and val_result.roc_auc is not None
                else val_result.accuracy
            )
            history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "val_acc": val_result.accuracy,
                "val_roc": val_result.roc_auc,
            })

            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(model, tmp_path)
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                break

        if tmp_path.exists():
            load_checkpoint(model, tmp_path)
    finally:
        if use_tmp and tmp_path.exists():
            tmp_path.unlink()

    test_result = evaluate_model(
        model, x, edge_index, y, dataset.test_mask.to(device),
        compute_roc_auc=config.use_roc_auc_stopping,
    )

    return TrainingResult(
        best_val_acc=best_val_score,
        test_result=test_result,
        train_history=history,
        best_epoch=best_epoch,
    )

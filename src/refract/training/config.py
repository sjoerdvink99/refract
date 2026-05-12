from __future__ import annotations

from pathlib import Path

import yaml

from refract.datasets.constants import IMBALANCED_DATASETS, USE_ROC_AUC_STOPPING
from refract.training.trainer import TrainingConfig


def resolve_config(dataset_name: str, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        return p
    candidates = [
        Path(f"configs/hparams/{dataset_name}.yaml"),
        Path(f"configs/datasets/{dataset_name}.yaml"),
        Path("configs/datasets/default.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No config found for '{dataset_name}'. "
        f"Run: python scripts/tune.py {dataset_name} --device auto"
    )


def load_hps(config_path: Path) -> dict:
    with open(config_path) as f:
        doc = yaml.safe_load(f)
    return doc["config"] if "config" in doc else doc


def make_train_cfg(hps: dict, dataset_name: str, device: str) -> TrainingConfig:
    return TrainingConfig(
        epochs=hps["epochs"],
        patience=hps["patience"],
        lr=hps["lr"],
        weight_decay=hps["weight_decay"],
        lambda_align=hps["lambda_align"],
        lambda_smooth=hps["lambda_smooth"],
        lambda_expert_aux=hps["lambda_expert_aux"],
        aux_floor=hps["aux_floor"],
        T_gate_start=hps["T_gate_start"],
        T_gate_end=1.0,
        T_gate_anneal=hps["T_gate_anneal"],
        device=device,
        class_weighted=dataset_name in IMBALANCED_DATASETS,
        drop_edge_p=hps.get("drop_edge_p", 0.0),
        lambda_entropy=hps.get("lambda_entropy", 0.0),
        use_roc_auc_stopping=dataset_name in USE_ROC_AUC_STOPPING,
        lambda_z_loss=hps.get("lambda_z_loss", 0.005),
        grad_clip=hps.get("grad_clip", 1.0),
        warmup_epochs=hps.get("warmup_epochs", 10),
    )

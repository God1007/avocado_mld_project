from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .dataset import AvocadoDataset, collate_batch
from .losses import temporal_monotonicity_loss, view_consistency_loss
from .metadata import build_metadata_frame, make_instance_split
from .metrics import (
    accuracy_score,
    macro_f1_score,
    mean_absolute_error,
    mean_squared_error,
    monotonic_violation_rate,
    quadratic_weighted_kappa,
    r2_score,
    rank_correlation,
)
from .models import MLDModel, StageClassifier
from .pairs import build_temporal_pairs, build_view_pairs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_ordinal_targets(stage_index: torch.Tensor, num_stages: int) -> torch.Tensor:
    thresholds = torch.arange(num_stages - 1, device=stage_index.device)
    return (stage_index.unsqueeze(1) > thresholds).float()


def decode_ordinal_logits(ordinal_logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(ordinal_logits) > 0.5).sum(dim=1)


def batch_pairs(batch: dict[str, object]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    frame = pd.DataFrame(
        {
            "fruit_id": batch["fruit_id"],
            "day": batch["day"].cpu().tolist(),
            "view": batch["view"],
        }
    )
    return build_temporal_pairs(frame), build_view_pairs(frame)


def mld_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    config: TrainingConfig,
    num_stages: int = 5,
) -> tuple[torch.Tensor, dict[str, float]]:
    stage_index = batch["stage_index"]
    remaining_days = batch["remaining_days"]
    temporal_pairs, view_pairs = batch_pairs(batch)

    ordinal_targets = build_ordinal_targets(stage_index, num_stages)
    ord_loss = F.binary_cross_entropy_with_logits(outputs["ordinal_logits"], ordinal_targets)
    reg_loss = F.smooth_l1_loss(outputs["remaining_days"], remaining_days)
    temp_loss = temporal_monotonicity_loss(outputs["latent"], temporal_pairs, config.latent_margin)
    view_loss = view_consistency_loss(outputs["latent"], view_pairs)
    total = (
        config.lambda_ord * ord_loss
        + config.lambda_reg * reg_loss
        + config.lambda_temp * temp_loss
        + config.lambda_view * view_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "ord_loss": float(ord_loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "temp_loss": float(temp_loss.detach().cpu()),
        "view_loss": float(view_loss.detach().cpu()),
    }


def create_loader(frame: pd.DataFrame, config: TrainingConfig, shuffle: bool) -> DataLoader:
    dataset = AvocadoDataset(frame, image_root=config.image_root, image_size=config.image_size)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "image": batch["image"].to(device),
        "stage_index": batch["stage_index"].to(device),
        "remaining_days": batch["remaining_days"].to(device),
        "image_path": batch["image_path"],
        "fruit_id": batch["fruit_id"],
        "storage_condition": batch["storage_condition"],
        "day": batch["day"].to(device),
        "view": batch["view"],
    }


def build_prediction_frame(
    batch: dict[str, object],
    stage_pred: list[int],
    model_name: str,
    remaining_pred: list[float] | None = None,
    latent: list[float] | None = None,
) -> pd.DataFrame:
    row_count = len(stage_pred)
    remaining_values = remaining_pred if remaining_pred is not None else [float("nan")] * row_count
    latent_values = latent if latent is not None else [float("nan")] * row_count
    stage_true = [value + 1 for value in batch["stage_index"].cpu().tolist()]
    stage_pred_label = [value + 1 for value in stage_pred]
    return pd.DataFrame(
        {
            "model": [model_name] * row_count,
            "image_path": batch["image_path"],
            "fruit_id": batch["fruit_id"],
            "storage_condition": batch["storage_condition"],
            "day": batch["day"].cpu().tolist(),
            "view": batch["view"],
            "stage_true": stage_true,
            "stage_pred": stage_pred_label,
            "remaining_true": batch["remaining_days"].cpu().tolist(),
            "remaining_pred": remaining_values,
            "latent": latent_values,
        }
    )


def save_run_artifacts(
    output_dir: str | Path,
    metrics: dict[str, float],
    history: list[dict[str, float]],
    predictions: pd.DataFrame,
    model_state: dict[str, torch.Tensor] | None,
    config: TrainingConfig | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "test_metrics.json"
    history_path = output / "training_history.json"
    predictions_path = output / "test_predictions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)

    artifacts = {
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
        "predictions_path": str(predictions_path),
    }
    if config is not None:
        config_path = output / "run_config.json"
        config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        artifacts["config_path"] = str(config_path)
    if model_state is not None:
        model_path = output / "model.pt"
        torch.save(model_state, model_path)
        artifacts["model_path"] = str(model_path)
    return artifacts


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "ord_loss": 0.0, "reg_loss": 0.0, "temp_loss": 0.0, "view_loss": 0.0}
    steps = 0
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch["image"])
        loss, parts = mld_loss(outputs, batch, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for key, value in parts.items():
            running[key] += value
        steps += 1
    if steps == 0:
        return running
    return {key: value / steps for key, value in running.items()}


@torch.no_grad()
def evaluate_mld(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    model_name: str = "mld",
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    all_stage_true: list[int] = []
    all_stage_pred: list[int] = []
    all_remaining_true: list[float] = []
    all_remaining_pred: list[float] = []
    all_latent: list[float] = []
    temporal_pairs_all: list[tuple[int, int]] = []
    prediction_frames: list[pd.DataFrame] = []
    offset = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch["image"])
        stage_pred = decode_ordinal_logits(outputs["ordinal_logits"]).cpu().tolist()
        remaining_pred = outputs["remaining_days"].cpu().tolist()
        latent = outputs["latent"].cpu().tolist()
        all_stage_true.extend(batch["stage_index"].cpu().tolist())
        all_stage_pred.extend(stage_pred)
        all_remaining_true.extend(batch["remaining_days"].cpu().tolist())
        all_remaining_pred.extend(remaining_pred)
        all_latent.extend(latent)
        prediction_frames.append(
            build_prediction_frame(
                batch,
                stage_pred=stage_pred,
                remaining_pred=remaining_pred,
                latent=latent,
                model_name=model_name,
            )
        )

        local_temporal_pairs, _ = batch_pairs(batch)
        temporal_pairs_all.extend([(left + offset, right + offset) for left, right in local_temporal_pairs])
        offset += len(stage_pred)

    y_true = np.asarray(all_stage_true, dtype=int)
    y_pred = np.asarray(all_stage_pred, dtype=int)
    remaining_true = np.asarray(all_remaining_true, dtype=float)
    remaining_pred = np.asarray(all_remaining_pred, dtype=float)
    latent = np.asarray(all_latent, dtype=float)

    mse = mean_squared_error(remaining_true, remaining_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1_score(y_true, y_pred)),
        "qwk": float(quadratic_weighted_kappa(y_true, y_pred)),
        "mae": float(mean_absolute_error(remaining_true, remaining_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(remaining_true, remaining_pred)),
        "spearman": float(rank_correlation(remaining_true, remaining_pred)),
        "monotonic_violation_rate": float(monotonic_violation_rate(latent, temporal_pairs_all)),
    }
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def train_mld_model(config: TrainingConfig, model_name: str = "mld") -> dict[str, object]:
    set_seed(config.random_state)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_metadata_frame(config.metadata_csv)
    train_df, val_df, test_df = make_instance_split(frame, random_state=config.random_state)

    train_loader = create_loader(train_df, config, shuffle=True)
    val_loader = create_loader(val_df, config, shuffle=False)
    test_loader = create_loader(test_df, config, shuffle=False)

    device = resolve_device(config.device)
    model = MLDModel(num_stages=5).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_qwk = float("-inf")
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config)
        val_metrics, _ = evaluate_mld(model, val_loader, device, config, model_name=model_name)
        epoch_row = {"epoch": epoch + 1, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_row)
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, test_predictions = evaluate_mld(model, test_loader, device, config, model_name=model_name)
    artifacts = save_run_artifacts(
        output_dir=output_dir,
        metrics=test_metrics,
        history=history,
        predictions=test_predictions,
        model_state={name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        config=config,
    )
    return {
        "history": history,
        "test_metrics": test_metrics,
        "predictions": test_predictions,
        **artifacts,
    }


@torch.no_grad()
def evaluate_classifier(
    model: StageClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_frames: list[pd.DataFrame] = []
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        images = batch["image"]
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().tolist()
        y_true.extend(batch["stage_index"].cpu().tolist())
        y_pred.extend(pred)
        prediction_frames.append(build_prediction_frame(batch, stage_pred=pred, model_name="stage_classifier"))
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "macro_f1": float(macro_f1_score(y_true_array, y_pred_array)),
        "qwk": float(quadratic_weighted_kappa(y_true_array, y_pred_array)),
    }
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metrics, predictions


def train_stage_classifier(config: TrainingConfig, model_name: str = "stage_classifier") -> dict[str, object]:
    set_seed(config.random_state)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_metadata_frame(config.metadata_csv)
    train_df, val_df, test_df = make_instance_split(frame, random_state=config.random_state)
    train_loader = create_loader(train_df, config, shuffle=True)
    val_loader = create_loader(val_df, config, shuffle=False)
    test_loader = create_loader(test_df, config, shuffle=False)

    device = resolve_device(config.device)
    model = StageClassifier(num_stages=5).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_qwk = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["stage_index"].to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            steps += 1
        val_metrics, _ = evaluate_classifier(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": 0.0 if steps == 0 else running_loss / steps,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, test_predictions = evaluate_classifier(model, test_loader, device)
    if not test_predictions.empty:
        test_predictions["model"] = model_name
    artifacts = save_run_artifacts(
        output_dir=output_dir,
        metrics=test_metrics,
        history=history,
        predictions=test_predictions,
        model_state={name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        config=config,
    )
    return {
        "history": history,
        "test_metrics": test_metrics,
        "predictions": test_predictions,
        **artifacts,
    }


def run_baselines_from_config(config: TrainingConfig) -> dict[str, dict[str, float]]:
    root_output = Path(config.output_dir)
    root_output.mkdir(parents=True, exist_ok=True)
    classifier_config = TrainingConfig(**{**asdict(config), "output_dir": str(root_output / "stage_classifier")})
    baseline_config = TrainingConfig(
        **{
            **asdict(config),
            "output_dir": str(root_output / "multitask_no_pair_constraints"),
            "lambda_temp": 0.0,
            "lambda_view": 0.0,
        }
    )
    classifier_result = train_stage_classifier(classifier_config, model_name="stage_classifier")
    multitask_result = train_mld_model(baseline_config, model_name="multitask_no_pair_constraints")
    summary = {
        "stage_classifier": classifier_result["test_metrics"],
        "multitask_no_pair_constraints": multitask_result["test_metrics"],
    }
    (root_output / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "stage_classifier": classifier_result["test_metrics"],
        "multitask_no_pair_constraints": multitask_result["test_metrics"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the avocado monotonic latent degradation model.")
    parser.add_argument("--metadata-csv", default="data/metadata.csv")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = train_mld_model(config)
    print(json.dumps(result["test_metrics"], indent=2))


def run_baselines() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_baselines_from_config(config)
    print(json.dumps(result, indent=2))

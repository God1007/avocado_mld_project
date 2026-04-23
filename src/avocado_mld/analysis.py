from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from .pairs import build_temporal_pairs


MODEL_PALETTE = {
    "stage_classifier": "#355070",
    "multitask_no_pair_constraints": "#6D597A",
    "mld": "#E56B6F",
    "ambient": "#4C78A8",
    "cold": "#2A9D8F",
    "cool": "#577590",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "qwk": "QWK",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R^2",
    "spearman": "Spearman",
    "monotonic_violation_rate": "Monotonic Violations",
}

LOWER_IS_BETTER = {"mae", "rmse", "monotonic_violation_rate", "loss", "ord_loss", "reg_loss", "temp_loss", "view_loss"}


def _apply_plot_theme() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F7F5EF",
            "axes.facecolor": "#FFFDF8",
            "axes.edgecolor": "#3B3B3B",
            "axes.labelcolor": "#262626",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#262626",
            "ytick.color": "#262626",
            "grid.color": "#D7D1C4",
            "grid.alpha": 0.6,
            "font.size": 10,
            "savefig.facecolor": "#F7F5EF",
            "savefig.bbox": "tight",
        }
    )


def _color_for(label: str, fallback_index: int) -> str:
    if label in MODEL_PALETTE:
        return MODEL_PALETTE[label]
    cmap = plt.cm.get_cmap("tab10")
    return cmap(fallback_index % 10)


def build_results_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    metric_names: list[str] = []
    for metrics in results.values():
        for name in metrics:
            if name not in metric_names:
                metric_names.append(name)

    rows: list[dict[str, float | str]] = []
    for model_name, metrics in results.items():
        row: dict[str, float | str] = {"model": model_name}
        for metric_name in metric_names:
            row[metric_name] = float(metrics[metric_name]) if metric_name in metrics else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=["model", *metric_names])


def summarize_results_frame(frame: pd.DataFrame) -> dict[str, float | str]:
    best_row = frame.sort_values("qwk", ascending=False).iloc[0]
    return {
        "best_model": str(best_row["model"]),
        "best_qwk": float(best_row["qwk"]),
        "best_mae": float(best_row["mae"]),
    }


def summarize_group_metrics(frame: pd.DataFrame, group_col: str = "storage_condition") -> pd.DataFrame:
    if group_col not in frame.columns:
        raise ValueError(f"Missing grouping column: {group_col}")
    if "model" in frame.columns and group_col != "model":
        group_columns = ["model", group_col]
    else:
        group_columns = [group_col]
    rows: list[dict[str, float | str]] = []
    for key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key)}
        stage_true = group["stage_true"].to_numpy(dtype=int)
        stage_pred = group["stage_pred"].to_numpy(dtype=int)
        row["sample_count"] = int(len(group))
        row["accuracy"] = accuracy_score(stage_true, stage_pred)
        row["macro_f1"] = macro_f1_score(stage_true, stage_pred)
        row["qwk"] = quadratic_weighted_kappa(stage_true, stage_pred)

        if {"remaining_true", "remaining_pred"}.issubset(group.columns):
            valid = group["remaining_pred"].notna()
            if valid.any():
                remaining_true = group.loc[valid, "remaining_true"].to_numpy(dtype=float)
                remaining_pred = group.loc[valid, "remaining_pred"].to_numpy(dtype=float)
                row["mae"] = mean_absolute_error(remaining_true, remaining_pred)
                row["rmse"] = float(np.sqrt(mean_squared_error(remaining_true, remaining_pred)))
                row["r2"] = r2_score(remaining_true, remaining_pred)
                row["spearman"] = rank_correlation(remaining_true, remaining_pred)
            else:
                row["mae"] = float("nan")
                row["rmse"] = float("nan")
                row["r2"] = float("nan")
                row["spearman"] = float("nan")

        if {"latent", "fruit_id", "day"}.issubset(group.columns) and group["latent"].notna().any():
            temporal_source = group[["fruit_id", "day"]].reset_index(drop=True)
            temporal_pairs = build_temporal_pairs(temporal_source)
            row["monotonic_violation_rate"] = monotonic_violation_rate(
                group["latent"].to_numpy(dtype=float),
                temporal_pairs,
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(group_columns).reset_index(drop=True)


def _resolve_output_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _apply_plot_theme()
    return output


def plot_metric_comparison(
    frame: pd.DataFrame,
    output_path: str | Path,
    metrics: tuple[str, ...] = ("accuracy", "qwk", "mae", "spearman"),
) -> str:
    output = _resolve_output_path(output_path)
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), 4.8))
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        metric_frame = frame[["model", metric]].dropna().copy()
        ascending = metric in LOWER_IS_BETTER
        metric_frame = metric_frame.sort_values(metric, ascending=ascending)
        labels = metric_frame["model"].tolist()
        values = metric_frame[metric].astype(float).to_numpy()
        colors = [_color_for(label, index) for index, label in enumerate(labels)]
        bars = axis.barh(labels, values, color=colors, edgecolor="#2D2D2D", linewidth=0.9)
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.grid(axis="x", linestyle="--")
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                value,
                bar.get_y() + (bar.get_height() / 2),
                f" {value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
    figure.suptitle("Model Comparison Across Core Metrics", y=1.02, fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def plot_ablation_heatmap(
    frame: pd.DataFrame,
    output_path: str | Path,
    reference_model: str = "mld",
    metrics: tuple[str, ...] = ("accuracy", "qwk", "mae", "spearman", "monotonic_violation_rate"),
) -> str:
    output = _resolve_output_path(output_path)
    if reference_model not in set(frame["model"]):
        raise ValueError(f"Reference model '{reference_model}' not found.")
    reference_row = frame.set_index("model").loc[reference_model]
    ablations = frame[frame["model"] != reference_model].copy()
    heatmap = np.zeros((len(ablations), len(metrics)), dtype=float)
    for row_index, (_, row) in enumerate(ablations.iterrows()):
        for metric_index, metric in enumerate(metrics):
            direction = -1.0 if metric in LOWER_IS_BETTER else 1.0
            heatmap[row_index, metric_index] = (float(row[metric]) - float(reference_row[metric])) * direction

    figure, axis = plt.subplots(figsize=(1.7 * len(metrics) + 1.8, 0.8 * len(ablations) + 2.2))
    image = axis.imshow(heatmap, cmap="RdYlGn", aspect="auto")
    axis.set_xticks(np.arange(len(metrics)))
    axis.set_xticklabels([METRIC_LABELS.get(metric, metric) for metric in metrics], rotation=20, ha="right")
    axis.set_yticks(np.arange(len(ablations)))
    axis.set_yticklabels(ablations["model"].tolist())
    axis.set_title(f"Ablation Delta vs {reference_model}")
    for row_index in range(heatmap.shape[0]):
        for metric_index in range(heatmap.shape[1]):
            axis.text(
                metric_index,
                row_index,
                f"{heatmap[row_index, metric_index]:+.3f}",
                ha="center",
                va="center",
                color="#1F1F1F",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Signed improvement vs reference")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def plot_latent_trajectories(frame: pd.DataFrame, output_path: str | Path, max_fruits: int = 8) -> str:
    output = _resolve_output_path(output_path)
    figure, axis = plt.subplots(figsize=(9.4, 5.2))

    grouped = []
    for fruit_id, group in frame.groupby("fruit_id"):
        ordered = group.sort_values("day")
        grouped.append((fruit_id, ordered, float(ordered["latent"].max() - ordered["latent"].min())))
    grouped.sort(key=lambda item: item[2], reverse=True)
    selected = grouped[:max_fruits]

    for index, (fruit_id, ordered, _) in enumerate(selected):
        color = _color_for(str(fruit_id), index)
        axis.plot(
            ordered["day"],
            ordered["latent"],
            marker="o",
            linewidth=2.0,
            alpha=0.92,
            color=color,
            label=str(fruit_id),
        )
    axis.set_xlabel("Day")
    axis.set_ylabel("Latent degradation")
    axis.set_title("Representative Latent Trajectories")
    axis.grid(True, linestyle="--")
    if selected:
        axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def plot_stage_distribution(frame: pd.DataFrame, output_path: str | Path) -> str:
    output = _resolve_output_path(output_path)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    stages = sorted(frame["stage"].dropna().unique().tolist())
    latent_groups = [frame.loc[frame["stage"] == stage, "latent"].to_numpy(dtype=float) for stage in stages]
    violins = axis.violinplot(latent_groups, positions=stages, widths=0.8, showmeans=False, showmedians=True)
    for body_index, body in enumerate(violins["bodies"]):
        body.set_facecolor(_color_for(f"stage_{body_index}", body_index))
        body.set_alpha(0.45)
        body.set_edgecolor("#2D2D2D")
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        violins[key].set_color("#2D2D2D")

    rng = np.random.default_rng(42)
    for index, stage in enumerate(stages):
        latents = latent_groups[index]
        jitter = rng.uniform(-0.12, 0.12, size=len(latents))
        axis.scatter(
            np.full(len(latents), stage, dtype=float) + jitter,
            latents,
            s=18,
            alpha=0.35,
            color="#1F1F1F",
            edgecolors="none",
        )
    axis.set_title("Latent Distribution by Ripening Stage")
    axis.set_xlabel("Stage")
    axis.set_ylabel("Latent degradation")
    axis.grid(axis="y", linestyle="--")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def plot_stage_confusion(frame: pd.DataFrame, output_path: str | Path) -> str:
    output = _resolve_output_path(output_path)
    stages = sorted(set(frame["stage_true"]).union(set(frame["stage_pred"])))
    label_to_index = {label: index for index, label in enumerate(stages)}
    confusion = np.zeros((len(stages), len(stages)), dtype=int)
    for actual, predicted in zip(frame["stage_true"], frame["stage_pred"]):
        confusion[label_to_index[int(actual)], label_to_index[int(predicted)]] += 1

    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(confusion, cmap="YlOrBr")
    axis.set_xticks(np.arange(len(stages)))
    axis.set_yticks(np.arange(len(stages)))
    axis.set_xticklabels(stages)
    axis.set_yticklabels(stages)
    axis.set_xlabel("Predicted stage")
    axis.set_ylabel("True stage")
    axis.set_title("Stage Confusion Matrix")
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(confusion[row_index, column_index]),
                ha="center",
                va="center",
                color="#231F20",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def plot_remaining_scatter(frame: pd.DataFrame, output_path: str | Path, max_points: int = 900) -> str:
    output = _resolve_output_path(output_path)
    data = frame.dropna(subset=["remaining_true", "remaining_pred"]).copy()
    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42)
    figure, axis = plt.subplots(figsize=(6.6, 5.4))
    scatter = axis.scatter(
        data["remaining_true"],
        data["remaining_pred"],
        c=data["stage_true"],
        cmap="viridis",
        s=32,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.35,
    )
    lower = min(data["remaining_true"].min(), data["remaining_pred"].min())
    upper = max(data["remaining_true"].max(), data["remaining_pred"].max())
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#A23E48", linewidth=1.5)
    axis.set_xlabel("True remaining days")
    axis.set_ylabel("Predicted remaining days")
    axis.set_title("Shelf-Life Prediction Scatter")
    axis.grid(True, linestyle="--")
    figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label="True stage")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def plot_condition_metric_grid(
    frame: pd.DataFrame,
    output_path: str | Path,
    metrics: tuple[str, ...] = ("accuracy", "qwk", "mae", "spearman"),
    group_col: str = "storage_condition",
) -> str:
    output = _resolve_output_path(output_path)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    axes_flat = axes.flatten()
    for axis, metric in zip(axes_flat, metrics):
        metric_frame = frame[[group_col, metric]].dropna().copy().sort_values(metric, ascending=metric in LOWER_IS_BETTER)
        labels = metric_frame[group_col].astype(str).tolist()
        values = metric_frame[metric].astype(float).to_numpy()
        colors = [_color_for(label, index) for index, label in enumerate(labels)]
        bars = axis.bar(labels, values, color=colors, edgecolor="#2D2D2D", linewidth=0.9)
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", linestyle="--")
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + (bar.get_width() / 2),
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Condition-Wise Metric Breakdown", y=1.02, fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def plot_training_curves(history_frame: pd.DataFrame, output_path: str | Path) -> str:
    output = _resolve_output_path(output_path)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    if "model" not in history_frame.columns:
        history_frame = history_frame.assign(model="run")
    for index, (model_name, group) in enumerate(history_frame.groupby("model")):
        ordered = group.sort_values("epoch")
        color = _color_for(str(model_name), index)
        axes[0].plot(ordered["epoch"], ordered["loss"], marker="o", linewidth=2.0, color=color, label=str(model_name))
        if "val_qwk" in ordered.columns:
            axes[1].plot(ordered["epoch"], ordered["val_qwk"], marker="o", linewidth=2.0, color=color, label=str(model_name))
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--")
    axes[1].set_title("Validation QWK")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("QWK")
    axes[1].grid(True, linestyle="--")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def plot_metric_radar(
    frame: pd.DataFrame,
    output_path: str | Path,
    metrics: tuple[str, ...] = ("accuracy", "qwk", "mae", "spearman"),
) -> str:
    output = _resolve_output_path(output_path)
    normalized = frame[["model", *metrics]].copy()
    for metric in metrics:
        values = normalized[metric].astype(float)
        if metric in LOWER_IS_BETTER:
            values = -values
        minimum = values.min()
        maximum = values.max()
        if np.isclose(maximum, minimum):
            normalized[metric] = 1.0
        else:
            normalized[metric] = (values - minimum) / (maximum - minimum)

    angles = np.linspace(0.0, 2.0 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    figure = plt.figure(figsize=(6.6, 6.2))
    axis = figure.add_subplot(111, polar=True)
    for index, row in normalized.iterrows():
        values = row[list(metrics)].to_numpy(dtype=float)
        values = np.concatenate([values, [values[0]]])
        color = _color_for(str(row["model"]), index)
        axis.plot(angles, values, linewidth=2.2, color=color, label=str(row["model"]))
        axis.fill(angles, values, color=color, alpha=0.18)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels([METRIC_LABELS.get(metric, metric) for metric in metrics])
    axis.set_yticklabels([])
    axis.set_title("Normalized Model Radar", y=1.08, fontsize=13, fontweight="bold")
    axis.legend(loc="upper right", bbox_to_anchor=(1.22, 1.15), frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return str(output)


def save_results_table(frame: pd.DataFrame, output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return str(output)


def load_run_artifacts(run_dirs: dict[str, str | Path]) -> dict[str, pd.DataFrame]:
    result_rows: dict[str, dict[str, float]] = {}
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []

    for model_name, run_dir in run_dirs.items():
        directory = Path(run_dir)
        metrics_path = directory / "test_metrics.json"
        history_path = directory / "training_history.json"
        predictions_path = directory / "test_predictions.csv"

        result_rows[model_name] = json.loads(metrics_path.read_text(encoding="utf-8"))

        if history_path.exists():
            history = pd.DataFrame(json.loads(history_path.read_text(encoding="utf-8")))
            history["model"] = model_name
            history_frames.append(history)

        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            predictions["model"] = model_name
            prediction_frames.append(predictions)

    return {
        "results_frame": build_results_frame(result_rows),
        "history_frame": pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame(),
        "predictions_frame": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
    }

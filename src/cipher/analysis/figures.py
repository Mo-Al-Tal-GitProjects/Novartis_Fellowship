from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from cipher.config import ConfigContext
from cipher.provenance import atomic_write_json, sha256_file

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

MODEL_ORDER = ["clip", "dino", "plip", "uni"]
MODEL_LABELS = {name: name.upper() for name in MODEL_ORDER}
MODEL_COLORS = {
    "clip": "#687386",
    "dino": "#4472C4",
    "plip": "#B565A7",
    "uni": "#2A9D8F",
}


def _analysis_directory(context: ConfigContext, analysis_id: str) -> Path:
    directory = context.resolve(context.project.paths.artifacts_root) / "analyses" / analysis_id
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(f"analysis artifact {analysis_id!r} not found")
    return directory


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _map_figure(analysis: Path, output: Path) -> None:
    uncertainty = pd.read_parquet(analysis / "uncertainty.parquet")
    metrics = ["cosine", "inner_product", "euclidean"]
    labels = ["Cosine", "Raw inner product", "Raw Euclidean"]
    figure, axis = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(metrics), dtype=float)
    width = 0.18
    for offset, model in enumerate(MODEL_ORDER):
        rows = uncertainty.loc[uncertainty["model"] == model].set_index("metric").loc[metrics]
        means = rows["map"].to_numpy()
        errors = np.vstack(
            [means - rows["bootstrap_ci_low"], rows["bootstrap_ci_high"] - means]
        )
        axis.bar(
            x + (offset - 1.5) * width,
            means,
            width,
            yerr=errors,
            capsize=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean average precision")
    axis.set_title("CIPHER baseline retrieval with query-bootstrap intervals")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=4, loc="upper right")
    axis.text(
        0,
        -0.22,
        "Intervals resample 11 queries and are descriptive, not population-level validation.",
        transform=axis.transAxes,
        fontsize=8,
    )
    _save(figure, output)


def _query_heatmap(analysis: Path, output: Path) -> None:
    queries = pd.read_parquet(analysis / "query_summary.parquet").sort_values(
        "mean_cosine_ap_across_models", ascending=False
    )
    matrix = queries[[f"{model}_cosine_ap" for model in MODEL_ORDER]].to_numpy()
    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=matrix.max())
    axis.set_xticks(range(4), [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.set_yticks(range(len(queries)), queries["query_id"])
    axis.set_xlabel("Model")
    axis.set_ylabel("ARCH query ID")
    axis.set_title("Cosine average precision is strongly query-dependent")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] < matrix.max() * 0.55 else "black"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color=color,
            )
    figure.colorbar(image, ax=axis, label="Average precision")
    _save(figure, output)


def _agreement_heatmap(analysis: Path, output: Path) -> None:
    agreement = pd.read_parquet(analysis / "model_agreement.parquet")
    agreement = agreement.loc[agreement["top_k"] == 10]
    matrix = np.eye(len(MODEL_ORDER), dtype=float)
    for row in agreement.itertuples():
        a = MODEL_ORDER.index(row.model_a)
        b = MODEL_ORDER.index(row.model_b)
        matrix[a, b] = matrix[b, a] = row.mean_jaccard
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    axis.set_xticks(range(4), labels)
    axis.set_yticks(range(4), labels)
    axis.set_title("Cross-model overlap of cosine top-10 retrievals")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Mean query-level Jaccard similarity")
    _save(figure, output)


def _norm_effect_figure(analysis: Path, output: Path) -> None:
    correlations = pd.read_parquet(analysis / "norm_rank_correlations.parquet")
    grouped = correlations.groupby("model")[
        "spearman_retrieval_rank_vs_gallery_norm"
    ].agg(["mean", "std"])
    grouped = grouped.loc[MODEL_ORDER]
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.bar(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        grouped["mean"],
        yerr=grouped["std"],
        capsize=3,
        color=[MODEL_COLORS[model] for model in MODEL_ORDER],
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Spearman correlation: retrieval rank vs gallery norm")
    axis.set_title("Raw inner product is sensitive to embedding magnitude")
    axis.text(
        0,
        -0.22,
        "Negative values mean higher-norm gallery vectors tend to appear earlier.",
        transform=axis.transAxes,
        fontsize=8,
    )
    _save(figure, output)


def _retrieval_panel(
    context: ConfigContext,
    benchmark: dict[str, Any],
    query_id: str,
    output: Path,
) -> None:
    images = pd.read_parquet(
        context.resolve(context.project.paths.manifests_root) / "images.parquet"
    )
    query_row = images.loc[(images["role"] == "query") & (images["query_id"] == query_id)].iloc[0]
    query_image = Image.open(context.root / query_row["relative_path"]).convert("RGB")
    figure, axes = plt.subplots(4, 6, figsize=(13.5, 8.8))
    set_id = benchmark["identity"]["embedding_set_id"]
    for row_index, model in enumerate(MODEL_ORDER):
        axes[row_index, 0].imshow(query_image)
        axes[row_index, 0].set_title(f"{MODEL_LABELS[model]}\nquery {query_id}", fontsize=9)
        axes[row_index, 0].axis("off")
        run_id = benchmark["retrieval_runs"][model]["cosine"]
        ranking_path = (
            context.resolve(context.project.paths.artifacts_root)
            / "retrieval"
            / set_id
            / model
            / "cosine"
            / run_id
            / "rankings.parquet"
        )
        ranking = pd.read_parquet(ranking_path)
        top = ranking.loc[ranking["query_id"] == query_id].nsmallest(5, "rank")
        for column, result in enumerate(top.itertuples(), start=1):
            with Image.open(context.root / result.gallery_relative_path) as image:
                axes[row_index, column].imshow(image.convert("RGB"))
            status = "curated +" if result.is_relevant else "unjudged"
            axes[row_index, column].set_title(f"rank {result.rank}: {status}", fontsize=8)
            for spine in axes[row_index, column].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)
                spine.set_edgecolor("#2A9D8F" if result.is_relevant else "#C8CDD5")
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
    figure.suptitle(
        f"Cosine retrieval panel for ARCH query {query_id}\n"
        "Green = curated positive; gray = unjudged (not a confirmed biological negative)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _save(figure, output)


def make_figures(
    context: ConfigContext,
    *,
    analysis_id: str,
    benchmark_id: str,
) -> dict[str, Any]:
    analysis = _analysis_directory(context, analysis_id)
    benchmark_path = (
        context.resolve(context.project.paths.artifacts_root)
        / "benchmarks"
        / benchmark_id
        / "manifest.json"
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    output = context.root / "reports" / analysis_id / "figures"
    paths = {
        "baseline_map": output / "baseline_map.png",
        "query_ap": output / "query_ap_heatmap.png",
        "model_agreement": output / "model_agreement_top10.png",
        "norm_effect": output / "norm_effect.png",
    }
    _map_figure(analysis, paths["baseline_map"])
    _query_heatmap(analysis, paths["query_ap"])
    _agreement_heatmap(analysis, paths["model_agreement"])
    _norm_effect_figure(analysis, paths["norm_effect"])
    queries = pd.read_parquet(analysis / "query_summary.parquet").sort_values(
        "mean_cosine_ap_across_models"
    )
    selected = [queries.iloc[0], queries.iloc[len(queries) // 2], queries.iloc[-1]]
    for query in selected:
        key = f"retrieval_panel_{query.query_id}"
        paths[key] = output / f"retrieval_panel_{query.query_id}.png"
        _retrieval_panel(context, benchmark, str(query.query_id), paths[key])
    manifest = {
        "analysis_id": analysis_id,
        "benchmark_id": benchmark_id,
        "figures": {
            name: {
                "path": str(path.relative_to(context.root)),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    atomic_write_json(output.parent / "figures.json", manifest)
    return {
        "status": "generated",
        "analysis_id": analysis_id,
        "figure_count": len(paths),
        "output_directory": str(output.relative_to(context.root)),
        "figures": {name: str(path.relative_to(context.root)) for name, path in paths.items()},
    }

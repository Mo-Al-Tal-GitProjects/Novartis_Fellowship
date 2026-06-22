from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from cipher import __version__
from cipher.analysis.run import load_benchmark
from cipher.config import ConfigContext
from cipher.embeddings.generate import _canonical_digest
from cipher.embeddings.store import atomic_write_parquet
from cipher.provenance import (
    atomic_write_json,
    runtime_provenance,
    sha256_file,
    sha256_python_tree,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

MODEL_ORDER = ["clip", "dino", "plip", "uni"]
VARIANT_ORDER = ["model_native", "grayscale", "center_square"]
VARIANT_LABELS = {
    "model_native": "Model-native",
    "grayscale": "Grayscale",
    "center_square": "Center-square crop",
}


def compare_preprocessing(
    context: ConfigContext,
    *,
    baseline_benchmark_id: str,
    grayscale_benchmark_id: str,
    center_square_benchmark_id: str,
    bootstrap_iterations: int = 10_000,
) -> dict[str, Any]:
    benchmark_ids = {
        "model_native": baseline_benchmark_id,
        "grayscale": grayscale_benchmark_id,
        "center_square": center_square_benchmark_id,
    }
    benchmarks = {
        variant: load_benchmark(context, benchmark_id)
        for variant, benchmark_id in benchmark_ids.items()
    }
    identity = {
        "schema_version": 1,
        "cipher_version": __version__,
        "cipher_source_sha256": sha256_python_tree(context.root / "src" / "cipher"),
        "benchmarks": {
            variant: {
                "benchmark_id": manifest["benchmark_id"],
                "manifest_sha256": sha256_file(path),
            }
            for variant, (path, manifest) in benchmarks.items()
        },
        "metric": "cosine",
        "bootstrap_iterations": bootstrap_iterations,
        "seed": context.project.seed,
    }
    identity_sha256 = _canonical_digest(identity)
    ablation_id = identity_sha256[:16]
    directory = context.resolve(context.project.paths.artifacts_root) / "ablations" / ablation_id
    query_ids: list[str] | None = None
    values: dict[tuple[str, str], np.ndarray] = {}
    frames = {}
    for variant, (path, _) in benchmarks.items():
        frame = pd.read_parquet(path.parent / "per_query.parquet")
        frame = frame.loc[frame["metric"] == "cosine"]
        frames[variant] = frame
        current_ids = sorted(frame["query_id"].astype(str).unique(), key=int)
        if query_ids is None:
            query_ids = current_ids
        elif current_ids != query_ids:
            raise ValueError("preprocessing benchmarks contain different query IDs")
        for model in MODEL_ORDER:
            values[(variant, model)] = (
                frame.loc[frame["model"] == model]
                .set_index("query_id")
                .loc[query_ids, "average_precision"]
                .to_numpy(dtype=np.float64)
            )
    assert query_ids is not None
    indices = np.random.default_rng(context.project.seed).integers(
        0, len(query_ids), size=(bootstrap_iterations, len(query_ids))
    )
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        baseline = values[("model_native", model)]
        for variant in VARIANT_ORDER:
            observed = values[(variant, model)]
            samples = observed[indices].mean(axis=1)
            delta = observed - baseline
            delta_samples = delta[indices].mean(axis=1)
            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "map": float(observed.mean()),
                    "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                    "mean_ap_difference_vs_native": float(delta.mean()),
                    "difference_ci_low": float(np.quantile(delta_samples, 0.025)),
                    "difference_ci_high": float(np.quantile(delta_samples, 0.975)),
                    "queries_improved": int((delta > 0).sum()),
                    "queries_unchanged": int((delta == 0).sum()),
                    "queries_worsened": int((delta < 0).sum()),
                }
            )
    results = pd.DataFrame(rows)
    directory.mkdir(parents=True, exist_ok=True)
    results_path = directory / "preprocessing_results.parquet"
    atomic_write_parquet(results_path, results)
    summary_path = directory / "summary.json"
    atomic_write_json(
        summary_path,
        {
            "ablation_id": ablation_id,
            "benchmark_ids": benchmark_ids,
            "rows": json.loads(results.to_json(orient="records")),
            "interpretation_boundary": (
                "Grayscale probes color dependence and center-square crop probes framing; "
                "neither isolates a single biological factor."
            ),
        },
    )

    report_root = context.root / "reports" / ablation_id
    report_root.mkdir(parents=True, exist_ok=True)
    figure_path = report_root / "preprocessing_ablation.png"
    figure, axis = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(MODEL_ORDER), dtype=float)
    width = 0.25
    colors = ["#4472C4", "#A0A6B0", "#D98E73"]
    for offset, variant in enumerate(VARIANT_ORDER):
        subset = results.loc[results["variant"] == variant].set_index("model").loc[MODEL_ORDER]
        axis.bar(
            x + (offset - 1) * width,
            subset["map"],
            width,
            label=VARIANT_LABELS[variant],
            color=colors[offset],
        )
    axis.set_xticks(x, [model.upper() for model in MODEL_ORDER])
    axis.set_ylabel("Cosine mean average precision")
    axis.set_title("Preprocessing sensitivity on 11 curated queries")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    axis.text(
        0,
        -0.2,
        "Grayscale removes hue; center-square crop changes field of view "
        "before model-native processing.",
        transform=axis.transAxes,
        fontsize=8,
    )
    figure.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "ablation_id": ablation_id,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "file_sha256": {
            "preprocessing_results.parquet": sha256_file(results_path),
            "summary.json": sha256_file(summary_path),
        },
        "figure": {
            "path": str(figure_path.relative_to(context.root)),
            "sha256": sha256_file(figure_path),
        },
        "runtime": runtime_provenance(),
    }
    atomic_write_json(directory / "manifest.json", manifest)
    return {
        "status": "generated",
        "ablation_id": ablation_id,
        "artifact_directory": str(directory.relative_to(context.root)),
        "figure": str(figure_path.relative_to(context.root)),
        "results": json.loads(results.to_json(orient="records")),
    }

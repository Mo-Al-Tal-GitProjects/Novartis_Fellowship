from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from cipher.config import ConfigContext
from cipher.provenance import atomic_write_json, sha256_file


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_analysis_report(
    context: ConfigContext,
    *,
    analysis_id: str,
    benchmark_id: str,
) -> dict[str, Any]:
    analysis = context.resolve(context.project.paths.artifacts_root) / "analyses" / analysis_id
    report_root = context.root / "reports" / analysis_id
    figures_manifest = json.loads(
        (report_root / "figures.json").read_text(encoding="utf-8")
    )
    uncertainty = pd.read_parquet(analysis / "uncertainty.parquet")
    pairwise = pd.read_parquet(analysis / "pairwise_differences.parquet")
    agreement = pd.read_parquet(analysis / "model_agreement.parquet")
    sensitivity = pd.read_parquet(analysis / "metric_sensitivity.parquet")
    norm_correlations = pd.read_parquet(analysis / "norm_rank_correlations.parquet")
    queries = pd.read_parquet(analysis / "query_summary.parquet")

    cosine = uncertainty.loc[uncertainty["metric"] == "cosine"].sort_values(
        "map", ascending=False
    )
    cosine_table = _markdown_table(
        ["Model", "MAP", "95% query-bootstrap interval"],
        [
            [
                str(row.model).upper(),
                f"{row.map:.4f}",
                f"[{row.bootstrap_ci_low:.4f}, {row.bootstrap_ci_high:.4f}]",
            ]
            for row in cosine.itertuples()
        ],
    )
    cosine_pairs = pairwise.loc[pairwise["metric"] == "cosine"]
    pair_table = _markdown_table(
        ["Comparison", "Mean AP difference", "95% interval", "P(A > B)"],
        [
            [
                f"{row.model_a.upper()} − {row.model_b.upper()}",
                f"{row.mean_ap_difference_a_minus_b:.4f}",
                f"[{row.bootstrap_ci_low:.4f}, {row.bootstrap_ci_high:.4f}]",
                f"{row.bootstrap_probability_a_greater_b:.3f}",
            ]
            for row in cosine_pairs.itertuples()
        ],
    )
    agreement_10 = agreement.loc[agreement["top_k"] == 10].sort_values(
        "mean_jaccard", ascending=False
    )
    agreement_table = _markdown_table(
        ["Model pair", "Mean top-10 Jaccard"],
        [
            [f"{row.model_a.upper()} / {row.model_b.upper()}", f"{row.mean_jaccard:.3f}"]
            for row in agreement_10.itertuples()
        ],
    )
    mean_norm_correlations = (
        norm_correlations.groupby("model")[
            "spearman_retrieval_rank_vs_gallery_norm"
        ]
        .mean()
        .sort_values()
    )
    norm_table = _markdown_table(
        ["Model", "Mean rank–norm Spearman correlation"],
        [[model.upper(), f"{value:.3f}"] for model, value in mean_norm_correlations.items()],
    )
    sensitivity_table = _markdown_table(
        ["Model", "Alternative", "Mean AP change", "Improved / worsened queries"],
        [
            [
                row.model.upper(),
                row.alternative_metric,
                f"{row.mean_ap_difference:+.4f}",
                f"{row.queries_improved} / {row.queries_worsened}",
            ]
            for row in sensitivity.itertuples()
        ],
    )
    diagnostic_cutoff = int(queries["diagnostic_cutoff"].iloc[0])
    query_table = _markdown_table(
        [
            "Query",
            "Mean cosine AP",
            "Best model",
            f"Models with hit@{diagnostic_cutoff}",
        ],
        [
            [
                str(row.query_id),
                f"{row.mean_cosine_ap_across_models:.4f}",
                row.best_model.upper(),
                str(row.models_with_hit_at_cutoff),
            ]
            for row in queries.itertuples()
        ],
    )
    figure_paths = {
        name: Path(payload["path"]).relative_to(report_root.relative_to(context.root))
        for name, payload in figures_manifest["figures"].items()
    }
    hard_panel = figure_paths[f"retrieval_panel_{queries.iloc[0].query_id}"]
    median_panel = figure_paths[
        f"retrieval_panel_{queries.iloc[len(queries) // 2].query_id}"
    ]
    easy_panel = figure_paths[f"retrieval_panel_{queries.iloc[-1].query_id}"]
    content = f"""# CIPHER retrieval analysis report

This report records new CIPHER reconstruction analyses. It is not an official result
of the Fall 2024 challenge project and does not imply institutional or advisor
endorsement.

- Benchmark: `{benchmark_id}`
- Analysis: `{analysis_id}`
- Queries: 11
- Gallery images: 3,321
- Curated binary positives: 49

## Uncertainty-aware baseline

{cosine_table}

![MAP with bootstrap intervals]({figure_paths['baseline_map'].as_posix()})

Intervals resample the 11 queries 10,000 times. They quantify sensitivity to this
small query set; they are not confidence intervals for clinical deployment or a
larger histopathology population.

## Paired model differences under cosine retrieval

{pair_table}

Wide intervals and query-level reversals prevent a definitive model hierarchy.

## Query difficulty

{query_table}

![Per-query cosine AP]({figure_paths['query_ap'].as_posix()})

The easiest consensus query is {queries.iloc[-1].query_id}; the hardest is
{queries.iloc[0].query_id}. Difficulty is not interchangeable with biological
complexity because stain, morphology, tissue, and magnification have not yet been
expert-adjudicated as explanatory labels.

## Cross-model agreement

{agreement_table}

![Top-10 agreement]({figure_paths['model_agreement'].as_posix()})

Low overlap means models often obtain similar aggregate scores using different
neighbors. Agreement is descriptive and does not determine which unjudged neighbors
are biologically appropriate.

## Embedding norm and metric sensitivity

{norm_table}

![Norm effect]({figure_paths['norm_effect'].as_posix()})

{sensitivity_table}

The rank–norm correlation shows how raw inner product favors or suppresses vectors
according to magnitude. It does not identify the visual or biological property
encoded by the norm.

## Qualitative panels

![Hard query]({hard_panel.as_posix()})

![Median-difficulty query]({median_panel.as_posix()})

![Easy query]({easy_panel.as_posix()})

Green borders denote curated positives. Gray borders denote unjudged images, not
confirmed false biological matches.

## Interpretation boundary

The analysis supports query-sensitive differences, metric sensitivity, model
disagreement, and long-tailed failure behavior. It cannot yet causally attribute a
retrieval to tissue, disease, stain, morphology, magnification, texture, or artifact.
The optional annotation schema is ready, but no explanatory annotations were
invented or inferred automatically.
"""
    report_path = report_root / "stage_4c_6_report.md"
    report_path.write_text(content, encoding="utf-8")
    return {
        "status": "generated",
        "analysis_id": analysis_id,
        "report": str(report_path.relative_to(context.root)),
        "sha256": sha256_file(report_path),
    }


def build_paper_bundle(context: ConfigContext, *, analysis_id: str) -> dict[str, Any]:
    manuscript = context.root / "paper" / "unofficial_manuscript.md"
    if not manuscript.is_file():
        raise FileNotFoundError("paper/unofficial_manuscript.md not found")
    text = manuscript.read_text(encoding="utf-8")
    required_sections = [
        "# CIPHER",
        "## Abstract",
        "## Dataset",
        "## Methods",
        "## Results",
        "## Limitations",
        "## Reproducibility statement",
        "## References",
    ]
    missing = [section for section in required_sections if section not in text]
    if missing:
        raise ValueError(f"manuscript is missing required sections: {missing}")
    lowered = text.lower()
    if "unofficial" not in lowered or "does not imply endorsement" not in lowered:
        raise ValueError("manuscript must retain the unofficial non-endorsement statement")
    output = context.root / "reports" / analysis_id / "paper"
    output.mkdir(parents=True, exist_ok=True)
    rendered = output / "unofficial_manuscript.md"
    shutil.copyfile(manuscript, rendered)
    manifest = {
        "analysis_id": analysis_id,
        "source": str(manuscript.relative_to(context.root)),
        "source_sha256": sha256_file(manuscript),
        "built_manuscript": str(rendered.relative_to(context.root)),
        "built_sha256": sha256_file(rendered),
        "format": "markdown",
    }
    atomic_write_json(output / "build.json", manifest)
    return {"status": "generated", **manifest}

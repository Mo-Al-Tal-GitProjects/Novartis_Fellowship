from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer

from cipher import __version__
from cipher.analysis.ablations import compare_preprocessing
from cipher.analysis.figures import make_figures as generate_figures
from cipher.analysis.report import build_analysis_report, build_paper_bundle
from cipher.analysis.run import run_analysis
from cipher.config import (
    ConfigContext,
    load_context,
    load_model_spec,
    load_retrieval_config,
)
from cipher.data.download import download_archive, extract_pubmed_archive, import_matches
from cipher.data.ground_truth import build_relevance_manifest, validate_ground_truth
from cipher.data.manifest import build_image_manifest, inspect_prepared_data
from cipher.embeddings.generate import generate_embeddings
from cipher.embeddings.set import DEFAULT_MODEL_NAMES, assemble_embedding_set
from cipher.evaluation.benchmark import run_benchmark
from cipher.models.access import check_model_access
from cipher.models.device import resolve_device
from cipher.provenance import atomic_write_json
from cipher.retrieval.run import run_retrieval

app = typer.Typer(
    name="cipher",
    help="CIPHER histopathology embedding retrieval research workflow.",
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_enable=False,
)
data_app = typer.Typer(help="Prepare and inspect research data.", no_args_is_help=True)
ground_truth_app = typer.Typer(help="Validate curated retrieval judgments.", no_args_is_help=True)
models_app = typer.Typer(
    help="Inspect pinned embedding model configurations.", no_args_is_help=True
)
embeddings_app = typer.Typer(help="Generate and cache image embeddings.", no_args_is_help=True)
retrieval_app = typer.Typer(help="Run exact deterministic image retrieval.", no_args_is_help=True)
evaluation_app = typer.Typer(
    help="Evaluate retrieval against curated judgments.", no_args_is_help=True
)
analysis_app = typer.Typer(help="Analyze benchmark behavior and uncertainty.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(ground_truth_app, name="ground-truth")
app.add_typer(models_app, name="models")
app.add_typer(embeddings_app, name="embeddings")
app.add_typer(retrieval_app, name="retrieval")
app.add_typer(evaluation_app, name="evaluate")
app.add_typer(analysis_app, name="analysis")

ConfigOption = Annotated[
    Path,
    typer.Option("--config", help="Project configuration YAML."),
]
PolicyOption = Annotated[
    Path,
    typer.Option("--policy", help="Ground-truth policy YAML."),
]


def _context(config: Path, policy: Path) -> ConfigContext:
    try:
        return load_context(config, policy)
    except Exception as error:
        raise typer.BadParameter(str(error)) from None


def _print(payload: dict) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _fail(error: Exception) -> NoReturn:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the CIPHER version and exit.", is_eager=True),
    ] = False,
) -> None:
    if version:
        typer.echo(f"cipher {__version__}")
        raise typer.Exit()


@data_app.command("prepare")
def prepare_data(
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
    matches_source: Annotated[
        Path,
        typer.Option(help="Directory containing the supplied curated match images."),
    ] = Path("../matches"),
    archive: Annotated[
        Path | None,
        typer.Option(help="Use an existing ARCH PubMed ZIP instead of the configured archive."),
    ] = None,
    no_download: Annotated[
        bool,
        typer.Option(help="Require a local archive and disable network download."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(help="Refresh archive extraction and local match copies."),
    ] = False,
) -> None:
    """Prepare ARCH, import curated matches, and build validated manifests."""
    context = _context(config, policy)
    archive_path = archive or context.project.paths.archive
    archive_path = archive_path if archive_path.is_absolute() else context.root / archive_path
    if not matches_source.is_absolute():
        matches_source = (context.root / matches_source).resolve()
    arch_root = context.resolve(context.project.paths.arch_root)
    matches_root = context.resolve(context.project.paths.matches_root)
    manifests_root = context.resolve(context.project.paths.manifests_root)

    try:
        if no_download:
            if not archive_path.is_file():
                raise FileNotFoundError(f"ARCH archive not found: {archive_path}")
            download = {
                "path": str(archive_path),
                "downloaded": False,
                "bytes": archive_path.stat().st_size,
            }
        else:
            download = download_archive(
                context.project.arch.url,
                archive_path,
                force=force,
                workers=context.project.arch.download_workers,
            )
        extracted_files = extract_pubmed_archive(archive_path, arch_root, force=force)
        imported_matches = import_matches(
            matches_source,
            matches_root,
            context.project.validation.allowed_extensions,
            force=force,
        )
        images, image_summary = build_image_manifest(context)
        _, relevance_summary = build_relevance_manifest(context, images)
        validation = validate_ground_truth(context)
    except Exception as error:
        _fail(error)

    preparation = {
        "archive": download,
        "arch_extracted_files": extracted_files,
        "curated_match_files": imported_matches,
        "images": image_summary,
        "relevance": relevance_summary,
        "validation_status": validation["status"],
    }
    atomic_write_json(manifests_root / "preparation.json", preparation)
    _print(preparation)


@data_app.command("inspect")
def inspect_data(
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Summarize prepared data without modifying it."""
    context = _context(config, policy)
    try:
        report = inspect_prepared_data(context)
    except Exception as error:
        _fail(error)
    _print(report)


@ground_truth_app.command("validate")
def validate_curated_ground_truth(
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Validate query mappings, relevance judgments, duplicates, and split safety."""
    context = _context(config, policy)
    try:
        report = validate_ground_truth(context)
    except Exception as error:
        _fail(error)
    _print(report)


@models_app.command("inspect")
def inspect_model(
    model: Annotated[str, typer.Option(help="Pinned model configuration name.")] = "clip",
    device: Annotated[
        Literal["auto", "cpu", "cuda", "mps"],
        typer.Option(help="Inference device request."),
    ] = "auto",
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Inspect a model pin and resolve its runtime device without loading weights."""
    context = _context(config, policy)
    try:
        path, spec = load_model_spec(model, root=context.root)
        selected_device = resolve_device(device)
    except Exception as error:
        _fail(error)
    _print(
        {
            "config": str(path.relative_to(context.root)),
            "model": spec.model_dump(mode="json"),
            "resolved_device": selected_device,
            "weights_loaded": False,
        }
    )


@models_app.command("access")
def inspect_model_access(
    model: Annotated[str, typer.Option(help="Pinned model configuration name.")] = "uni",
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Check local authentication and gated-file access without exposing credentials."""
    context = _context(config, policy)
    try:
        _, spec = load_model_spec(model, root=context.root)
        report = check_model_access(spec)
    except Exception as error:
        _fail(error)
    _print(report)
    if not report["model_accessible"]:
        raise typer.Exit(code=1)


@embeddings_app.command("generate")
def generate_model_embeddings(
    model: Annotated[str, typer.Option(help="Pinned model configuration name.")] = "clip",
    role: Annotated[
        Literal["query", "gallery", "all"],
        typer.Option(help="Manifest role to embed; all means active query and gallery rows."),
    ] = "query",
    device: Annotated[
        Literal["auto", "cpu", "cuda", "mps"],
        typer.Option(help="Inference device request."),
    ] = "auto",
    batch_size: Annotated[
        int | None,
        typer.Option(min=1, help="Override the model configuration batch size."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Embed only the first N deterministic rows (smoke tests)."),
    ] = None,
    preprocessing_variant: Annotated[
        Literal["model_native", "grayscale", "center_square"],
        typer.Option(help="Controlled input preprocessing ablation."),
    ] = "model_native",
    force: Annotated[
        bool,
        typer.Option(help="Regenerate even if the verified content-addressed artifact exists."),
    ] = False,
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Generate raw and L2-normalized embeddings with full provenance."""
    context = _context(config, policy)
    try:
        _, spec = load_model_spec(model, root=context.root)
        selected_device = resolve_device(device)
        report = generate_embeddings(
            context,
            spec,
            role=role,
            device=selected_device,
            batch_size=batch_size,
            limit=limit,
            force=force,
            preprocessing_variant=preprocessing_variant,
        )
    except Exception as error:
        _fail(error)
    _print(report)


@embeddings_app.command("assemble")
def assemble_model_embeddings(
    models: Annotated[
        str,
        typer.Option(help="Comma-separated model names that must have verified artifacts."),
    ] = ",".join(DEFAULT_MODEL_NAMES),
    role: Annotated[
        Literal["query", "gallery", "all"],
        typer.Option(help="Previously generated manifest role to assemble."),
    ] = "all",
    device: Annotated[
        Literal["auto", "cpu", "cuda", "mps"],
        typer.Option(help="Device used by the expected embedding artifacts."),
    ] = "auto",
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Match a limited smoke artifact selection."),
    ] = None,
    preprocessing_variant: Annotated[
        Literal["model_native", "grayscale", "center_square"],
        typer.Option(help="Preprocessing variant used by the artifacts."),
    ] = "model_native",
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Verify row alignment and assemble model artifacts into one provenance set."""
    context = _context(config, policy)
    names = [name.strip() for name in models.split(",") if name.strip()]
    try:
        selected_device = resolve_device(device)
        report = assemble_embedding_set(
            context,
            model_names=names,
            role=role,
            device=selected_device,
            limit=limit,
            preprocessing_variant=preprocessing_variant,
        )
    except Exception as error:
        _fail(error)
    _print(report)


@retrieval_app.command("run")
def run_exact_retrieval(
    model: Annotated[str, typer.Option(help="Model in the aligned embedding set.")] = "clip",
    metric: Annotated[
        Literal["cosine", "inner_product", "euclidean", "euclidean_l2"],
        typer.Option(help="Exact ranking contract."),
    ] = "cosine",
    embedding_set: Annotated[
        str | None,
        typer.Option(help="Embedding-set ID; inferred only when exactly one set exists."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Regenerate the retrieval artifact.")] = False,
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Rank the complete gallery for every benchmark query."""
    context = _context(config, policy)
    try:
        report = run_retrieval(
            context,
            model_name=model,
            metric_name=metric,
            embedding_set_id=embedding_set,
            force=force,
        )
    except Exception as error:
        _fail(error)
    _print(report)


@evaluation_app.command("benchmark")
def evaluate_benchmark(
    experiment_config: Annotated[
        Path,
        typer.Option(
            "--experiment-config",
            help="Retrieval metrics, top-k values, and model selection YAML.",
        ),
    ] = Path("configs/retrieval/base.yaml"),
    embedding_set: Annotated[
        str | None,
        typer.Option(help="Embedding-set ID; inferred only when exactly one set exists."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(help="Regenerate retrieval and benchmark outputs."),
    ] = False,
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Run and score every configured model/metric retrieval combination."""
    context = _context(config, policy)
    try:
        _, retrieval_config = load_retrieval_config(experiment_config, root=context.root)
        report = run_benchmark(
            context,
            retrieval_config,
            embedding_set_id=embedding_set,
            force=force,
        )
    except Exception as error:
        _fail(error)
    _print(report)


@analysis_app.command("run")
def analyze_benchmark(
    benchmark_id: Annotated[
        str,
        typer.Option(help="Verified benchmark ID to analyze."),
    ],
    bootstrap_iterations: Annotated[
        int,
        typer.Option(min=100, help="Paired query-bootstrap resamples."),
    ] = 10_000,
    force: Annotated[bool, typer.Option(help="Regenerate a verified analysis artifact.")] = False,
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Run uncertainty, agreement, query, norm, and metric diagnostics."""
    context = _context(config, policy)
    try:
        report = run_analysis(
            context,
            benchmark_id=benchmark_id,
            bootstrap_iterations=bootstrap_iterations,
            force=force,
        )
    except Exception as error:
        _fail(error)
    _print(report)


@analysis_app.command("compare-preprocessing")
def analyze_preprocessing(
    baseline_benchmark: Annotated[str, typer.Option(help="Model-native benchmark ID.")],
    grayscale_benchmark: Annotated[str, typer.Option(help="Grayscale benchmark ID.")],
    center_square_benchmark: Annotated[
        str,
        typer.Option(help="Center-square-crop benchmark ID."),
    ],
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Compare controlled preprocessing variants with paired query uncertainty."""
    context = _context(config, policy)
    try:
        report = compare_preprocessing(
            context,
            baseline_benchmark_id=baseline_benchmark,
            grayscale_benchmark_id=grayscale_benchmark,
            center_square_benchmark_id=center_square_benchmark,
        )
    except Exception as error:
        _fail(error)
    _print(report)


@app.command("make-figures")
def make_analysis_figures(
    analysis_id: Annotated[str, typer.Option(help="Verified analysis artifact ID.")],
    benchmark_id: Annotated[str, typer.Option(help="Benchmark used by the analysis.")],
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Generate scientific figures, qualitative panels, and an analysis report."""
    context = _context(config, policy)
    try:
        figures = generate_figures(
            context,
            analysis_id=analysis_id,
            benchmark_id=benchmark_id,
        )
        report = build_analysis_report(
            context,
            analysis_id=analysis_id,
            benchmark_id=benchmark_id,
        )
    except Exception as error:
        _fail(error)
    _print({"figures": figures, "report": report})


@app.command("build-paper")
def build_paper(
    analysis_id: Annotated[str, typer.Option(help="Analysis ID cited by the manuscript.")],
    config: ConfigOption = Path("configs/base.yaml"),
    policy: PolicyOption = Path("configs/ground_truth.yaml"),
) -> None:
    """Validate and bundle the unofficial Markdown manuscript."""
    context = _context(config, policy)
    try:
        report = build_paper_bundle(context, analysis_id=analysis_id)
    except Exception as error:
        _fail(error)
    _print(report)

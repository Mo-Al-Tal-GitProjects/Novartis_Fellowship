import json

import numpy as np
from typer.testing import CliRunner

from cipher.cli import app
from cipher.config import load_model_spec
from cipher.data.ground_truth import build_relevance_manifest
from cipher.data.manifest import build_image_manifest
from cipher.embeddings.generate import generate_embeddings
from cipher.embeddings.set import assemble_embedding_set

runner = CliRunner()


class DimensionAwareFakeEncoder:
    def __init__(self, spec, device: str) -> None:
        self.spec = spec
        self.device = device

    def encode(self, images) -> np.ndarray:
        rows = []
        for image in images:
            pixels = np.asarray(image, dtype=np.float32)
            channel_means = pixels.mean(axis=(0, 1))
            values = np.resize(channel_means, self.spec.expected_dimension)
            rows.append(values + np.arange(self.spec.expected_dimension, dtype=np.float32))
        return np.asarray(rows, dtype=np.float32)


def test_version_command() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "cipher 0.1.0"


def test_data_inspect_before_manifest(project_fixture, monkeypatch) -> None:
    monkeypatch.chdir(project_fixture.root)
    result = runner.invoke(app, ["data", "inspect"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["prepared"] is False
    assert payload["arch_images_on_disk"] == 3
    assert payload["match_images_on_disk"] == 4


def test_ground_truth_validate_after_manifest(project_fixture, monkeypatch) -> None:
    from cipher.data.ground_truth import build_relevance_manifest
    from cipher.data.manifest import build_image_manifest

    images, _ = build_image_manifest(project_fixture)
    build_relevance_manifest(project_fixture, images)
    monkeypatch.chdir(project_fixture.root)
    result = runner.invoke(app, ["ground-truth", "validate"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["positive_images"] == 3


def test_models_inspect_does_not_load_weights(project_fixture, monkeypatch) -> None:
    monkeypatch.chdir(project_fixture.root)
    monkeypatch.setattr("cipher.cli.resolve_device", lambda request: "cpu")
    result = runner.invoke(app, ["models", "inspect", "--model", "clip"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["model"]["revision"] == "a" * 40
    assert payload["model"]["expected_dimension"] == 4
    assert payload["resolved_device"] == "cpu"
    assert payload["weights_loaded"] is False


def test_models_access_prints_only_sanitized_report(project_fixture, monkeypatch) -> None:
    monkeypatch.chdir(project_fixture.root)
    monkeypatch.setattr(
        "cipher.cli.check_model_access",
        lambda spec: {
            "model": spec.name,
            "gated": spec.gated,
            "credential_present": True,
            "credential_source": "local_store",
            "authentication_valid": True,
            "model_accessible": True,
            "status": "accessible",
        },
    )
    result = runner.invoke(app, ["models", "access", "--model", "uni"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "accessible"
    assert payload["model"] == "uni"
    assert "token" not in payload


def test_models_access_exits_nonzero_when_unavailable(project_fixture, monkeypatch) -> None:
    monkeypatch.chdir(project_fixture.root)
    monkeypatch.setattr(
        "cipher.cli.check_model_access",
        lambda spec: {
            "model": spec.name,
            "model_accessible": False,
            "status": "invalid_token",
        },
    )
    result = runner.invoke(app, ["models", "access", "--model", "uni"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "invalid_token"


def test_expected_model_failure_has_no_traceback(project_fixture, monkeypatch) -> None:
    monkeypatch.chdir(project_fixture.root)
    monkeypatch.setattr("cipher.cli.resolve_device", lambda request: "cpu")

    def fail_cleanly(*args, **kwargs):
        raise RuntimeError("gated model login required")

    monkeypatch.setattr("cipher.cli.generate_embeddings", fail_cleanly)
    result = runner.invoke(app, ["embeddings", "generate", "--model", "uni"])
    assert result.exit_code == 1
    assert result.stderr.strip() == "Error: gated model login required"
    assert "Traceback" not in result.output


def test_benchmark_cli_scores_aligned_embedding_set(project_fixture, monkeypatch) -> None:
    images, _ = build_image_manifest(project_fixture)
    build_relevance_manifest(project_fixture, images)
    model_names = ["clip", "dino", "plip", "uni"]
    for model_name in model_names:
        _, spec = load_model_spec(model_name, root=project_fixture.root)
        generate_embeddings(
            project_fixture,
            spec,
            role="all",
            device="cpu",
            encoder_factory=DimensionAwareFakeEncoder,
        )
    embedding_set = assemble_embedding_set(
        project_fixture,
        model_names=model_names,
        role="all",
        device="cpu",
    )

    monkeypatch.chdir(project_fixture.root)
    result = runner.invoke(
        app,
        ["evaluate", "benchmark", "--embedding-set", embedding_set["set_id"]],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "generated"
    assert len(payload["primary_results"]) == 16
    assert set(payload["retrieval_runs"]) == set(model_names)
    assert (project_fixture.root / payload["summary"]).is_file()

    analysis = runner.invoke(
        app,
        [
            "analysis",
            "run",
            "--benchmark-id",
            payload["benchmark_id"],
            "--bootstrap-iterations",
            "200",
        ],
    )
    assert analysis.exit_code == 0, analysis.output
    analysis_payload = json.loads(analysis.stdout)
    assert analysis_payload["status"] == "generated"
    assert analysis_payload["benchmark_id"] == payload["benchmark_id"]
    assert (
        project_fixture.root
        / analysis_payload["artifact_directory"]
        / "pairwise_differences.parquet"
    ).is_file()

    cached = runner.invoke(
        app,
        ["evaluate", "benchmark", "--embedding-set", embedding_set["set_id"]],
    )
    assert cached.exit_code == 0, cached.output
    assert json.loads(cached.stdout)["status"] == "cache_hit"

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path
    archive: Path
    arch_root: Path
    matches_root: Path
    manifests_root: Path
    artifacts_root: Path


class ArchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    expected_caption_records: int = Field(gt=0)
    download_workers: int = Field(default=4, ge=1, le=16)


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_extensions: list[str]
    compute_perceptual_hash: bool = True
    fail_on_decode_error: bool = True

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        normalized = [
            value.lower() if value.startswith(".") else f".{value.lower()}" for value in values
        ]
        if not normalized:
            raise ValueError("at least one image extension is required")
        return sorted(set(normalized))


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    seed: int
    paths: PathsConfig
    arch: ArchConfig
    validation: ValidationConfig


class ModelSpec(BaseModel):
    """Immutable model and inference settings that define an embedding space."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    adapter: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    expected_dimension: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    dtype: Literal["float32"]
    preprocessing: Literal["model_native"]
    embedding_output: Literal["image_projection", "cls_token"]
    gated: bool = False
    trust_remote_code: bool = False


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    models: list[str] = Field(min_length=1)
    metrics: list[
        Literal["cosine", "inner_product", "euclidean", "euclidean_l2"]
    ] = Field(min_length=1)
    top_k: list[int] = Field(min_length=1)
    query_role: Literal["query"] = "query"
    gallery_role: Literal["gallery"] = "gallery"
    tie_breaker: Literal["gallery_image_id_ascending"]
    relevance: Literal["binary_curated"]

    @field_validator("models", "metrics")
    @classmethod
    def unique_strings(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("values must be unique")
        return values

    @field_validator("top_k")
    @classmethod
    def valid_top_k(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("top-k values must be positive")
        if values != sorted(set(values)):
            raise ValueError("top-k values must be unique and increasing")
        return values


class DeduplicationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: str
    excluded: list[str]
    reason: str


class RetainedAnomaly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[str]
    reason: str


class GroundTruthPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    queries: dict[str, str]
    deduplications: list[DeduplicationDecision] = Field(default_factory=list)
    retained_anomalies: list[RetainedAnomaly] = Field(default_factory=list)
    missing_rank_policy: Literal["use_observed_only"]
    unjudged_gallery_policy: Literal["operational_negative"]
    expected_positive_images: int = Field(default=49, gt=0)

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, values: dict[str, str]) -> dict[str, str]:
        if not values:
            raise ValueError("at least one query mapping is required")
        for query_id, uuid in values.items():
            if not query_id.isdigit():
                raise ValueError(f"query ID must be numeric: {query_id}")
            if len(uuid) != 36:
                raise ValueError(f"query UUID has unexpected length: {uuid}")
        return values


@dataclass(frozen=True)
class ConfigContext:
    root: Path
    config_path: Path
    project: ProjectConfig
    policy_path: Path
    policy: GroundTruthPolicy

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.root / path).resolve()


def find_project_root(start: Path | None = None) -> Path:
    configured = os.getenv("CIPHER_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    cursor = (start or Path.cwd()).resolve()
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    raise FileNotFoundError("could not locate the CIPHER project root")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def load_context(
    config_path: Path | str = Path("configs/base.yaml"),
    policy_path: Path | str = Path("configs/ground_truth.yaml"),
    *,
    root: Path | None = None,
) -> ConfigContext:
    project_root = (root or find_project_root()).resolve()
    config = Path(config_path)
    policy = Path(policy_path)
    config = config if config.is_absolute() else project_root / config
    policy = policy if policy.is_absolute() else project_root / policy
    return ConfigContext(
        root=project_root,
        config_path=config.resolve(),
        project=ProjectConfig.model_validate(_load_yaml(config)),
        policy_path=policy.resolve(),
        policy=GroundTruthPolicy.model_validate(_load_yaml(policy)),
    )


def load_model_spec(name: str, *, root: Path | None = None) -> tuple[Path, ModelSpec]:
    project_root = (root or find_project_root()).resolve()
    path = (project_root / "configs" / "models" / f"{name}.yaml").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model configuration not found: {path}")
    spec = ModelSpec.model_validate(_load_yaml(path))
    if spec.name != name:
        raise ValueError(f"model configuration name {spec.name!r} does not match {name!r}")
    return path, spec


def load_retrieval_config(
    path: Path | str = Path("configs/retrieval/base.yaml"),
    *,
    root: Path | None = None,
) -> tuple[Path, RetrievalConfig]:
    project_root = (root or find_project_root()).resolve()
    config_path = Path(path)
    config_path = (
        config_path if config_path.is_absolute() else (project_root / config_path).resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"retrieval configuration not found: {config_path}")
    return config_path, RetrievalConfig.model_validate(_load_yaml(config_path))

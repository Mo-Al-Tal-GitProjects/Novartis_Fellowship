from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from cipher.config import ConfigContext, load_context


def write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture()
def project_fixture(tmp_path: Path) -> ConfigContext:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    configs = tmp_path / "configs"
    configs.mkdir()
    base = {
        "version": 1,
        "seed": 42,
        "paths": {
            "data_root": "data",
            "archive": "data/external/pubmed_set.zip",
            "arch_root": "data/raw/arch/pubmed_set",
            "matches_root": "data/raw/matches",
            "manifests_root": "data/manifests",
            "artifacts_root": "artifacts",
        },
        "arch": {"url": "https://example.invalid/pubmed.zip", "expected_caption_records": 3},
        "validation": {
            "allowed_extensions": [".jpg", ".jpeg", ".png"],
            "compute_perceptual_hash": True,
            "fail_on_decode_error": True,
        },
    }
    policy = {
        "version": 1,
        "expected_positive_images": 3,
        "queries": {
            "1": "11111111-1111-1111-1111-111111111111",
            "2": "22222222-2222-2222-2222-222222222222",
        },
        "deduplications": [
            {
                "canonical": "1_01.png",
                "excluded": ["1_01.jpg"],
                "reason": "fixture alternate encoding",
            }
        ],
        "retained_anomalies": [
            {
                "files": ["1_02.jpg"],
                "reason": "fixture repeated rank decision",
            }
        ],
        "missing_rank_policy": "use_observed_only",
        "unjudged_gallery_policy": "operational_negative",
    }
    (configs / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    (configs / "ground_truth.yaml").write_text(
        yaml.safe_dump(policy, sort_keys=False), encoding="utf-8"
    )
    models = configs / "models"
    models.mkdir()
    clip = {
        "version": 1,
        "name": "clip",
        "adapter": "clip",
        "model_id": "openai/clip-vit-base-patch32",
        "revision": "a" * 40,
        "expected_dimension": 4,
        "batch_size": 2,
        "dtype": "float32",
        "preprocessing": "model_native",
        "embedding_output": "image_projection",
        "gated": False,
        "trust_remote_code": False,
    }
    (models / "clip.yaml").write_text(yaml.safe_dump(clip, sort_keys=False), encoding="utf-8")
    dino = {
        **clip,
        "name": "dino",
        "adapter": "dino",
        "model_id": "facebook/dino-vitb16",
        "revision": "b" * 40,
        "expected_dimension": 6,
        "embedding_output": "cls_token",
    }
    plip = {
        **clip,
        "name": "plip",
        "model_id": "vinid/plip",
        "revision": "c" * 40,
    }
    uni = {
        **clip,
        "name": "uni",
        "adapter": "uni",
        "model_id": "MahmoodLab/UNI",
        "revision": "d" * 40,
        "expected_dimension": 8,
        "embedding_output": "cls_token",
        "gated": True,
    }
    (models / "dino.yaml").write_text(yaml.safe_dump(dino, sort_keys=False), encoding="utf-8")
    (models / "plip.yaml").write_text(yaml.safe_dump(plip, sort_keys=False), encoding="utf-8")
    (models / "uni.yaml").write_text(yaml.safe_dump(uni, sort_keys=False), encoding="utf-8")
    retrieval = configs / "retrieval"
    retrieval.mkdir()
    retrieval_config = {
        "version": 1,
        "models": ["clip", "dino", "plip", "uni"],
        "metrics": ["cosine", "inner_product", "euclidean", "euclidean_l2"],
        "top_k": [1, 2, 4],
        "query_role": "query",
        "gallery_role": "gallery",
        "tie_breaker": "gallery_image_id_ascending",
        "relevance": "binary_curated",
    }
    (retrieval / "base.yaml").write_text(
        yaml.safe_dump(retrieval_config, sort_keys=False), encoding="utf-8"
    )

    arch_root = tmp_path / "data/raw/arch/pubmed_set"
    images = arch_root / "images"
    write_image(images / "11111111-1111-1111-1111-111111111111.jpg", (120, 10, 10))
    write_image(images / "22222222-2222-2222-2222-222222222222.png", (10, 120, 10))
    write_image(images / "33333333-3333-3333-3333-333333333333.jpg", (10, 10, 120))
    captions = {
        "0": {"caption": "query one", "uuid": "11111111-1111-1111-1111-111111111111"},
        "1": {"caption": "query two", "uuid": "22222222-2222-2222-2222-222222222222"},
        "2": {"caption": "distractor", "uuid": "33333333-3333-3333-3333-333333333333"},
    }
    arch_root.mkdir(parents=True, exist_ok=True)
    (arch_root / "captions.json").write_text(json.dumps(captions), encoding="utf-8")

    matches = tmp_path / "data/raw/matches"
    write_image(matches / "1_01.jpg", (90, 20, 20))
    write_image(matches / "1_01.png", (90, 20, 20))
    write_image(matches / "1_02.jpg", (80, 30, 20))
    write_image(matches / "2_01.PNG", (20, 80, 30))
    return load_context(root=tmp_path)

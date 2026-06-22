from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cipher.config import ModelSpec, load_model_spec
from cipher.models.access import check_model_access
from cipher.models.device import resolve_device
from cipher.models.dino import DinoEncoder
from cipher.models.uni import UniEncoder


def test_load_model_spec_is_strict_and_named(project_fixture) -> None:
    path, spec = load_model_spec("clip", root=project_fixture.root)
    assert path.name == "clip.yaml"
    assert spec.model_id == "openai/clip-vit-base-patch32"
    assert spec.revision == "a" * 40
    assert spec.preprocessing == "model_native"


@pytest.mark.parametrize(
    ("name", "model_id", "dimension", "output"),
    [
        ("dino", "facebook/dino-vitb16", 6, "cls_token"),
        ("plip", "vinid/plip", 4, "image_projection"),
        ("uni", "MahmoodLab/UNI", 8, "cls_token"),
    ],
)
def test_additional_model_specs(project_fixture, name, model_id, dimension, output) -> None:
    _, spec = load_model_spec(name, root=project_fixture.root)
    assert spec.model_id == model_id
    assert spec.expected_dimension == dimension
    assert spec.embedding_output == output


def test_auto_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_unavailable_explicit_device_fails(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_dino_adapter_selects_cls_token(monkeypatch) -> None:
    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls()

        def __call__(self, *, images, return_tensors):
            return {"pixel_values": torch.zeros((len(images), 3, 8, 8))}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert kwargs["add_pooling_layer"] is False
            return cls()

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, *, pixel_values):
            batch = pixel_values.shape[0]
            states = torch.arange(batch * 3 * 6, dtype=torch.float32).reshape(batch, 3, 6)
            return SimpleNamespace(last_hidden_state=states)

    monkeypatch.setattr("cipher.models.dino.ViTImageProcessor", FakeProcessor)
    monkeypatch.setattr("cipher.models.dino.ViTModel", FakeModel)
    spec = ModelSpec(
        version=1,
        name="dino",
        adapter="dino",
        model_id="facebook/dino-vitb16",
        revision="b" * 40,
        expected_dimension=6,
        batch_size=2,
        dtype="float32",
        preprocessing="model_native",
        embedding_output="cls_token",
        trust_remote_code=False,
    )
    encoder = DinoEncoder(spec, "cpu")
    result = encoder.encode([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])
    np.testing.assert_array_equal(result[0], np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.arange(18, 24, dtype=np.float32))


def test_gated_access_report_never_requires_a_token_value(project_fixture, monkeypatch) -> None:
    _, spec = load_model_spec("uni", root=project_fixture.root)
    monkeypatch.setattr("cipher.models.access.get_token", lambda: None)
    report = check_model_access(spec)
    assert report["status"] == "login_required"
    assert report["credential_present"] is False
    assert report["credential_source"] is None
    assert report["model_accessible"] is False
    assert "token" not in report


def test_uni_adapter_loads_strict_weights_and_selects_cls_token(
    project_fixture, monkeypatch
) -> None:
    class FakeModel:
        def load_state_dict(self, state_dict, strict):
            assert state_dict == {"fixture": "weights"}
            assert strict is True

        def eval(self):
            return self

        def to(self, device):
            return self

        def forward_features(self, pixels):
            batch = pixels.shape[0]
            assert pixels.shape[1:] == (3, 224, 224)
            return torch.arange(batch * 3 * 8, dtype=torch.float32).reshape(batch, 3, 8)

    config = {
        "architecture": "vit_large_patch16_224",
        "dynamic_img_size": True,
        "global_pool": "token",
        "img_size": 224,
        "num_classes": 0,
        "num_features": 8,
        "patch_size": 16,
        "pretrained_cfg": {
            "crop_pct": 1,
            "input_size": [3, 224, 224],
            "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    config_path = project_fixture.root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        "cipher.models.uni.download_model_file",
        lambda spec, filename: project_fixture.root / filename,
    )
    monkeypatch.setattr("cipher.models.uni.timm.create_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(
        "cipher.models.uni.torch.load", lambda *args, **kwargs: {"fixture": "weights"}
    )
    _, spec = load_model_spec("uni", root=project_fixture.root)
    encoder = UniEncoder(spec, "cpu")
    result = encoder.encode(
        [Image.new("RGB", (40, 20), "red"), Image.new("RGB", (20, 40), "blue")]
    )
    np.testing.assert_array_equal(result[0], np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.arange(24, 32, dtype=np.float32))

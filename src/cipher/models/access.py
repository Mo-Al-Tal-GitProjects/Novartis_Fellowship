from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_token, hf_hub_download, try_to_load_from_cache

from cipher.config import ModelSpec


class ModelAccessError(RuntimeError):
    """A sanitized model-access failure that never includes credential values."""


def _http_status(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def check_model_access(spec: ModelSpec) -> dict[str, Any]:
    """Verify authentication and gated-file access without returning identity or secrets."""
    token = get_token()
    report: dict[str, Any] = {
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "gated": spec.gated,
        "credential_present": bool(token),
        "credential_source": "environment" if os.getenv("HF_TOKEN") else "local_store",
        "authentication_valid": False,
        "model_accessible": False,
        "config_cached": isinstance(
            try_to_load_from_cache(spec.model_id, "config.json", revision=spec.revision), str
        ),
        "weights_cached": isinstance(
            try_to_load_from_cache(spec.model_id, "pytorch_model.bin", revision=spec.revision),
            str,
        ),
    }
    if not token:
        report["credential_source"] = None
        report["status"] = "login_required"
        report["next_action"] = "Run `hf auth login` locally with a read-only token."
        return report
    try:
        HfApi().whoami(token=token)
    except Exception as error:
        report["status"] = "invalid_token"
        report["http_status"] = _http_status(error)
        report["next_action"] = "Run `hf auth login` locally to replace the invalid token."
        return report
    report["authentication_valid"] = True
    try:
        hf_hub_download(
            repo_id=spec.model_id,
            filename="config.json",
            revision=spec.revision,
            token=token,
        )
    except Exception as error:
        report["status"] = "access_denied"
        report["http_status"] = _http_status(error)
        report["next_action"] = (
            "Confirm that this Hugging Face account has accepted the model terms."
        )
        return report
    report["config_cached"] = True
    report["model_accessible"] = True
    report["status"] = "accessible"
    return report


def download_model_file(spec: ModelSpec, filename: str) -> Path:
    """Download one immutable model file using locally managed credentials."""
    cached = try_to_load_from_cache(spec.model_id, filename, revision=spec.revision)
    if isinstance(cached, str):
        return Path(cached)
    token = get_token()
    if spec.gated and not token:
        raise ModelAccessError(
            f"{spec.model_id} requires local Hugging Face authentication; run `hf auth login`"
        )
    try:
        return Path(
            hf_hub_download(
                repo_id=spec.model_id,
                filename=filename,
                revision=spec.revision,
                token=token,
            )
        )
    except Exception as error:
        status = _http_status(error)
        suffix = f" (HTTP {status})" if status else ""
        raise ModelAccessError(
            f"could not access pinned file {filename!r} from {spec.model_id}{suffix}; "
            "verify `hf auth login` and gated-model approval"
        ) from None

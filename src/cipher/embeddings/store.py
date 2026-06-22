from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from cipher.provenance import sha256_file


def atomic_write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.save(handle, values, allow_pickle=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_cached_artifact(directory: Path, identity_sha256: str) -> dict[str, Any] | None:
    run_path = directory / "run.json"
    if not run_path.is_file():
        return None
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        if payload["identity_sha256"] != identity_sha256:
            return None
        for filename, expected_hash in payload["file_sha256"].items():
            path = directory / filename
            if not path.is_file() or sha256_file(path) != expected_hash:
                return None
        raw = np.load(directory / "embeddings.raw.npy", mmap_mode="r", allow_pickle=False)
        normalized = np.load(directory / "embeddings.l2.npy", mmap_mode="r", allow_pickle=False)
        if list(raw.shape) != payload["shape"] or normalized.shape != raw.shape:
            return None
        if raw.dtype != np.float32 or normalized.dtype != np.float32:
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload

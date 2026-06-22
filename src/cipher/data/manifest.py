from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cipher.config import ConfigContext
from cipher.data.images import inspect_image, is_allowed_image
from cipher.provenance import atomic_write_json, runtime_provenance, sha256_file

MATCH_PATTERN = re.compile(r"^(?P<query_id>\d+)_(?P<rank>\d+)\.(?:jpe?g|png)$", re.IGNORECASE)


def parse_match_filename(filename: str) -> tuple[str, str]:
    match = MATCH_PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f"invalid curated-match filename: {filename}")
    return match.group("query_id"), match.group("rank")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _inspect_record(path: Path, context: ConfigContext) -> dict[str, Any]:
    try:
        return inspect_image(
            path,
            compute_perceptual_hash=context.project.validation.compute_perceptual_hash,
        )
    except Exception as error:
        if context.project.validation.fail_on_decode_error:
            raise ValueError(f"failed to decode image {path}: {error}") from error
        return {
            "extension": path.suffix.lower(),
            "format": None,
            "width": None,
            "height": None,
            "mode": None,
            "channels": None,
            "sha256": sha256_file(path),
            "perceptual_hash": None,
            "decoder_warnings": str(error),
        }


def _duplicate_policy(context: ConfigContext) -> tuple[dict[str, str], dict[str, str]]:
    excluded_to_canonical: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for decision in context.policy.deduplications:
        reasons[decision.canonical.lower()] = decision.reason
        for excluded in decision.excluded:
            key = excluded.lower()
            if key in excluded_to_canonical:
                raise ValueError(f"duplicate exclusion policy for {excluded}")
            excluded_to_canonical[key] = decision.canonical
            reasons[key] = decision.reason
    return excluded_to_canonical, reasons


def _anomaly_notes(context: ConfigContext) -> dict[str, str]:
    notes: dict[str, str] = {}
    for anomaly in context.policy.retained_anomalies:
        for filename in anomaly.files:
            notes[filename.lower()] = anomaly.reason
    return notes


def _exclude_exact_arch_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one physical ARCH image per exact hash and record every exclusion."""
    arch = frame.loc[frame["source"] == "arch_pubmed"]
    for digest, group in arch.groupby("sha256", sort=False):
        if len(group) < 2:
            continue
        queries = group.loc[group["role"] == "query"]
        if len(queries) > 1:
            raise ValueError(f"multiple benchmark queries share exact image hash {digest}")
        if len(queries) == 1:
            canonical_index = queries.index[0]
        else:
            canonical_index = min(
                group.index,
                key=lambda index: (
                    int(frame.at[index, "source_record_id"]),
                    frame.at[index, "image_id"],
                ),
            )
        canonical_id = frame.at[canonical_index, "image_id"]
        for index in group.index:
            if index == canonical_index:
                continue
            frame.at[index, "role"] = "excluded"
            frame.at[index, "is_canonical"] = False
            frame.at[index, "duplicate_of"] = canonical_id
            frame.at[index, "validation_status"] = "excluded_exact_duplicate"
            frame.at[index, "validation_notes"] = (
                f"Exact ARCH duplicate of {canonical_id}; canonical row selected deterministically."
            )
    return frame


def build_image_manifest(context: ConfigContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    arch_root = context.resolve(context.project.paths.arch_root)
    matches_root = context.resolve(context.project.paths.matches_root)
    manifests_root = context.resolve(context.project.paths.manifests_root)
    captions_path = arch_root / "captions.json"
    images_root = arch_root / "images"
    if not captions_path.is_file() or not images_root.is_dir():
        raise FileNotFoundError("ARCH PubMed data is not prepared; run `cipher data prepare`")
    if not matches_root.is_dir():
        raise FileNotFoundError("curated match data is not prepared; run `cipher data prepare`")

    with captions_path.open("r", encoding="utf-8") as handle:
        captions = json.load(handle)
    if len(captions) != context.project.arch.expected_caption_records:
        raise ValueError(
            f"expected {context.project.arch.expected_caption_records} ARCH captions, "
            f"found {len(captions)}"
        )

    allowed = context.project.validation.allowed_extensions
    arch_images = [path for path in images_root.iterdir() if is_allowed_image(path, allowed)]
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in arch_images:
        by_stem[path.stem.lower()].append(path)

    rows: list[dict[str, Any]] = []
    query_by_uuid = {uuid.lower(): query_id for query_id, uuid in context.policy.queries.items()}
    for record_id in sorted(captions, key=lambda value: int(value)):
        entry = captions[record_id]
        uuid = str(entry["uuid"])
        candidates = by_stem.get(uuid.lower(), [])
        if len(candidates) != 1:
            raise ValueError(f"expected one image for ARCH UUID {uuid}, found {len(candidates)}")
        path = candidates[0]
        query_id = query_by_uuid.get(uuid.lower())
        details = _inspect_record(path, context)
        rows.append(
            {
                "image_id": f"arch:{uuid}",
                "relative_path": _relative(path, context.root),
                "filename": path.name,
                "source": "arch_pubmed",
                "source_record_id": str(record_id),
                "source_uuid": uuid,
                "query_id": query_id,
                "match_rank": None,
                "role": "query" if query_id else "gallery",
                "is_canonical": True,
                "duplicate_of": None,
                "caption": str(entry["caption"]),
                "caption_source": "arch_pubmed_captions",
                "validation_status": "valid",
                "validation_notes": details.pop("decoder_warnings"),
                **details,
            }
        )

    excluded_to_canonical, duplicate_reasons = _duplicate_policy(context)
    anomaly_notes = _anomaly_notes(context)
    match_files = [path for path in matches_root.iterdir() if is_allowed_image(path, allowed)]
    for path in sorted(match_files, key=lambda item: item.name.lower()):
        query_id, rank = parse_match_filename(path.name)
        if query_id not in context.policy.queries:
            raise ValueError(f"match file references unknown query {query_id}: {path.name}")
        lower_name = path.name.lower()
        excluded = lower_name in excluded_to_canonical
        details = _inspect_record(path, context)
        decoder_warnings = details.pop("decoder_warnings")
        notes = (
            duplicate_reasons.get(lower_name) or anomaly_notes.get(lower_name) or decoder_warnings
        )
        rows.append(
            {
                "image_id": f"match:{lower_name}",
                "relative_path": _relative(path, context.root),
                "filename": path.name,
                "source": "curated_match",
                "source_record_id": None,
                "source_uuid": None,
                "query_id": query_id,
                "match_rank": rank,
                "role": "excluded" if excluded else "gallery",
                "is_canonical": not excluded,
                "duplicate_of": (
                    f"match:{excluded_to_canonical[lower_name].lower()}" if excluded else None
                ),
                "caption": None,
                "caption_source": None,
                "validation_status": "excluded_duplicate" if excluded else "valid",
                "validation_notes": notes,
                **details,
            }
        )

    frame = _exclude_exact_arch_duplicates(pd.DataFrame(rows))
    frame = frame.sort_values(["source", "filename"], kind="stable").reset_index(drop=True)
    if frame["image_id"].duplicated().any():
        duplicates = frame.loc[frame["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"duplicate image IDs: {duplicates}")

    manifests_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_root / "images.parquet"
    frame.to_parquet(manifest_path, index=False)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest": _relative(manifest_path, context.root),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": len(frame),
        "source_counts": dict(Counter(frame["source"])),
        "role_counts": dict(Counter(frame["role"])),
        "valid_rows": int((frame["validation_status"] == "valid").sum()),
        "excluded_rows": int((frame["role"] == "excluded").sum()),
        "exact_arch_duplicates_excluded": int(
            (frame["validation_status"] == "excluded_exact_duplicate").sum()
        ),
        "runtime": runtime_provenance(),
    }
    atomic_write_json(manifests_root / "images.summary.json", summary)
    return frame, summary


def inspect_prepared_data(context: ConfigContext) -> dict[str, Any]:
    manifests_root = context.resolve(context.project.paths.manifests_root)
    manifest_path = manifests_root / "images.parquet"
    if manifest_path.is_file():
        frame = pd.read_parquet(manifest_path)
        return {
            "prepared": True,
            "image_rows": len(frame),
            "sources": frame["source"].value_counts().sort_index().to_dict(),
            "roles": frame["role"].value_counts().sort_index().to_dict(),
            "queries": int((frame["role"] == "query").sum()),
            "canonical_matches": int(
                ((frame["source"] == "curated_match") & (frame["role"] == "gallery")).sum()
            ),
            "manifest": _relative(manifest_path, context.root),
            "manifest_sha256": sha256_file(manifest_path),
        }

    arch_root = context.resolve(context.project.paths.arch_root)
    matches_root = context.resolve(context.project.paths.matches_root)
    allowed = context.project.validation.allowed_extensions
    arch_images = arch_root / "images"
    return {
        "prepared": False,
        "arch_images_on_disk": sum(
            1 for item in arch_images.iterdir() if is_allowed_image(item, allowed)
        )
        if arch_images.is_dir()
        else 0,
        "match_images_on_disk": sum(
            1 for item in matches_root.iterdir() if is_allowed_image(item, allowed)
        )
        if matches_root.is_dir()
        else 0,
        "message": "run `cipher data prepare` to generate manifests",
    }

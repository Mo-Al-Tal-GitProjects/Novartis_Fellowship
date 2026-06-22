from __future__ import annotations

import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import httpx

from cipher.provenance import sha256_file


def _byte_ranges(total_bytes: int, workers: int) -> list[tuple[int, int]]:
    workers = min(workers, total_bytes)
    chunk_size = (total_bytes + workers - 1) // workers
    return [
        (start, min(start + chunk_size - 1, total_bytes - 1))
        for start in range(0, total_bytes, chunk_size)
    ]


def _download_range(url: str, start: int, end: int, destination: Path) -> None:
    expected = end - start + 1
    written = 0
    with httpx.stream(
        "GET",
        url,
        headers={"Range": f"bytes={start}-{end}"},
        follow_redirects=True,
        timeout=120.0,
    ) as response:
        if response.status_code != httpx.codes.PARTIAL_CONTENT:
            raise RuntimeError(f"server rejected byte range {start}-{end}")
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                handle.write(chunk)
                written += len(chunk)
    if written != expected:
        raise RuntimeError(f"range {start}-{end} expected {expected} bytes, received {written}")


def _download_single(url: str, destination: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                handle.write(chunk)


def _download_parallel(url: str, destination: Path, total_bytes: int, workers: int) -> None:
    ranges = _byte_ranges(total_bytes, workers)
    parts = [
        destination.with_name(f"{destination.name}.{index:02d}") for index in range(len(ranges))
    ]
    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [
                executor.submit(_download_range, url, start, end, part)
                for (start, end), part in zip(ranges, parts, strict=True)
            ]
            for future in futures:
                future.result()
        with destination.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        if destination.stat().st_size != total_bytes:
            raise RuntimeError("parallel download size does not match Content-Length")
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def download_archive(
    url: str,
    destination: Path,
    *,
    force: bool = False,
    workers: int = 4,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return {
            "path": str(destination),
            "downloaded": False,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            head = client.head(url)
            head.raise_for_status()
            total_bytes = int(head.headers.get("content-length", "0"))
            supports_ranges = head.headers.get("accept-ranges", "").lower() == "bytes"

        if workers > 1 and supports_ranges and total_bytes > 0:
            try:
                _download_parallel(url, temporary, total_bytes, workers)
            except Exception:
                temporary.unlink(missing_ok=True)
                _download_single(url, temporary)
        else:
            _download_single(url, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "path": str(destination),
        "downloaded": True,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def extract_pubmed_archive(archive: Path, target: Path, *, force: bool = False) -> int:
    captions = target / "captions.json"
    images = target / "images"
    if captions.is_file() and images.is_dir() and not force:
        return sum(1 for item in images.iterdir() if item.is_file())

    temporary = target.with_name(f".{target.name}.extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    extracted = 0
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                member_path = PurePosixPath(member.filename)
                if not member_path.parts or member_path.parts[0] != "pubmed_set":
                    continue
                relative_parts = member_path.parts[1:]
                if not relative_parts or "__MACOSX" in member_path.parts:
                    continue
                if any(part in {"", ".", ".."} for part in relative_parts):
                    raise ValueError(f"unsafe ZIP member: {member.filename}")
                if relative_parts[-1] == ".DS_Store" or relative_parts[-1].startswith("._"):
                    continue

                destination = temporary.joinpath(*relative_parts)
                destination.resolve().relative_to(temporary.resolve())
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1

        if not (temporary / "captions.json").is_file() or not (temporary / "images").is_dir():
            raise ValueError("ARCH archive does not contain pubmed_set/captions.json and images/")
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return extracted


def import_matches(
    source: Path,
    destination: Path,
    allowed_extensions: list[str],
    *,
    force: bool = False,
) -> int:
    if not source.is_dir():
        raise FileNotFoundError(f"curated match directory not found: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in sorted(source.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_file() or item.suffix.lower() not in allowed_extensions:
            continue
        target = destination / item.name
        if force or not target.exists() or sha256_file(item) != sha256_file(target):
            temporary = target.with_suffix(target.suffix + ".part")
            shutil.copy2(item, temporary)
            os.replace(temporary, target)
        copied += 1
    return copied

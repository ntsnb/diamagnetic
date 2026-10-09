"""Relocate a frozen experiment matrix without importing the training stack.

All CSV files must already exist directly under --data-root with their original
basenames. Source configurations and CSV bytes are verified before any output is
created. Only data paths and their resulting configuration hashes are changed;
the original matrix, configurations, and data are never written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_hash(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{label}: missing or invalid SHA256")
    return value.lower()


def _portable_path(value, label: str):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: expected a nonempty path")
    windows = PureWindowsPath(value)
    return windows if windows.drive or "\\" in value else PurePosixPath(value)


def _config_path(value, source_root: Path) -> Path:
    portable = _portable_path(value, "configuration reference")
    if (portable.is_absolute() or portable.drive or portable.root
            or ".." in portable.parts or "\\" in value
            or not portable.name or portable.as_posix().casefold() == "matrix.json"):
        raise ValueError(f"configuration reference must be a safe relative POSIX path: {value!r}")
    relative = Path(*portable.parts)
    try:
        (source_root / relative).resolve().relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"configuration reference escapes the source directory: {value!r}") from error
    return relative


def _json_bytes(value) -> bytes:
    # Explicit bytes prevent host newline conversion from changing the new hash.
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def relocate_matrix(matrix_path: Path, data_root: Path, output: Path) -> dict:
    """Validate and copy a matrix to a new directory; return a compact summary."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output must not exist: {output}")
    output = output.resolve()
    matrix_path = Path(matrix_path).resolve(strict=True)
    data_root = Path(data_root).resolve(strict=True)
    if not data_root.is_dir():
        raise NotADirectoryError(f"data root is not a directory: {data_root}")

    original_bytes = matrix_path.read_bytes()
    original = json.loads(original_bytes.decode("utf-8-sig"))
    if not isinstance(original, dict) or original.get("schema_version") != 1:
        raise ValueError("expected an experiment matrix with schema_version 1")
    for key in ("datasets", "jobs", "baseline_jobs"):
        if not isinstance(original.get(key), list):
            raise ValueError(f"matrix.{key} must be a list")
    for items, count in (("datasets", "dataset_count"), ("jobs", "job_count"),
                         ("baseline_jobs", "baseline_job_count")):
        if original.get(count) != len(original[items]):
            raise ValueError(f"matrix.{count} does not match {items}")

    relocated = copy.deepcopy(original)
    datasets = {}
    basenames = set()
    for source, target in zip(original["datasets"], relocated["datasets"]):
        dataset_id = source.get("id")
        if not isinstance(dataset_id, str) or not dataset_id or dataset_id in datasets:
            raise ValueError(f"dataset ID is missing or duplicated: {dataset_id!r}")
        old_path = _portable_path(source.get("path"), f"dataset {dataset_id}")
        basename = old_path.name
        if not basename or basename in (".", "..") or basename.casefold() in basenames:
            raise ValueError(f"dataset basename collision or invalid basename: {basename!r}")
        basenames.add(basename.casefold())
        input_data = source.get("input_data")
        if not isinstance(input_data, dict):
            raise ValueError(f"dataset {dataset_id}: input_data must be an object")
        input_path = _portable_path(input_data.get("path"), f"dataset {dataset_id}.input_data")
        if input_path.as_posix() != old_path.as_posix():
            raise ValueError(f"dataset {dataset_id}: path and input_data.path disagree")
        data_path = data_root / basename
        expected = _expected_hash(source.get("source_sha256"), f"dataset {dataset_id}")
        if not data_path.is_file():
            raise FileNotFoundError(f"dataset {dataset_id}: missing CSV {data_path}")
        if _sha256(data_path) != expected:
            raise ValueError(f"dataset {dataset_id}: data SHA256 mismatch: {data_path}")
        new_path = data_path.as_posix()
        target["path"] = new_path
        target["input_data"]["path"] = new_path
        datasets[dataset_id] = (old_path.as_posix(), new_path)

    prepared = []
    config_paths = set()
    for group in ("jobs", "baseline_jobs"):
        for source, target in zip(original[group], relocated[group]):
            relative = _config_path(source.get("config"), matrix_path.parent)
            path_key = relative.as_posix().casefold()
            if path_key in config_paths:
                raise ValueError(f"duplicate configuration reference: {source['config']}")
            config_paths.add(path_key)
            source_config = matrix_path.parent / relative
            expected = _expected_hash(source.get("config_sha256"), str(relative))
            content = source_config.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"configuration SHA256 mismatch: {relative}")
            configuration = json.loads(content.decode("utf-8-sig"))
            dataset_id = source.get("dataset")
            if not isinstance(dataset_id, str) or dataset_id not in datasets:
                raise ValueError(f"configuration {relative}: unknown dataset {dataset_id!r}")
            data = configuration.get("data") if isinstance(configuration, dict) else None
            if not isinstance(data, dict):
                raise ValueError(f"configuration {relative}: data must be an object")
            old_path, new_path = datasets[dataset_id]
            if _portable_path(data.get("path"), str(relative)).as_posix() != old_path:
                raise ValueError(f"configuration {relative}: data.path disagrees with its dataset")
            data["path"] = new_path
            updated_bytes = _json_bytes(configuration)
            target["config_sha256"] = hashlib.sha256(updated_bytes).hexdigest()
            prepared.append((relative, updated_bytes))

    provenance = {
        "source_matrix": matrix_path.as_posix(),
        "source_matrix_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "verified_config_count": len(prepared),
        "verified_dataset_count": len(datasets),
        "data_root": data_root.as_posix(),
        "changes": ["configuration data.path", "dataset path and input_data.path",
                    "configuration SHA256 and this relocation record"],
        "note": "Copied frozen settings without rebuilding or tuning the experiment matrix. "
                "Original files were not modified. Data SHA256 values are unchanged. "
                "New JSON files use UTF-8 and LF; new configuration hashes are intentional.",
    }
    if "relocation" in original:
        provenance["previous_relocation"] = original["relocation"]
    relocated["relocation"] = provenance
    matrix_bytes = _json_bytes(relocated)

    # All data/configuration checks and serialization finish before the first write.
    output.mkdir(parents=True, exist_ok=False)
    for relative, content in prepared:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            target.write(content)
    with (output / "matrix.json").open("xb") as target:
        target.write(matrix_bytes)
    return {"output": output.as_posix(), "datasets": len(datasets), "configs": len(prepared),
            "source_matrix_sha256": provenance["source_matrix_sha256"],
            "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="frozen source matrix.json")
    parser.add_argument("--data-root", type=Path, required=True, help="directory containing the unchanged CSV files")
    parser.add_argument("--output", type=Path, required=True, help="new matrix directory; must not exist")
    args = parser.parse_args(argv)
    try:
        summary = relocate_matrix(args.matrix, args.data_root, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    # Keep machine-readable stdout independent of the shell's text encoding.
    print(json.dumps(summary, ensure_ascii=True))
    return summary


if __name__ == "__main__":
    main()

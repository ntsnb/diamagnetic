"""Relocation must preserve frozen evidence and reject incomplete transfers."""
import copy
import hashlib
import json
from pathlib import PureWindowsPath
import subprocess
import sys

import pytest

from tools.relocate_experiment_matrix import relocate_matrix


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Frozen Windows evidence includes CRLF; hashing must use the original bytes.
    content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8")
    path.write_bytes(content)
    return content


@pytest.fixture
def frozen(tmp_path):
    source = tmp_path / "frozen"
    data_root = tmp_path / "flat data"
    data_root.mkdir()
    matrix = {"schema_version": 1, "dataset_count": 2, "job_count": 2, "baseline_job_count": 1,
              "seeds": [42, 43], "builder_sha256": "b" * 64,
              "structural_definitions": {"full": {"variants": ["stages_111"]}},
              "datasets": [], "jobs": [], "baseline_jobs": []}
    for name, old_path in (("panama", "C:/research/load/panama.csv"),
                           ("city", "/research/load/丽水.csv")):
        basename = PureWindowsPath(old_path).name
        csv = b"time,load\r\n2026-01-01,1\r\n"
        (data_root / basename).write_bytes(csv)
        matrix["datasets"].append({"id": name, "path": old_path, "source_sha256": digest(csv),
                                   "input_data": {"path": old_path, "target_columns": ["load"]}})
    for group, name, seed in (("jobs", "panama", 42), ("jobs", "city", 43),
                               ("baseline_jobs", "panama", 43)):
        old_path = next(d["path"] for d in matrix["datasets"] if d["id"] == name)
        if group == "baseline_jobs":
            old_path = str(PureWindowsPath(old_path))
        config = {"seed": seed, "data": {"path": old_path, "train_fraction": .55,
                                          "validation_fraction": .10, "lookback": 24, "horizon": 12},
                  "training": {"epochs": 5}, "test_windows": None}
        relative = f"{group}/{name}/seed{seed}.json"
        content = write_json(source / relative, config)
        matrix[group].append({"dataset": name, "seed": seed, "config": relative,
                              "config_sha256": digest(content), "status": "configured_not_run"})
    matrix_path = source / "matrix.json"
    write_json(matrix_path, matrix)
    return matrix_path, data_root, tmp_path / "relocated", matrix


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_relocation_preserves_all_nonpath_settings_and_original_bytes(frozen):
    matrix_path, data_root, output, original = frozen
    before_source, before_data = snapshot(matrix_path.parent), snapshot(data_root)
    summary = relocate_matrix(matrix_path, data_root, output)
    assert summary["datasets"] == 2 and summary["configs"] == 3
    assert snapshot(matrix_path.parent) == before_source
    assert snapshot(data_root) == before_data
    migrated = json.loads((output / "matrix.json").read_text(encoding="utf-8"))
    expected = copy.deepcopy(original)
    for dataset in expected["datasets"]:
        path = (data_root / PureWindowsPath(dataset["path"]).name).resolve().as_posix()
        dataset["path"] = dataset["input_data"]["path"] = path
    for group in ("jobs", "baseline_jobs"):
        for expected_job, actual_job in zip(expected[group], migrated[group]):
            relative = expected_job["config"]
            old_config = json.loads(before_source[relative])
            old_config["data"]["path"] = next(d["path"] for d in expected["datasets"]
                                               if d["id"] == expected_job["dataset"])
            actual_bytes = (output / relative).read_bytes()
            assert json.loads(actual_bytes) == old_config
            assert b"\r\n" not in actual_bytes
            assert actual_job["config_sha256"] == digest(actual_bytes)
            expected_job["config_sha256"] = actual_job["config_sha256"]
    provenance = migrated.pop("relocation")
    assert provenance["source_matrix_sha256"] == digest(before_source["matrix.json"])
    assert provenance["verified_config_count"] == 3
    assert provenance["verified_dataset_count"] == 2
    assert migrated == expected


@pytest.mark.parametrize("group", ["jobs", "baseline_jobs"])
@pytest.mark.parametrize("failure", ["tampered", "missing"])
def test_bad_configuration_aborts_before_creating_output(frozen, group, failure):
    matrix_path, data_root, output, matrix = frozen
    path = matrix_path.parent / matrix[group][-1]["config"]
    if failure == "tampered":
        path.write_bytes(path.read_bytes() + b" ")
        error, message = ValueError, "configuration SHA256 mismatch"
    else:
        path.unlink()
        error, message = FileNotFoundError, None
    with pytest.raises(error, match=message):
        relocate_matrix(matrix_path, data_root, output)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["tampered", "missing"])
def test_bad_data_aborts_before_creating_output(frozen, failure):
    matrix_path, data_root, output, _ = frozen
    path = data_root / "丽水.csv"
    if failure == "tampered":
        path.write_bytes(b"different observations\n")
        error, message = ValueError, "data SHA256 mismatch"
    else:
        path.unlink()
        error, message = FileNotFoundError, "missing CSV"
    with pytest.raises(error, match=message):
        relocate_matrix(matrix_path, data_root, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(frozen):
    matrix_path, data_root, output, _ = frozen
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_bytes(b"previous evidence")
    with pytest.raises(FileExistsError, match="output must not exist"):
        relocate_matrix(matrix_path, data_root, output)
    assert snapshot(output) == {"keep.txt": b"previous evidence"}


def test_flat_dataset_basename_collision_is_rejected(frozen):
    matrix_path, data_root, output, matrix = frozen
    matrix["datasets"][1]["path"] = "D:\\elsewhere\\panama.csv"
    matrix["datasets"][1]["input_data"]["path"] = matrix["datasets"][1]["path"]
    write_json(matrix_path, matrix)
    with pytest.raises(ValueError, match="basename collision"):
        relocate_matrix(matrix_path, data_root, output)
    assert not output.exists()


@pytest.mark.parametrize("reference", ["../outside.json", "C:/outside.json", "/outside.json"])
def test_configuration_paths_cannot_escape_source_or_output(frozen, reference):
    matrix_path, data_root, output, matrix = frozen
    matrix["jobs"][0]["config"] = reference
    write_json(matrix_path, matrix)
    with pytest.raises(ValueError, match="safe relative POSIX path"):
        relocate_matrix(matrix_path, data_root, output)
    assert not output.exists()


def test_cli_runs_without_site_packages(frozen):
    matrix_path, data_root, output, _ = frozen
    output = output.with_name("迁移矩阵")
    script = __import__("tools.relocate_experiment_matrix", fromlist=["__file__"]).__file__
    result = subprocess.run([sys.executable, "-I", "-S", script, "--matrix", str(matrix_path),
                             "--data-root", str(data_root), "--output", str(output)],
                            check=True, capture_output=True)
    summary = json.loads(result.stdout.decode("ascii"))
    assert summary["configs"] == 3
    assert summary["output"] == output.resolve().as_posix()
    assert (output / "matrix.json").is_file()

import csv
import hashlib
import pandas as pd
import pytest
from tools.paper_impl_repair_spain_clock import repair


def test_repair_preserves_each_observation_and_source(tmp_path):
    source, output = tmp_path / "original.csv", tmp_path / "corrected.csv"
    dates = pd.date_range("2015-01-01", periods=24*40, freq="h")
    corrupted = [t.replace(month=t.day, day=t.month) if t.day <= 12 else t for t in dates]
    with source.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "load", "feature"])
        writer.writerows((str(t), f"{i}.123456789", "") for i,t in enumerate(corrupted))
    original = source.read_bytes()
    manifest = repair(source, output)
    assert source.read_bytes() == original
    assert manifest["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["all_observation_strings_unchanged"]
    frame = pd.read_csv(output)
    assert (pd.to_datetime(frame.time).to_numpy() == dates.to_numpy()).all()
    assert frame.feature.isna().all()
    with pytest.raises(FileExistsError):
        repair(source, output)


def test_repair_refuses_unrelated_disorder(tmp_path):
    source = tmp_path / "unrelated.csv"
    source.write_text("time,load\n2015-02-01 00:00:00,1\n2015-02-03 00:00:00,2\n", encoding="utf8")
    with pytest.raises(ValueError, match="does not match"):
        repair(source, tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()

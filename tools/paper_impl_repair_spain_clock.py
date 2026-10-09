"""修复已核验的Spain日月歧义时间戳；保持观测字符串与行顺序不变。

仅接受逆转歧义日月后形成严格逐小时序列的文件。原文件不排序、不覆盖。
这项修复不撤销上游文件可能已经进行的插值，也不恢复缺失值掩码。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd


def repair(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.with_suffix(".provenance.json").exists():
        raise FileExistsError("choose a new output filename")
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding="utf8", newline="") as handle:
        records = list(csv.reader(handle))
    header, rows = records[0], records[1:]
    column = header.index("time")
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError("invalid CSV rows")
    before = pd.DatetimeIndex(pd.to_datetime([row[column] for row in rows], errors="raise"))
    after = pd.DatetimeIndex([value.replace(month=value.day, day=value.month)
                              if value.day <= 12 else value for value in before])
    expected = pd.date_range(after[0], periods=len(after), freq="h")
    if not (after == expected).all() or (before == after).all():
        raise ValueError("file does not match the verified ambiguous day/month defect")
    rewritten = [list(row) for row in rows]
    for row, timestamp in zip(rewritten, after):
        row[column] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rewritten)
    with output.open(encoding="utf8", newline="") as handle:
        actual = list(csv.reader(handle))[1:]
    unchanged = all(a[:column] + a[column+1:] == b[:column] + b[column+1:]
                    for a, b in zip(rows, actual)) and len(rows) == len(actual)
    if not unchanged or hashlib.sha256(source.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("observation/source preservation check failed")
    manifest = {"source": str(source), "source_sha256": original_hash, "output": str(output),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "rows": len(rows),
                "changed_timestamp_rows": int((before != after).sum()), "interval_seconds": 3600,
                "operation": "invert month/day for ambiguous stored dates (day <= 12); verify every row against hourly chronology",
                "all_observation_strings_unchanged": unchanged, "row_order_unchanged": True,
                "first_timestamp": str(after[0]), "last_timestamp": str(after[-1]),
                "upstream_imputation_status": "not audited; this is the previously downloaded cleaned dataset"}
    output.with_suffix(".provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.source, args.output), ensure_ascii=False))

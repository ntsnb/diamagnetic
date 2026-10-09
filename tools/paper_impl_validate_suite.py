"""顺序执行有界真实数据验证，记录失败并保护既有实验输出。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    args = parser.parse_args()
    configs = sorted(args.configs.resolve().glob("*.json"))
    if not configs or args.timeout_seconds < 1:
        raise ValueError("nonempty config directory and positive timeout required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"started": datetime.now(timezone.utc).isoformat(), "scope": "functional validation only", "runs": []}
    for config in configs:
        destination = output / config.stem
        command = [sys.executable, "-m", "paper_impl.run", "--config", str(config), "--output", str(destination)]
        started = time.perf_counter()
        record = {"name": config.stem, "command": command,
                  "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
        try:
            with (output / (config.stem + ".log")).open("w", encoding="utf8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout_seconds, check=False,
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            record["returncode"] = result.returncode
            summary = destination / "summary.json"
            record["completed"] = result.returncode == 0 and summary.exists()
        except subprocess.TimeoutExpired:
            record.update(completed=False, error="time budget exceeded")
        record["seconds"] = time.perf_counter() - started
        report["runs"].append(record)
        (output / "suite_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report["completed"] = all(row["completed"] for row in report["runs"])
    report["finished"] = datetime.now(timezone.utc).isoformat()
    (output / "suite_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

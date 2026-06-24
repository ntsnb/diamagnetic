from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lishui_dataset import prepare_lishui_family
from utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="dataset/raw/lishui")
    parser.add_argument("--output-dir", default="dataset/load_benchmarks/lishui")
    args = parser.parse_args()

    raw_root = resolve_project_path(args.raw_root)
    output_dir = resolve_project_path(args.output_dir)
    prepared = prepare_lishui_family(str(raw_root), str(output_dir))
    print(f"prepared {len(prepared)} county datasets into {output_dir}")
    for item in prepared:
        s = item.summary
        print(
            f"[{item.county}] station={item.station} rows={s['rows']} "
            f"range=({s['start']}, {s['end']}) load_mean={s['load_mean']:.3f}"
        )


if __name__ == "__main__":
    main()

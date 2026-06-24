from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PIPELINE = {
    "backbone_screening": "train/report_backbone_screening.py",
    "backbone_variants": "train/report_backbone_variants.py",
    "multidataset_panels": "train/report_detailed_multidataset_figures.py",
    "case_studies": "train/analyze_case_study_multidataset.py",
    "tsne_public": "train/analyze_tsne_components.py",
    "tsne_lishui": "train/analyze_tsne_lishui_family.py",
    "lishui_family_csv": "train/report_lishui_family_benchmark.py",
    "lishui_shifted_csv": "train/report_lishui_shifted_sota.py",
    "lishui_summary_figure": "train/report_lishui_summary_figure.py",
    "lishui_shifted_nine_counties": "train/report_lishui_shifted_sota_nine_counties.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate all paper figures from existing experiment outputs.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional subset of steps to run. Choices are the pipeline keys printed by --list.",
    )
    parser.add_argument("--list", action="store_true", help="List available figure-generation steps and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for key, script in PIPELINE.items():
            print(f"{key}: {script}")
        return

    selected = args.only or list(PIPELINE.keys())
    unknown = [name for name in selected if name not in PIPELINE]
    if unknown:
        raise SystemExit(f"Unknown steps: {', '.join(unknown)}")

    for name in selected:
        script = ROOT / PIPELINE[name]
        print(f"[make_all_paper_figures] running {name}: {script}")
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()

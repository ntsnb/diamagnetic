from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOT_DIRS = {
    "core",
    "CLI-Anything",
    "autoresearch-master",
}

FORBIDDEN_ROOT_FILES = {
    "core.zip",
    "main_results.tsv",
    "main_search_results.tsv",
    "New_IEEEtran_how-to.tex",
    "New_IEEEtran_how-to.pdf",
    "New_IEEEtran_how-to.aux",
    "New_IEEEtran_how-to.log",
    "New_IEEEtran_how-to.synctex.gz",
    "fig1.png",
}

FORBIDDEN_ROOT_SUFFIXES = {
    ".aux",
    ".log",
    ".fls",
    ".fdb_latexmk",
    ".xdv",
}

REQUIRED_PATHS = {
    "archive/references/papers",
    "archive/references/extracted_text",
    "archive/latex_templates/ieee_howto",
    "archive/external",
    "archive/external_sources/core",
    "archive/external_sources/CLI-Anything",
    "archive/external_sources/autoresearch-master",
    "artifacts/latex/build",
    "artifacts/autoresearch/main_results.tsv",
    "artifacts/autoresearch/main_search_results.tsv",
    "fig1.jpg",
    "fig2.jpg",
}


def audit_layout() -> list[str]:
    problems: list[str] = []

    for name in sorted(FORBIDDEN_ROOT_DIRS):
        if (ROOT / name).exists():
            problems.append(f"forbidden root directory still exists: {name}")

    for item in ROOT.iterdir():
        if not item.is_file():
            continue
        if item.name in FORBIDDEN_ROOT_FILES:
            problems.append(f"forbidden root file still exists: {item.name}")
        if item.name.endswith(".synctex.gz") or item.suffix in FORBIDDEN_ROOT_SUFFIXES:
            problems.append(f"latex intermediate remains at root: {item.name}")

    for rel in sorted(REQUIRED_PATHS):
        if not (ROOT / rel).exists():
            problems.append(f"required organized path missing: {rel}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the high-level repository layout.")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    problems = audit_layout()
    print(f"checked_root={ROOT}")
    print(f"layout_problems={len(problems)}")
    for item in problems:
        print(f"  problem: {item}")

    if args.strict and problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

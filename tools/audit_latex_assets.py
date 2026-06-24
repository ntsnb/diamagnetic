from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
GRAPHIC_EXTENSIONS = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")


def _candidate_paths(tex_file: Path, ref: str) -> list[Path]:
    raw = Path(ref)
    if raw.is_absolute():
        bases = [raw]
    else:
        bases = [tex_file.parent / raw]
    candidates: list[Path] = []
    for base in bases:
        if base.suffix:
            candidates.append(base)
        else:
            candidates.extend(Path(str(base) + ext) for ext in GRAPHIC_EXTENSIONS if ext)
    return candidates


def audit_tex_file(tex_file: Path) -> list[str]:
    missing: list[str] = []
    text = tex_file.read_text(encoding="utf-8", errors="ignore")
    for match in GRAPHICS_RE.finditer(text):
        ref = match.group(1).strip()
        candidates = _candidate_paths(tex_file, ref)
        if not any(path.exists() for path in candidates):
            missing.append(f"{tex_file.relative_to(ROOT)} -> {ref}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check LaTeX includegraphics paths.")
    parser.add_argument(
        "tex_files",
        nargs="*",
        default=[
            "bare_jrnl_new_sample4.tex",
            "bare_jrnl_new_sample4_redline.tex",
            "bare_jrnl_new_sample4_audit_annotated.tex",
            "bare_jrnl_new_sample4_revision_markup.tex",
            "archive/latex_templates/ieee_howto/New_IEEEtran_how-to.tex",
        ],
    )
    args = parser.parse_args()

    missing: list[str] = []
    for item in args.tex_files:
        tex_file = ROOT / item
        if not tex_file.exists():
            missing.append(f"missing tex file: {item}")
            continue
        missing.extend(audit_tex_file(tex_file))

    print(f"checked_tex_files={len(args.tex_files)}")
    print(f"missing_graphics={len(missing)}")
    for item in missing:
        print(f"  missing: {item}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

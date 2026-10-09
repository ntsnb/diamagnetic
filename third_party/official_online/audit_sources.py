"""Record exact upstream snapshot commits and per-file SHA256 evidence.

Run before removing nested .git directories from the vendored snapshots.
Later runs verify files against the saved commit metadata without requiring git.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
SOURCES = {
    "OnlineTSF": {
        "url": "https://github.com/SJTU-DMTai/OnlineTSF",
        "paper": "https://arxiv.org/abs/2412.08435",
        "license": "No repository LICENSE/COPYING file at pinned commit; no grant inferred.",
        "role": "PROCEED authors' implementation; delayed-feedback OneNet and SOLID++ reference",
    },
    "OneNet": {
        "url": "https://github.com/yfzhang114/OneNet",
        "paper": "https://arxiv.org/abs/2309.12659",
        "license": "README states MIT but its linked LICENSE file is absent at pinned commit.",
        "role": "Original OneNet authors' ensemble/decision optimizer reference",
    },
    "SOLID": {
        "url": "https://github.com/HALF111/calibration_CDS",
        "paper": "https://arxiv.org/abs/2310.14838",
        "license": "MIT; full source LICENSE preserved (DAMO Academy @ Alibaba, 2021).",
        "role": "Original SOLID episodic prediction-head adaptation and context-selection reference",
    },
}


def build_manifest():
    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    for name, metadata in SOURCES.items():
        folder = ROOT / name
        if (folder / ".git").exists():
            commit = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
            committed_at = subprocess.check_output(["git", "-C", str(folder), "show", "-s", "--format=%cI", "HEAD"], text=True).strip()
        else:
            commit = previous["sources"][name]["commit"]
            committed_at = previous["sources"][name]["committed_at"]
        hashes = {}
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder)
            if path.is_file() and ".git" not in relative.parts and "__pycache__" not in relative.parts:
                hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        pinned = previous.get("sources", {}).get(name)
        if pinned and (pinned["commit"] != commit or pinned["sha256"] != hashes):
            raise RuntimeError(f"{name}: snapshot bytes or commit differ from the saved source evidence")
        records[name] = dict(metadata, commit=commit, committed_at=committed_at,
                             source_files_modified=False, sha256=hashes)
    result = {"retrieved_on": "2026-10-07", "sources": records,
              "port": "paper_impl/online_baselines.py",
              "audit": "docs/audit/full_experiment_20261007/ONLINE_BASELINES.md"}
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    manifest = build_manifest()
    for name, record in manifest["sources"].items():
        print(f"{name}: {record['commit']}; {len(record['sha256'])} files; {record['license']}")

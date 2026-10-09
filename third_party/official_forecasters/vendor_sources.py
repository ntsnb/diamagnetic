"""Fetch and pin public author sources; never used at model runtime.

The manifest records exact upstream bytes. Runtime copies are subsequently
namespaced and their compatibility patches recorded separately.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
REPOSITORIES = {
    "itransformer": ("thuml/iTransformer", ("model/iTransformer.py", "layers/", "utils/masking.py")),
    "crossformer": ("Thinklab-SJTU/Crossformer", ("cross_models/",)),
    "autoformer": ("thuml/Autoformer", ("models/Autoformer.py", "layers/", "utils/masking.py")),
    "transformer": ("thuml/Time-Series-Library", ("models/Transformer.py", "layers/Transformer_EncDec.py", "layers/SelfAttention_Family.py", "layers/Embed.py", "utils/masking.py")),
    "patchtst": ("yuqinie98/PatchTST", ("PatchTST_supervised/models/PatchTST.py", "PatchTST_supervised/layers/")),
    "informer": ("zhouhaoyi/Informer2020", ("models/", "utils/masking.py")),
    "fedformer": ("DAMO-DI-ML/ICML2022-FEDformer", ("models/FEDformer.py", "layers/", "utils/masking.py")),
    "pyraformer": ("ant-research/Pyraformer", ("pyraformer/", "long_range_main.py")),
}


def fetch(name, repo, includes, existing):
    url = f"https://github.com/{repo}"
    sha = existing.get(name, {}).get("commit")
    if not sha:
        sha = subprocess.check_output(
            ["git", "-c", "http.sslBackend=openssl", "ls-remote", url + ".git", "HEAD"],
            text=True,
        ).split()[0]
    request = urllib.request.Request(f"https://codeload.github.com/{repo}/zip/{sha}", headers={"User-Agent": "official-forecast-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        archive_bytes = response.read()
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    prefix = archive.namelist()[0]
    record = {"repository": url, "commit": sha, "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(), "files": {}}
    for item in archive.infolist():
        path = item.filename.removeprefix(prefix)
        if not path or item.is_dir():
            continue
        top_text = "/" not in path and (path.lower().startswith(("license", "notice", "readme")) or path.lower() == "requirements.txt")
        include = path.endswith(".py") and any(path.startswith(p) if p.endswith("/") else path == p for p in includes)
        if not (top_text or include):
            continue
        content = archive.read(item)
        target = ROOT / "upstream" / name / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        record["files"][path] = hashlib.sha256(content).hexdigest()
    print(name, sha, len(record["files"]), flush=True)
    return name, record


if __name__ == "__main__":
    manifest_file = ROOT / "SOURCE_MANIFEST.json"
    previous = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, name, repo, includes, previous) for name, (repo, includes) in REPOSITORIES.items()]
        for result in concurrent.futures.as_completed(futures):
            name, record = result.result()
            records[name] = record
    manifest_file.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

"""Recreate isolated runtime copies and exact unified compatibility diffs."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def rewrite(name: str, path: str, text: str) -> str:
    namespace = f"third_party.official_forecasters.runtime.{name}"
    if name == "patchtst":
        namespace += ".PatchTST_supervised"
    text = re.sub(r"(?m)^(from )(layers|models|model|utils|cross_models)(?=[. ])", rf"\g<1>{namespace}.\2", text)
    # These forecast paths never instantiate the unrelated Reformer class. Keep
    # that class intact but defer its optional third-party dependency to use.
    if "from reformer_pytorch import LSHSelfAttention\n" in text:
        text = text.replace("from reformer_pytorch import LSHSelfAttention\n", "")
        text = text.replace("        self.attn = LSHSelfAttention(", "        from reformer_pytorch import LSHSelfAttention\n        self.attn = LSHSelfAttention(")
    # Preserve the same normalization while allowing gradients with respect to
    # the input; variance backward saves the tensor modified by upstream /=.
    if name == "itransformer" and path == "model/iTransformer.py":
        text = text.replace("            x_enc /= stdev", "            x_enc = x_enc / stdev")
    # This tensor is unused by FEDformer's forward, but the original creates it
    # on a process-global CUDA device even when the model and input are on CPU.
    if name == "fedformer" and path == "models/FEDformer.py":
        text = text.replace("torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)", "x_enc.new_zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]])")
    # graph_attention is an unused profiling module; keep its sibling import
    # within the same package if it is explicitly imported by a caller.
    if name == "pyraformer" and path == "pyraformer/graph_attention.py":
        text = text.replace("from hierarchical_mm_tvm import", "from .hierarchical_mm_tvm import")
    return text


if __name__ == "__main__":
    sources = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    runtime_manifest = {}
    for name, record in sorted(sources.items()):
        patches = []
        hashes = {}
        for path in sorted(record["files"]):
            if not path.endswith(".py") or path == "long_range_main.py":
                continue
            src = ROOT / "upstream" / name / path
            original = src.read_text(encoding="utf-8")
            content = rewrite(name, path, original)
            target = ROOT / "runtime" / name / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
            hashes[path] = hashlib.sha256(target.read_bytes()).hexdigest()
            patches.extend(difflib.unified_diff(original.splitlines(keepends=True), content.splitlines(keepends=True), fromfile=f"upstream/{name}/{path}", tofile=f"runtime/{name}/{path}"))
        patch_path = ROOT / "patches" / (name + ".patch")
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text("".join(patches), encoding="utf-8", newline="\n")
        runtime_manifest[name] = {"commit": record["commit"], "files": hashes, "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest()}
    (ROOT / "RUNTIME_MANIFEST.json").write_text(json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

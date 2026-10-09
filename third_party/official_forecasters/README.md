# Pinned complete author forecasting networks

The adapter is `paper_impl/official_backbones.py`. Provenance, supported variants,
input semantics, Head-GD boundaries, and measured smoke-test coverage are recorded
in `docs/audit/full_experiment_20261007/OFFICIAL_BACKBONES.md`.

- `upstream/<name>/`: selected complete forecast modules and their dependency
  modules, original README, LICENSE, and requirements, stored byte-for-byte.
- `SOURCE_MANIFEST.json`: author repository, full commit, downloaded archive SHA256,
  and SHA256 of each stored upstream file.
- `runtime/<name>/`: import-isolated copies used at runtime.
- `patches/<name>.patch`: every source-text difference from upstream, excluding
  only newline normalization. These are compatibility/import patches, not new
  replacement networks.
- `RUNTIME_MANIFEST.json`: hashes of every runtime module and patch.
- `vendor_sources.py`: download sources at the existing manifest's pinned commits.
- `prepare_runtime.py`: regenerate runtime copies and exact patches locally.

No model runtime downloads occur. Different repos never share global `layers`,
`models`, or `utils` import aliases; no `sys.path` modification is needed.

The source snapshots contain supporting unused author classes for auditability;
the supported forecast entry points and variants are explicitly listed in the
audit document. Vendoring is not a claim to have reproduced any paper's scores.

To reproduce source preparation from the repository root:

```powershell
python third_party/official_forecasters/vendor_sources.py
python third_party/official_forecasters/prepare_runtime.py
python -m pytest tests/paper_impl/test_official_backbones.py -q
```

Runtime dependencies beyond the project's PyTorch/NumPy are listed in
`requirements-runtime.txt`. They were already installed during verification.

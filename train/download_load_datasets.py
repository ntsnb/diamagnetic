from __future__ import annotations

import argparse
import io
import os
import urllib.request
import zipfile


DATASETS = [
    {
        "name": "panama",
        "url": "https://raw.githubusercontent.com/kausnz/panama-electricity-load-forecast/main/data/continuous_dataset.csv",
        "output": "dataset/load_benchmarks/panama.csv",
        "kind": "file",
    },
    {
        "name": "tetouan",
        "url": "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip",
        "output": "dataset/load_benchmarks/tetouan.csv",
        "kind": "zip_member",
        "member": "Tetuan City power consumption.csv",
    },
    {
        "name": "spain_energy_weather",
        "url": "https://raw.githubusercontent.com/kolasniwash/short-term-energy-demand-forecasting/main/data/cleaned_data/energy_weather_2015_2018.csv",
        "output": "dataset/load_benchmarks/spain_energy_weather.csv",
        "kind": "file",
    },
]


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    for ds in DATASETS:
        out_path = os.path.join(root, ds["output"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"[download] {ds['name']} -> {out_path}")
        payload = download_bytes(ds["url"])
        if ds["kind"] == "file":
            with open(out_path, "wb") as f:
                f.write(payload)
        elif ds["kind"] == "zip_member":
            zf = zipfile.ZipFile(io.BytesIO(payload))
            with zf.open(ds["member"]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
        else:
            raise ValueError(f"unknown dataset kind: {ds['kind']}")
        print(f"[done] {ds['name']} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import base64
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


TD_REPO_RAW = "https://raw.githubusercontent.com/kushalareddy2006/Prediction-of-Polymer-Decomposition-Temperature-ML-DL/main/10000.xlsx"
TD_REPO_API = "https://api.github.com/repos/kushalareddy2006/Prediction-of-Polymer-Decomposition-Temperature-ML-DL/contents/10000.xlsx"
MULTI_BASE = (
    "https://cdn.jsdelivr.net/gh/"
    "abdullah85-github/-an-open-reference-dataset-of-60-polymers-across-ten-thermal@main"
)
OPOLY_TG_URL = (
    "https://cdn.jsdelivr.net/gh/maxhartml/NeuralIPS-Polymer@main/"
    "neurips-open-polymer-prediction-2025/external_data/"
    "TgSS_enriched_cleaned.csv"
)
MULTI_FILES = [
    "full_multi_property.csv",
    "train.csv",
    "validation.csv",
    "test.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize public polymer property datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Codex-PU-DataCollector/0.1"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())
    return destination


def download_github_binary(api_url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(
        api_url,
        headers={"User-Agent": "Codex-PU-DataCollector/0.1"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    destination.write_bytes(base64.b64decode(payload["content"]))
    return destination


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    raw_dir = output / "data" / "raw" / "public"
    training_dir = output / "data" / "training"
    raw_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    td_xlsx = download_github_binary(
        TD_REPO_API,
        raw_dir / "polymer_td_1023.xlsx",
    )
    td_frame = pd.read_excel(td_xlsx)
    td_frame.to_csv(training_dir / "network_polymer_td_1023.csv", index=False)

    multi_destination = raw_dir / "multi_property_439.csv"
    download(
        f"{MULTI_BASE}/full_multi_property.csv",
        multi_destination,
    )
    multi_frame = pd.read_csv(multi_destination)
    multi_frame.to_csv(
        training_dir / "network_polymer_multi_property_439.csv",
        index=False,
    )

    tg_destination = raw_dir / "polymer_tg_7284.csv"
    download(OPOLY_TG_URL, tg_destination)
    tg_frame = pd.read_csv(tg_destination)
    tg_frame.to_csv(training_dir / "tg_polymers.csv", index=False)

    print(
        json.dumps(
            {
                "polymer_td_rows": len(td_frame),
                "multi_property_rows": len(multi_frame),
                "tg_rows": len(tg_frame),
                "raw_dir": str(raw_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

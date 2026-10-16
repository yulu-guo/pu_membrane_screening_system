from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from pu_v2.model import morgan_bit_vector


SMILES_TOKEN = re.compile(r"'(\[\*\][^']+)'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train broad polymer Td models from public network data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--trees", type=int, default=250)
    return parser.parse_args()


def mixture_features(smiles_text: str, composition: float) -> np.ndarray:
    tokens = SMILES_TOKEN.findall(str(smiles_text))
    if not tokens:
        tokens = [str(smiles_text)]
    fingerprints = []
    for token in tokens[:2]:
        try:
            fingerprints.append(morgan_bit_vector(token))
        except Exception:
            continue
    if not fingerprints:
        return np.concatenate([np.zeros(2048), [composition]])
    if len(fingerprints) == 1:
        vector = fingerprints[0].astype(float)
    else:
        fraction = min(max(float(composition), 0.0), 1.0)
        vector = (
            fraction * fingerprints[0].astype(float)
            + (1.0 - fraction) * fingerprints[1].astype(float)
        )
    return np.concatenate([vector, [composition]])


def direct_smiles_features(smiles: str) -> np.ndarray:
    return np.concatenate([morgan_bit_vector(smiles).astype(float), [0.5]])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_td_model(
    output_dir: Path,
    trees: int,
) -> tuple[ExtraTreesRegressor, dict]:
    path = output_dir / "data" / "training" / "network_polymer_td_1023.csv"
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["Td", "smiles", "comp"]).drop_duplicates(
        ["smiles", "comp"]
    )
    x = np.vstack(
        [
            mixture_features(row["smiles"], row["comp"])
            for _, row in frame.iterrows()
        ]
    )
    y = frame["Td"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = ExtraTreesRegressor(
        n_estimators=trees,
        max_features="sqrt",
        max_depth=35,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    score = metrics(y_test, model.predict(x_test))
    final = ExtraTreesRegressor(
        n_estimators=trees,
        max_features="sqrt",
        max_depth=35,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    final.fit(x, y)
    return final, {
        "rows": len(frame),
        "holdout": score,
        "target": "Td",
        "source": "GitHub ML-DL Td dataset derived from Polymer Genome",
    }


def train_td5_model(
    output_dir: Path,
    trees: int,
) -> tuple[ExtraTreesRegressor, dict]:
    path = (
        output_dir
        / "data"
        / "training"
        / "network_polymer_multi_property_439.csv"
    )
    frame = pd.read_csv(path)
    frame = frame.loc[
        (frame["property_name"] == "Td5_C")
        & frame["repeat_unit_smiles_canonical"].notna()
        & frame["property_value_standardized"].notna()
    ].copy()
    frame["sample_weight"] = frame["confidence_score_final"].fillna(
        frame["confidence_score"]
    )
    grouped = (
        frame.groupby("repeat_unit_smiles_canonical")
        .agg(
            value=("property_value_standardized", "mean"),
            weight=("sample_weight", "mean"),
        )
        .reset_index()
    )
    x = np.vstack(
        [
            direct_smiles_features(smiles)
            for smiles in grouped["repeat_unit_smiles_canonical"]
        ]
    )
    y = grouped["value"].to_numpy(dtype=float)
    weights = grouped["weight"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test, w_train, _ = train_test_split(
        x,
        y,
        weights,
        test_size=0.2,
        random_state=42,
    )
    evaluation = ExtraTreesRegressor(
        n_estimators=trees,
        max_features="sqrt",
        max_depth=30,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    evaluation.fit(x_train, y_train, sample_weight=w_train)
    score = metrics(y_test, evaluation.predict(x_test))
    final = ExtraTreesRegressor(
        n_estimators=trees,
        max_features="sqrt",
        max_depth=30,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    final.fit(x, y, sample_weight=weights)
    return final, {
        "rows": len(grouped),
        "holdout": score,
        "target": "Td5_C",
        "source": "Open reference dataset of 60 polymers across ten thermal properties",
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    model_dir = output / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    td_model, td_metrics = train_td_model(output, args.trees)
    td5_model, td5_metrics = train_td5_model(output, args.trees)

    joblib.dump(
        {
            "model": td_model,
            "target": "Td",
            "fingerprint_bits": 2048,
            "metrics": td_metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        model_dir / "general_td_model.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "model": td5_model,
            "target": "Td5_C",
            "fingerprint_bits": 2048,
            "metrics": td5_metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        model_dir / "general_td5_model.joblib",
        compress=3,
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "general_td": td_metrics,
        "general_td5": td5_metrics,
    }
    (model_dir / "public_thermal_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

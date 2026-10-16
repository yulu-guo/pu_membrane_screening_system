from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from pu_v2.features import MECHANICAL_FEATURES
from pu_v2.model import morgan_bit_vector


RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Tg and PU membrane mechanical-property models."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_tg(
    output_dir: Path,
    trees: int,
    random_state: int,
) -> dict:
    training_path = output_dir / "data" / "training" / "tg_polymers.csv"
    pu_path = output_dir / "data" / "training" / "pu_tg_structures.csv"
    frame = pd.read_csv(training_path)
    supplement_path = (
        output_dir / "data" / "training" / "tg_polymers_extended.csv"
    )
    if supplement_path.exists():
        # Newly sourced public Tg records (see build_tg_supplement.py). They are
        # already filtered to canonical structures missing from the base file.
        supplement = pd.read_csv(supplement_path)
        frame = pd.concat([frame, supplement], ignore_index=True)
    frame = frame.dropna(subset=["SMILES", "Tg"]).drop_duplicates("SMILES")
    polyurethane_classes = (
        frame["Polymer Class"]
        .fillna("")
        .str.contains("Polyurethane|Polyurea", case=False, regex=True)
        | frame["SMILES"].str.contains("NC(=O)O", regex=False)
    )
    frame["sample_weight"] = np.where(polyurethane_classes, 8.0, 1.0)

    feature_matrix = np.vstack(
        [morgan_bit_vector(smiles) for smiles in frame["SMILES"]]
    )
    targets = frame["Tg"].to_numpy(dtype=float)
    weights = frame["sample_weight"].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test, w_train, _w_test = train_test_split(
        feature_matrix,
        targets,
        weights,
        test_size=0.2,
        random_state=random_state,
        stratify=polyurethane_classes.astype(int),
    )
    evaluation_model = ExtraTreesRegressor(
        n_estimators=trees,
        max_features="sqrt",
        max_depth=35,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )
    evaluation_model.fit(x_train, y_train, sample_weight=w_train)
    test_metrics = regression_metrics(
        y_test,
        evaluation_model.predict(x_test),
    )

    pu_frame = pd.read_csv(pu_path).dropna(subset=["SMILES", "Tg"])
    pu_x = np.vstack(
        [morgan_bit_vector(smiles) for smiles in pu_frame["SMILES"]]
    )
    pu_y = pu_frame["Tg"].to_numpy(dtype=float)
    pu_predictions = evaluation_model.predict(pu_x)
    pu_metrics = regression_metrics(pu_y, pu_predictions)

    final_model = ExtraTreesRegressor(
        n_estimators=trees,
        max_features="sqrt",
        max_depth=35,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )
    final_model.fit(feature_matrix, targets, sample_weight=weights)

    bundle = {
        "model": final_model,
        "fingerprint_bits": 2048,
        "training_smiles": frame["SMILES"].tolist(),
        "metrics": {
            "holdout": test_metrics,
            "pu_subset_diagnostic": pu_metrics,
            "rows_total": len(frame),
            "rows_polyurethane": int(polyurethane_classes.sum()),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "NeurIPS Open Polymer Prediction 2025 Tg data",
    }
    path = output_dir / "data" / "models" / "tg_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return bundle["metrics"]


def train_mechanical(
    output_dir: Path,
    trees: int,
    random_state: int,
) -> dict:
    path = output_dir / "data" / "training" / "pue_643_mechanical.csv"
    frame = pd.read_csv(path)
    targets = ["logYM", "logTS", "logEB"]
    x = frame[MECHANICAL_FEATURES].to_numpy(dtype=float)
    y = frame[targets].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )
    metrics: dict[str, dict[str, float]] = {}
    models: dict[str, ExtraTreesRegressor] = {}
    for index, target in enumerate(targets):
        evaluation_model = ExtraTreesRegressor(
            n_estimators=trees,
            # Unrestricted depth and all features per split beat the previous
            # depth-25 / sqrt configuration on both raw R2 and relative error
            # for modulus and tensile strength; see the benchmark table in the
            # README. Fitting stays on log10 targets so the relative error that
            # a screening tool needs is preserved.
            max_features=1.0,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
        evaluation_model.fit(x_train, y_train[:, index])
        metrics[target] = regression_metrics(
            y_test[:, index],
            evaluation_model.predict(x_test),
        )

        final_model = ExtraTreesRegressor(
            n_estimators=trees,
            max_features=1.0,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
        final_model.fit(x, y[:, index])
        models[target] = final_model

    bundle = {
        "models": models,
        "feature_columns": MECHANICAL_FEATURES,
        "metrics": metrics,
        "rows": len(frame),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": (
            "Science Data Bank DOI 10.57760/sciencedb.14957; "
            "supplement to DOI 10.1007/s10118-022-2838-6"
        ),
    }
    model_path = output_dir / "data" / "models" / "mechanical_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    return metrics


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    tg_metrics = train_tg(output, args.trees, args.random_state)
    mechanical_metrics = train_mechanical(
        output,
        args.trees,
        args.random_state,
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tg": tg_metrics,
        "mechanical": mechanical_metrics,
    }
    metrics_path = output / "data" / "models" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

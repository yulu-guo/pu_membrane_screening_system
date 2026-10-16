from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from pu_v2.features import MECHANICAL_FEATURES


TARGETS = {
    "tg_c": "Tg_C",
    "youngs_modulus_mpa": "Youngs_modulus_MPa",
    "tensile_strength_mpa": "Tensile_strength_MPa",
    "elongation_at_break_percent": "Elongation_at_break_percent",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lab-calibration models from imported measured PU data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--min-samples", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    frame = pd.read_csv(output / "data" / "lab" / "lab_features.csv")
    base_columns = [
        "base_tg_c",
        "base_youngs_modulus_mpa",
        "base_tensile_strength_mpa",
        "base_elongation_at_break_percent",
    ]
    x = frame[base_columns + MECHANICAL_FEATURES].to_numpy(dtype=float)
    models = {}
    metrics = {}
    for normalized_name in TARGETS:
        if normalized_name not in frame.columns:
            continue
        mask = frame[normalized_name].notna().to_numpy()
        if int(mask.sum()) < args.min_samples:
            continue
        y = frame.loc[mask, normalized_name].to_numpy(dtype=float)
        x_available = x[mask]
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )
        if len(y) >= 12:
            predictions = cross_val_predict(
                model,
                x_available,
                y,
                cv=LeaveOneOut(),
                n_jobs=-1,
            )
            metrics[normalized_name] = {
                "rows": int(len(y)),
                "loo_r2": float(r2_score(y, predictions)),
                "loo_mae": float(mean_absolute_error(y, predictions)),
            }
        else:
            metrics[normalized_name] = {
                "rows": int(len(y)),
                "loo_r2": None,
                "loo_mae": None,
            }
        model.fit(x_available, y)
        models[normalized_name] = model

    if not models:
        raise SystemExit(
            f"No target had at least {args.min_samples} measured values."
        )
    bundle = {
        "models": models,
        "base_columns": base_columns,
        "feature_columns": MECHANICAL_FEATURES,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output / "data" / "models" / "lab_calibration.joblib"
    joblib.dump(bundle, path, compress=3)
    metadata_path = output / "data" / "models" / "lab_calibration.json"
    metadata_path.write_text(
        json.dumps(bundle["metrics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved calibration model to {path}")
    print(json.dumps(bundle["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

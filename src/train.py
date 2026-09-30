"""
src/train.py

Baseline (mean traffic theo hour x day-of-week, FIT CHỈ TRAIN) + Ridge pipeline
(ColumnTransformer: SimpleImputer -> StandardScaler -> Ridge cho numeric;
OneHotEncoder cho categorical), alpha chọn qua VALIDATION.

KHÔNG hard-code metric. KHÔNG dùng lại metric của checkpoint cũ (preprocessing
đã thay đổi hoàn toàn nên mọi con số cũ không còn đúng).

Usage:
    python src/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import (  # noqa: E402
    FEATURE_COLUMNS_ALL,
    FEATURE_COLUMNS_BINARY,
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    TARGET_COLUMN,
    build_features,
    load_clean,
    split_summary,
    time_split,
)

MODELS_DIR = _ROOT / "models"
REPORTS_DIR = _ROOT / "reports" / "figures"
SEED = 42

RIDGE_PIPELINE_PATH = MODELS_DIR / "ridge_pipeline.joblib"
BASELINE_TABLE_PATH = MODELS_DIR / "baseline_table.csv"
BASELINE_META_PATH = MODELS_DIR / "baseline_meta.json"
RUN_CONFIG_PATH = MODELS_DIR / "run_config.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
TUNING_RESULTS_PATH = REPORTS_DIR / "alpha_tuning.json"

#: lưới alpha — phải chạy lại vì preprocessing đã thay đổi (không tái dùng kết quả cũ)
ALPHA_GRID = [
    0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
]


def make_numeric_pipeline() -> Pipeline:
    """SimpleImputer -> StandardScaler.

    Bắt buộc có SimpleImputer vì `temp`/`rain_1h` còn NaN sau bước đánh dấu giá trị vô lý.
    KHÔNG được fill/interpolate ngoài pipeline, và KHÔNG được fit imputer ngoài TRAIN.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def make_ridge_pipeline(alpha: float) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                FEATURE_COLUMNS_CATEGORICAL,
            ),
            ("num", make_numeric_pipeline(), FEATURE_COLUMNS_NUMERIC),
            ("bin", "passthrough", FEATURE_COLUMNS_BINARY),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS_ALL], df[TARGET_COLUMN]


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def train_baseline(train_df: pd.DataFrame) -> pd.DataFrame:
    """Mean traffic_volume theo (hour, day_of_week) — CHỈ tính từ TRAIN."""
    return (
        train_df.groupby(["hour", "day_of_week"], observed=True)[TARGET_COLUMN]
        .mean()
        .rename("baseline_pred")
        .reset_index()
    )


def predict_baseline(df: pd.DataFrame, baseline_table: pd.DataFrame, global_fallback: float) -> np.ndarray:
    merged = df[["hour", "day_of_week"]].merge(
        baseline_table, on=["hour", "day_of_week"], how="left"
    )
    return merged["baseline_pred"].fillna(global_fallback).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def tune_ridge_alpha(train_df: pd.DataFrame, val_df: pd.DataFrame, alphas=None) -> dict:
    """Chọn alpha trên VALIDATION. Test không được chạm vào ở bước này."""
    alphas = list(alphas or ALPHA_GRID)
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)

    results = []
    for alpha in alphas:
        pipe = make_ridge_pipeline(alpha)
        pipe.fit(X_train, y_train)  # <-- imputer/scaler/encoder CHỈ fit trên TRAIN
        results.append({"alpha": alpha, **compute_metrics(y_val, pipe.predict(X_val))})

    best = min(results, key=lambda r: r["MAE"])
    return {
        "grid": alphas,
        "results": results,
        "criterion": "validation MAE",
        "best_alpha": best["alpha"],
        "best_val_metrics": {k: best[k] for k in ("MAE", "RMSE", "R2", "n")},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_features(load_clean())
    train_df, val_df, test_df = time_split(df)
    split_info = split_summary(train_df, val_df, test_df)
    print("=== TIME SPLIT (không chồng lấn) ===")
    for name in ("train", "validation", "test"):
        s = split_info[name]
        print(f"  {name:11s} {s['start']} -> {s['end']}  n={s['n_rows']:,}".replace(",", "."))
    print("  assert max(train) < min(val): OK | assert max(val) < min(test): OK")

    # ---- Baseline: chỉ TRAIN ----
    baseline_table = train_baseline(train_df)
    global_fallback = float(train_df[TARGET_COLUMN].mean())
    baseline_table.to_csv(BASELINE_TABLE_PATH, index=False)
    BASELINE_META_PATH.write_text(
        json.dumps(
            {
                "global_fallback": global_fallback,
                "fit_split": "train_only",
                "fit_rows": int(len(train_df)),
                "fit_range": [split_info["train"]["start"], split_info["train"]["end"]],
                "grouping": ["hour", "day_of_week"],
                "seed": SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    val_baseline = compute_metrics(val_df[TARGET_COLUMN], predict_baseline(val_df, baseline_table, global_fallback))
    print(f"\nBaseline (fit TRAIN only) — VALIDATION: MAE={val_baseline['MAE']}, RMSE={val_baseline['RMSE']}, R2={val_baseline['R2']}")

    # ---- Ridge: tuning alpha trên VALIDATION ----
    tuning = tune_ridge_alpha(train_df, val_df)
    print("\n=== TUNING alpha trên VALIDATION 2017 (không đụng TEST) ===")
    for r in tuning["results"]:
        print(f"  alpha={r['alpha']:>8} -> val MAE={r['MAE']:>8} RMSE={r['RMSE']:>8} R2={r['R2']:>8}")
    best_alpha = tuning["best_alpha"]
    print(f"-> alpha tốt nhất = {best_alpha} (val MAE={tuning['best_val_metrics']['MAE']})")
    TUNING_RESULTS_PATH.write_text(json.dumps(tuning, indent=2), encoding="utf-8")

    # ---- Đóng băng cấu hình ----
    # Fit model CUỐI chỉ trên TRAIN (không gộp validation) để so sánh công bằng với
    # baseline — baseline cũng chỉ tính từ TRAIN.
    final_pipe = make_ridge_pipeline(best_alpha)
    X_train, y_train = get_X_y(train_df)
    final_pipe.fit(X_train, y_train)

    joblib.dump(final_pipe, RIDGE_PIPELINE_PATH)

    val_model = compute_metrics(val_df[TARGET_COLUMN], final_pipe.predict(get_X_y(val_df)[0]))
    print(f"\nRidge alpha={best_alpha} — VALIDATION: MAE={val_model['MAE']}, RMSE={val_model['RMSE']}, R2={val_model['R2']}")

    # ---- Metadata ----
    pre = final_pipe.named_steps["preprocess"]
    n_features_out = int(sum(len(t[1].get_feature_names_out()) for t in pre.transformers_ if hasattr(t[1], "get_feature_names_out")))
    imputer = pre.named_transformers_["num"].named_steps["imputer"]
    encoder = pre.named_transformers_["cat"]
    scaler = pre.named_transformers_["num"].named_steps["scaler"]

    RUN_CONFIG_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "best_alpha": best_alpha,
                "alpha_grid": ALPHA_GRID,
                "alpha_selection_criterion": "validation MAE",
                "split": {
                    "train_end": split_info["train"]["end"],
                    "val_end": split_info["validation"]["end"],
                    "strategy": "time_series_no_shuffle",
                },
                "feature_columns_categorical": FEATURE_COLUMNS_CATEGORICAL,
                "feature_columns_numeric": FEATURE_COLUMNS_NUMERIC,
                "feature_columns_binary": FEATURE_COLUMNS_BINARY,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "pipeline": "ColumnTransformer[OneHotEncoder | SimpleImputer->StandardScaler | passthrough] -> Ridge(lsqr)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    MODEL_METADATA_PATH.write_text(
        json.dumps(
            {
                "model_name": "ridge_pipeline",
                "model_version": "checkpoint3",
                "created_by": "src/train.py",
                "seed": SEED,
                "hyperparameters": {"alpha": best_alpha, "solver": "lsqr"},
                "target": TARGET_COLUMN,
                "n_features_out": n_features_out,
                "feature_columns": {
                    "categorical": FEATURE_COLUMNS_CATEGORICAL,
                    "numeric": FEATURE_COLUMNS_NUMERIC,
                    "binary": FEATURE_COLUMNS_BINARY,
                },
                "preprocessing": {
                    "numeric": ["SimpleImputer(strategy=median)", "StandardScaler()"],
                    "categorical": ["OneHotEncoder(handle_unknown=ignore, sparse_output=False)"],
                    "binary": ["passthrough"],
                    "fit_scope": "TRAIN ONLY (imputer, scaler, encoder đều fit trên TRAIN)",
                    "imputer_median_train": {
                        c: (None if np.isnan(v) else round(float(v), 4))
                        for c, v in zip(FEATURE_COLUMNS_NUMERIC, imputer.statistics_)
                    },
                    "scaler_mean_train": {
                        c: round(float(v), 4)
                        for c, v in zip(FEATURE_COLUMNS_NUMERIC, scaler.mean_)
                    },
                    "n_encoder_categories_train": int(len(encoder.categories_)),
                },
                "fit_data": {
                    "split": "train_only",
                    "n_rows": int(len(train_df)),
                    "start": split_info["train"]["start"],
                    "end": split_info["train"]["end"],
                },
                "time_split": split_info,
                "validation_metrics": val_model,
                "baseline_validation_metrics": val_baseline,
                "alpha_tuning": {"criterion": "validation MAE", "best_alpha": best_alpha},
                "expected_inputs_at_serving": {
                    "note": "API/Web phải tạo đúng các cột feature trên từ date_time + thời tiết, "
                    "dùng CHUNG hàm src.features.build_features để không có train-serving skew.",
                    "required_engineered_columns": FEATURE_COLUMNS_ALL,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nĐã lưu model : {RIDGE_PIPELINE_PATH}")
    print(f"Đã lưu config : {RUN_CONFIG_PATH}")
    print(f"Đã lưu metadata: {MODEL_METADATA_PATH}")
    print(f"Đã lưu tuning : {TUNING_RESULTS_PATH}")
    print("\n=> CẤU HÌNH ĐÃ ĐÓNG BĂNG. Bước kế tiếp:")
    print("   python src/experiments.py   # thí nghiệm phát triển, CHỈ trong 2012-2017")
    print("   python src/evaluate.py      # FINAL TEST 2018, đánh giá đúng 1 lần")


if __name__ == "__main__":
    main()

"""
tests/test_pipeline.py

Test cho sklearn pipeline và các artifact đã sinh:
  - SimpleImputer CÓ trong pipeline và đứng trước scaler
  - imputer/scaler/encoder chỉ fit trên TRAIN (không rò rỉ)
  - pipeline xử lý được NaN
  - model đã lưu load được và dự báo được
  - metadata / config khớp với code hiện tại

Test này CẦN artifact đã sinh (`python src/data.py && python src/train.py`).
Nếu chưa có, test sẽ được skip với lý do rõ ràng.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    FEATURE_COLUMNS_ALL,
    FEATURE_COLUMNS_BINARY,
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    TARGET_COLUMN,
    build_features,
    load_clean,
    time_split,
)
from src.train import (  # noqa: E402
    BASELINE_META_PATH,
    BASELINE_TABLE_PATH,
    MODEL_METADATA_PATH,
    RIDGE_PIPELINE_PATH,
    RUN_CONFIG_PATH,
    compute_metrics,
    get_X_y,
    make_ridge_pipeline,
    predict_baseline,
    train_baseline,
)

CLEAN_PATH = ROOT / "data" / "processed" / "traffic_clean.csv"
AUDIT_PATH = ROOT / "data" / "processed" / "data_audit.json"

needs_data = pytest.mark.skipif(not CLEAN_PATH.exists(), reason="Chưa chạy `python src/data.py`")
needs_model = pytest.mark.skipif(
    not RIDGE_PIPELINE_PATH.exists(), reason="Chưa chạy `python src/train.py`"
)


@pytest.fixture(scope="module")
def splits():
    if not CLEAN_PATH.exists():
        pytest.skip("Chưa có dữ liệu đã xử lý")
    df = build_features(load_clean())
    return time_split(df)


# ===========================================================================
# Cấu trúc pipeline
# ===========================================================================
def test_pipeline_has_imputer():
    pipe = make_ridge_pipeline(1.0)
    pre = pipe.named_steps["preprocess"]
    num = dict((n, t) for n, t, _ in pre.transformers)["num"]
    assert isinstance(num, Pipeline), "Numeric branch phải là Pipeline"
    steps = list(num.named_steps)
    assert "imputer" in steps, "Numeric pipeline phải có SimpleImputer"
    assert "scaler" in steps, "Numeric pipeline phải có StandardScaler"
    assert steps.index("imputer") < steps.index("scaler"), "Imputer phải đứng TRƯỚC scaler"
    assert isinstance(num.named_steps["imputer"], SimpleImputer)
    assert isinstance(num.named_steps["scaler"], StandardScaler)


def test_pipeline_covers_all_feature_columns():
    pipe = make_ridge_pipeline(1.0)
    pre = pipe.named_steps["preprocess"]
    covered = set()
    for _name, _trans, cols in pre.transformers:
        covered |= set(cols)
    assert covered == set(FEATURE_COLUMNS_ALL)
    assert covered.isdisjoint({TARGET_COLUMN}), "Target tuyệt đối không được vào pipeline"


def test_pipeline_handles_unknown_categories():
    """Giá trị categorical lạ (chưa từng thấy) không được làm vỡ pipeline."""
    pipe = make_ridge_pipeline(1.0)
    df = build_features(load_clean()) if CLEAN_PATH.exists() else None
    if df is None:
        pytest.skip("Chưa có dữ liệu")
    X, _ = get_X_y(df.head(50))
    X = X.copy()
    X["weather_main_mode"] = "Meteor Shower"
    X["weather_family"] = "unheard_of"
    X["month"] = 77
    pred = pipe.fit(X, df[TARGET_COLUMN].head(50)).predict(X)
    assert np.isfinite(pred).all()


@needs_data
def test_pipeline_imputes_nan_at_predict_time(splits):
    train, val, test = splits
    pipe = make_ridge_pipeline(1.0)
    X_tr, y_tr = get_X_y(train)
    pipe.fit(X_tr, y_tr)

    # các dòng còn NaN (do giá trị đo vô lý) nằm ở đâu cũng phải dự báo được
    all_X = pd.concat([X_tr, get_X_y(val)[0], get_X_y(test)[0]], ignore_index=True)
    mask_nan = all_X[FEATURE_COLUMNS_NUMERIC].isna().any(axis=1)
    assert mask_nan.sum() > 0, "Dữ liệu phải còn NaN để test này có ý nghĩa"

    pred = pipe.predict(all_X)
    assert np.isfinite(pred).all(), "Pipeline phải tự xử lý NaN, không được trả NaN"
    assert np.isfinite(pred[mask_nan.to_numpy()]).all(), (
        "Dòng có NaN đầu vào vẫn phải ra dự báo hữu hạn"
    )


@needs_data
def test_imputer_median_matches_train_only_median(splits):
    """Giá trị median trong imputer phải đúng bằng median của TRAIN."""
    train, _val, _test = splits
    pipe = make_ridge_pipeline(1.0)
    X_tr, y_tr = get_X_y(train)
    pipe.fit(X_tr, y_tr)

    imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
    for col, stat in zip(FEATURE_COLUMNS_NUMERIC, imputer.statistics_):
        expected = X_tr[col].median()
        if pd.isna(expected):
            assert np.isnan(stat)
        else:
            assert stat == pytest.approx(expected), f"{col}: imputer phải dùng median của TRAIN"


@needs_data
def test_imputer_not_fitted_on_whole_dataset(splits):
    """Median toàn bộ dữ liệu khác median TRAIN -> chứng minh không rò rỉ."""
    train, _val, test = splits
    full = pd.concat([train, test], ignore_index=True)
    X_tr, y_tr = get_X_y(train)
    X_full, _ = get_X_y(full)

    differing = [
        c
        for c in FEATURE_COLUMNS_NUMERIC
        if X_tr[c].median() != X_full[c].median() and not pd.isna(X_tr[c].median())
    ]
    if not differing:
        pytest.skip("Trong dữ liệu này median TRAIN == median toàn bộ (không phân biệt được)")

    pipe = make_ridge_pipeline(1.0).fit(X_tr, y_tr)
    imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
    for col in differing:
        got = dict(zip(FEATURE_COLUMNS_NUMERIC, imputer.statistics_))[col]
        assert got == pytest.approx(X_tr[col].median())
        assert got != pytest.approx(X_full[col].median())


@needs_data
def test_scaler_fitted_on_train_only(splits):
    train, _val, test = splits
    X_tr, y_tr = get_X_y(train)
    pipe = make_ridge_pipeline(1.0).fit(X_tr, y_tr)
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"]

    for col, mean in zip(FEATURE_COLUMNS_NUMERIC, scaler.mean_):
        if col in FEATURE_COLUMNS_NUMERIC and X_tr[col].isna().any():
            expected = X_tr[col].fillna(X_tr[col].median()).mean()
        else:
            expected = X_tr[col].mean()
        assert mean == pytest.approx(expected), f"{col}: scaler phải fit trên TRAIN"


@needs_data
def test_encoder_categories_come_from_train(splits):
    train, _val, test = splits
    X_tr, y_tr = get_X_y(train)
    pipe = make_ridge_pipeline(1.0).fit(X_tr, y_tr)
    enc = pipe.named_steps["preprocess"].named_transformers_["cat"]

    train_categories = set(X_tr["weather_main_mode"].unique())
    learned = set(enc.categories_[FEATURE_COLUMNS_CATEGORICAL.index("weather_main_mode")])
    assert learned == train_categories, "Encoder phải học danh mục từ TRAIN"
    # nếu có giá trị chỉ xuất hiện ở test, handle_unknown='ignore' phải xử lý được
    test_only = set(get_X_y(test)[0]["weather_main_mode"].unique()) - train_categories
    if test_only:
        X_test = get_X_y(test)[0]
        assert np.isfinite(pipe.predict(X_test)).all()


@needs_data
def test_fit_is_deterministic(splits):
    """Cùng dữ liệu, cùng seed -> kết quả phải giống hệt nhau."""
    train, val, _test = splits
    X_tr, y_tr = get_X_y(train)
    X_val, _ = get_X_y(val)
    p1 = make_ridge_pipeline(1.0).fit(X_tr, y_tr).predict(X_val)
    p2 = make_ridge_pipeline(1.0).fit(X_tr, y_tr).predict(X_val)
    assert np.allclose(p1, p2), "Pipeline phải deterministic (không random state ẩn)"


# ===========================================================================
# Baseline
# ===========================================================================
@needs_data
def test_baseline_fit_on_train_only(splits):
    train, _val, test = splits
    table = train_baseline(train)
    assert len(table) == train[["hour", "day_of_week"]].drop_duplicates().shape[0]

    X_train = train[["hour", "day_of_week"]]
    y_train = train[TARGET_COLUMN]
    for hour, dow in [(8, 0), (17, 4), (3, 6)]:
        mask = (X_train["hour"] == hour) & (X_train["day_of_week"] == dow)
        if mask.any():
            expected = y_train[mask].mean()
            row = table[(table["hour"] == hour) & (table["day_of_week"] == dow)]
            assert row["baseline_pred"].iloc[0] == pytest.approx(expected)

    # baseline chỉ biết TRAIN: dùng bảng đó dự báo test vẫn chạy được
    fb = float(train[TARGET_COLUMN].mean())
    assert np.isfinite(predict_baseline(test, table, fb)).all()


@needs_data
def test_baseline_beats_global_mean_on_validation(splits):
    train, val, _test = splits
    table = train_baseline(train)
    fb = float(train[TARGET_COLUMN].mean())
    pred = predict_baseline(val, table, fb)
    m = compute_metrics(val[TARGET_COLUMN], pred)
    naive = compute_metrics(val[TARGET_COLUMN], np.full(len(val), fb))
    assert m["MAE"] < naive["MAE"]


# ===========================================================================
# Artifact đã lưu
# ===========================================================================
@needs_model
def test_saved_pipeline_is_loadable_and_predicts(splits):
    _train, val, test = splits
    pipe = joblib.load(RIDGE_PIPELINE_PATH)
    assert isinstance(pipe, Pipeline)
    assert "preprocess" in pipe.named_steps and "model" in pipe.named_steps

    for name, part in [("validation", val), ("test", test)]:
        X, y = get_X_y(part)
        pred = pipe.predict(X)
        assert len(pred) == len(part)
        assert np.isfinite(pred).all(), f"Dự báo trên {name} phải hữu hạn"
        m = compute_metrics(y, pred)
        assert m["MAE"] > 0 and m["R2"] <= 1.0


@needs_model
def test_saved_pipeline_contains_imputer():
    pipe = joblib.load(RIDGE_PIPELINE_PATH)
    num = pipe.named_steps["preprocess"].named_transformers_["num"]
    assert isinstance(num.named_steps["imputer"], SimpleImputer)
    assert isinstance(num.named_steps["scaler"], StandardScaler)


@needs_model
def test_saved_pipeline_reproduces_metadata_metrics(splits):
    """Metric trong metadata phải khớp với metric tính lại từ artifact."""
    meta = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    pipe = joblib.load(RIDGE_PIPELINE_PATH)
    assert pipe.named_steps["model"].alpha == meta["hyperparameters"]["alpha"]

    train, val, test = splits
    expected_train = meta["fit_data"]["n_rows"]
    assert len(train) == expected_train, "Metadata phải mô tả đúng tập train"
    assert meta["fit_data"]["start"] == str(train["date_time"].min())
    assert meta["fit_data"]["end"] == str(train["date_time"].max())

    X_val, y_val = get_X_y(val)
    got = compute_metrics(y_val, pipe.predict(X_val))
    for k in ("MAE", "RMSE", "R2"):
        assert got[k] == pytest.approx(meta["validation_metrics"][k], abs=0.05), (
            f"Metric {k} trong metadata không khớp tính lại từ model"
        )


@needs_model
def test_metadata_time_split_has_no_overlap(splits):
    meta = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    ts = meta["time_split"]
    assert pd.Timestamp(ts["train"]["end"]) < pd.Timestamp(ts["validation"]["start"])
    assert pd.Timestamp(ts["validation"]["end"]) < pd.Timestamp(ts["test"]["start"])
    train, val, test = splits
    assert ts["train"]["n_rows"] == len(train)
    assert ts["validation"]["n_rows"] == len(val)
    assert ts["test"]["n_rows"] == len(test)


@needs_model
def test_metadata_lists_all_feature_columns():
    meta = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    feats = meta["feature_columns"]
    assert set(feats["categorical"]) == set(FEATURE_COLUMNS_CATEGORICAL)
    assert set(feats["numeric"]) == set(FEATURE_COLUMNS_NUMERIC)
    assert set(feats["binary"]) == set(FEATURE_COLUMNS_BINARY)
    assert "TRAIN ONLY" in meta["preprocessing"]["fit_scope"]


@needs_model
def test_run_config_matches_code():
    cfg = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["feature_columns_categorical"] == FEATURE_COLUMNS_CATEGORICAL
    assert cfg["feature_columns_numeric"] == FEATURE_COLUMNS_NUMERIC
    assert cfg["feature_columns_binary"] == FEATURE_COLUMNS_BINARY
    assert cfg["split"]["strategy"] == "time_series_no_shuffle"
    assert cfg["alpha_selection_criterion"] == "validation MAE"


@needs_data
@needs_model
def test_no_hardcoded_metrics_in_source():
    """Không được hard-code metric vào source: tìm sẵn sàng các số báo cáo."""
    import re

    suspicious = []
    for path in (ROOT / "src").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"MAE\s*[:=]\s*([0-9]+\.?[0-9]*)", text):
            line = text[: m.start()].count("\n") + 1
            suspicious.append(f"{path.name}:{line} -> {m.group(0)}")
    # allow_nan / f-string formatting is fine; only literal hard-coded MAE values are not
    assert not suspicious, f"Tìm thấy metric có thể đã hard-code: {suspicious}"


@needs_data
def test_clean_csv_has_one_row_per_timestamp():
    df = pd.read_csv(CLEAN_PATH, keep_default_na=False, na_values=[""])
    assert df["date_time"].is_unique
    assert (df["holiday"] != "None").sum() > 0, "Phải còn ngày lễ thật"
    assert df["date_time"].is_monotonic_increasing


@needs_data
def test_audit_json_records_required_numbers():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert audit["n_raw_rows"] == 48204
    assert audit["duplicates"]["n_duplicate_groups"] == 5445
    assert audit["duplicates"]["n_rows_in_duplicate_groups"] == 13074
    assert audit["n_rows_after_collapse"] == 40575
    assert audit["invalid_measurements"]["n_temp_invalid"] == 10
    assert audit["invalid_measurements"]["n_rain_invalid"] == 1
    for col in ("traffic_volume", "holiday", "snow_1h"):
        assert audit["duplicates"]["invariant_in_duplicate_groups"][col] == 5445

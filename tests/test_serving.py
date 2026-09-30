"""
tests/test_serving.py

Test cho tầng serving — phần dễ sai nhất (train-serving skew).

Bao gồm:
  - Chuyển đổi °C -> K
  - `is_holiday` tự tính đúng từ lịch (không đọc dataset, không cần người dùng nhập)
  - Multi-weather -> multi-hot + nhãn đại diện + severity đúng
  - **Không train-serving skew**: dựng feature từ một dòng thô trong dataset phải
    cho ra ĐÚNG BẰNG feature mà `src.features.build_features` sinh ra trên dataset.
  - Chính sách max(0, ·) và kiểm tra số hữu hạn
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.serving import (  # noqa: E402
    WEATHER_MAIN_TO_FAMILY,
    build_scope,
    celsius_to_kelvin,
    engineer_features,
    predict,
    weather_family_for,
)
from src.features import FEATURE_COLUMNS_ALL, build_features  # noqa: E402
from src.holidays import holiday_name, is_holiday as is_holiday_day  # noqa: E402
from src.weather import WEATHER_MAIN_CATEGORIES, severity_of_weather_main  # noqa: E402

from conftest import requires_artifacts  # noqa: E402

CLEAN_CSV = config.data_dir() / "traffic_clean.csv"


def _payload(**overrides):
    base = {
        "date_time": datetime(2018, 6, 15, 8, 0, 0),
        "temperature_celsius": 20.0,
        "rain_1h_mm": 0.0,
        "snow_1h_mm": 0.0,
        "clouds_all": 20,
        "weather": ["Clear"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Đơn vị nhiệt độ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "celsius, kelvin",
    [(0.0, 273.15), (20.0, 293.15), (-10.5, 262.65), (37.0, 310.15), (-40.0, 233.15)],
)
def test_celsius_to_kelvin(celsius, kelvin):
    assert celsius_to_kelvin(celsius) == pytest.approx(kelvin)


def test_feature_temp_uses_kelvin():
    f = engineer_features(_payload(temperature_celsius=20.0))
    assert f["temp"] == pytest.approx(293.15)


# ---------------------------------------------------------------------------
# 2. is_holiday tự tính từ lịch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "day, expected_flag",
    [
        (date(2018, 7, 4), 1),   # Independence Day
        (date(2018, 11, 22), 1), # Thanksgiving
        (date(2018, 12, 25), 1), # Christmas
        (date(2018, 8, 23), 1),  # Minnesota State Fair
        (date(2018, 6, 15), 0),  # ngày thường
        (date(2018, 6, 16), 0),  # cuối tuần nhưng KHÔNG phải ngày lễ
    ],
)
def test_holiday_computed_from_calendar_not_input(day, expected_flag):
    """Người dùng KHÔNG nhập is_holiday — backend tự tính, kết quả khớp lịch."""
    f = engineer_features(_payload(date_time=datetime(day.year, day.month, day.day, 9, 0, 0)))
    assert f["is_holiday"] == expected_flag
    assert f["_derived"]["holiday_name"] == holiday_name(day)
    assert f["is_holiday"] == is_holiday_day(day)


def test_holiday_flag_applies_to_whole_day_not_only_midnight():
    """Cột `holiday` gốc chỉ có ở 00:00; feature phải đúng ở MỌI giờ của ngày lễ."""
    for hour in (0, 7, 13, 23):
        f = engineer_features(
            _payload(date_time=datetime(2018, 7, 4, hour, 0, 0))
        )
        assert f["is_holiday"] == 1, f"sai ở giờ {hour}"


# ---------------------------------------------------------------------------
# 3. Multi-weather (multi-label / multi-hot)
# ---------------------------------------------------------------------------
def test_multi_weather_multi_hot_correct():
    f = engineer_features(_payload(weather=["Rain", "Snow", "Thunderstorm"]))
    assert f["wm_rain"] == 1
    assert f["wm_snow"] == 1
    assert f["wm_thunderstorm"] == 1
    assert f["wm_clear"] == 0
    assert f["wm_fog"] == 0
    # severity = mức nghiêm trọng nhất trong giờ đó
    assert f["weather_severity"] == max(
        severity_of_weather_main(w) for w in ("Rain", "Snow", "Thunderstorm")
    )
    # nhãn đại diện deterministic (theo số lần xuất hiện, hoà -> alphabet)
    assert f["weather_main_mode"] == "Rain"
    assert f["_derived"]["weather_main_set"] == "Rain; Snow; Thunderstorm"
    assert f["_derived"]["weather_main_n"] == 3
    # cờ thời tiết cực đoan
    assert f["_derived"]["is_extreme_weather"] == 1


def test_duplicate_weather_values_are_deduplicated():
    a = engineer_features(_payload(weather=["Rain", "Rain", "Snow"]))
    b = engineer_features(_payload(weather=["Snow", "Rain"]))
    for col in FEATURE_COLUMNS_ALL:
        assert a[col] == b[col], f"khác ở {col}"


def test_weather_family_mapping_covers_all_categories():
    assert set(WEATHER_MAIN_TO_FAMILY) == set(WEATHER_MAIN_CATEGORIES)
    assert weather_family_for(["Clear"]) == "clear"
    assert weather_family_for(["Rain", "Snow"]) == "snow"  # snow nghiêm trọng hơn rain
    assert weather_family_for(["Clear", "Thunderstorm"]) == "thunder"


def test_single_weather_uses_same_path_as_training():
    f = engineer_features(_payload(weather=["Clouds"]))
    assert f["wm_clouds"] == 1
    assert f["weather_severity"] == severity_of_weather_main("Clouds")
    assert f["weather_family"] == WEATHER_MAIN_TO_FAMILY["Clouds"]


# ---------------------------------------------------------------------------
# 4. KHÔNG train-serving skew — bằng chứng quan trọng nhất
# ---------------------------------------------------------------------------
@requires_artifacts
@pytest.mark.skipif(not CLEAN_CSV.exists(), reason="Cần data/processed/traffic_clean.csv")
def test_no_train_serving_skew_on_real_rows():
    """Dựng feature từ dữ liệu thô phải GIỐNG HỆT feature lúc huấn luyện.

    Nếu đường dẫn serving lệch với đường dẫn train ở bất kỳ cột nào, test này đỏ.
    """
    from app.serving import build_serving_frame

    trained = build_features(pd.read_csv(CLEAN_CSV, parse_dates=["date_time"]))
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["date_time"])
    clean = clean.set_index("date_time", drop=False)

    # chọn một ngày lễ và một ngày thường để phủ cả hai nhánh is_holiday
    for ts in ["2018-07-04 08:00:00", "2018-06-15 08:00:00", "2017-11-23 17:00:00"]:
        ts = pd.Timestamp(ts)
        assert ts in clean.index, f"thiếu timestamp mẫu {ts}"
        row = clean.loc[ts]
        weather_values = sorted({str(v) for v in str(row["weather_main_set"]).split("; ") if v})

        frame = build_serving_frame(
            date_time=ts.to_pydatetime(),
            temperature_celsius=float(row["temp"]) - config.KELVIN_OFFSET,
            rain_1h_mm=float(row["rain_1h"]) if pd.notna(row["rain_1h"]) else 0.0,
            snow_1h_mm=float(row["snow_1h"]) if pd.notna(row["snow_1h"]) else 0.0,
            clouds_all=int(row["clouds_all"]),
            weather=weather_values,
        )
        served = engineer_features(
            {
                "date_time": ts.to_pydatetime(),
                "temperature_celsius": float(row["temp"]) - config.KELVIN_OFFSET,
                "rain_1h_mm": float(row["rain_1h"]) if pd.notna(row["rain_1h"]) else 0.0,
                "snow_1h_mm": float(row["snow_1h"]) if pd.notna(row["snow_1h"]) else 0.0,
                "clouds_all": int(row["clouds_all"]),
                "weather": weather_values,
            }
        )
        reference = trained.loc[trained["date_time"] == ts].iloc[0]
        for col in FEATURE_COLUMNS_ALL:
            got, want = served[col], reference[col]
            if isinstance(want, str) or isinstance(got, str):
                assert got == want, f"{ts} — lệch ở {col}: serving={got!r} train={want!r}"
            else:
                assert got == pytest.approx(float(want), abs=1e-6), (
                    f"{ts} — lệch ở {col}: serving={got!r} train={want!r}"
                )


@requires_artifacts
def test_serving_uses_the_same_feature_columns_as_training(artifacts):
    """Cột đưa vào pipeline lúc serving phải đúng bằng lúc huấn luyện."""
    features = engineer_features(_payload())
    features.pop("_derived")
    assert sorted(features) == sorted(FEATURE_COLUMNS_ALL)
    expected = artifacts.metadata["expected_inputs_at_serving"]["required_engineered_columns"]
    assert sorted(expected) == sorted(FEATURE_COLUMNS_ALL)


# ---------------------------------------------------------------------------
# 5. Dự báo: hữu hạn, không âm, chính sách hậu xử lý
# ---------------------------------------------------------------------------
@requires_artifacts
def test_prediction_is_finite_and_non_negative(artifacts):
    for hour, weather, temp in [
        (3, ["Clear"], -15.0),
        (8, ["Rain"], 5.0),
        (17, ["Snow", "Thunderstorm"], -8.0),
        (12, ["Fog"], 2.0),
    ]:
        out = predict(
            artifacts,
            _payload(
                date_time=datetime(2018, 2, 14, hour, 0, 0),
                temperature_celsius=temp,
                weather=weather,
            ),
        )
        assert math.isfinite(out["predicted_traffic_volume"])
        assert out["predicted_traffic_volume"] >= 0.0
        assert math.isfinite(out["raw_model_output"])


@requires_artifacts
def test_clipped_to_zero_is_reported_consistently(artifacts):
    out = predict(artifacts, _payload())
    assert out["serving_policy"]["policy_id"] in ("non_negative_projection", "none")
    assert "max(0" in out["postprocess_policy"] or "raw_ridge" in out["postprocess_policy"]
    expected_clip = out["raw_model_output"] < 0.0
    assert out["clipped_to_zero"] is expected_clip
    if expected_clip:
        assert out["predicted_traffic_volume"] == 0.0
    else:
        assert out["predicted_traffic_volume"] == pytest.approx(
            round(out["raw_model_output"], 2), abs=0.01
        )


@requires_artifacts
def test_prediction_never_crashes_on_extreme_but_valid_input(artifacts):
    """Input biên nhưng hợp lệ vẫn phải ra số hữu hạn."""
    out = predict(
        artifacts,
        _payload(
            date_time=datetime(2018, 1, 15, 2, 0, 0),
            temperature_celsius=config.TEMPERATURE_CELSIUS_MIN,
            rain_1h_mm=0.0,
            snow_1h_mm=config.PRECIPITATION_MM_MAX,
            clouds_all=100,
            weather=["Snow", "Thunderstorm", "Fog", "Rain", "Drizzle", "Mist"],
        ),
    )
    assert math.isfinite(out["predicted_traffic_volume"])


# ---------------------------------------------------------------------------
# 6. Cảnh báo phạm vi — không đoán ngầm
# ---------------------------------------------------------------------------
@requires_artifacts
def test_warning_when_outside_dataset_range(artifacts):
    out = predict(artifacts, _payload(date_time=datetime(2025, 6, 15, 8, 0, 0)))
    assert out["scope"]["in_dataset_range"] is False
    assert any("phạm vi" in w for w in out["warnings"])


@requires_artifacts
def test_warning_when_state_fair_calendar_unknown(artifacts):
    """Ngoài phạm vi năm của bảng lịch State Fair -> phải CẢNH BÁO, không tự đoán."""
    out = predict(artifacts, _payload(date_time=datetime(2030, 6, 15, 8, 0, 0)))
    assert out["scope"]["state_fair_calendar_known"] is False
    assert any("State Fair" in w for w in out["warnings"])


@requires_artifacts
def test_explicit_state_fair_override_sets_holiday(artifacts):
    """Người dùng TỰ CUNG CẤP ngày State Fair -> hệ thống dùng, không phải đoán."""
    out = predict(
        artifacts,
        _payload(
            date_time=datetime(2030, 8, 22, 10, 0, 0),
            state_fair_start_date=date(2030, 8, 22),
        ),
    )
    assert out["holiday"]["is_holiday"] == 1
    assert out["holiday"]["holiday_name"] == "State Fair"


@requires_artifacts
def test_no_warning_inside_known_range(artifacts):
    out = predict(artifacts, _payload())
    assert out["warnings"] == []
    assert out["scope"]["in_dataset_range"] is True


# ---------------------------------------------------------------------------
# 5b. Chính sách max(0, ·) — số đo phải khớp artifact kiểm toán
# ---------------------------------------------------------------------------
POSTPROCESS_AUDIT_PATH = config.reports_dir() / "postprocess_audit.json"


@requires_artifacts
@pytest.mark.skipif(not POSTPROCESS_AUDIT_PATH.exists(), reason="Cần chạy py src\\postprocess_audit.py")
def test_postprocess_audit_numbers_match_report(artifacts):
    """Số trong báo cáo về chính sách max(0,·) phải khớp artifact kiểm toán."""
    audit = json.loads(POSTPROCESS_AUDIT_PATH.read_text(encoding="utf-8"))
    test_split = next(
        s for s in audit["splits"] if s["split"].startswith("FINAL TEST")
    )
    # metric THÔ trong artifact kiểm toán phải bằng metric FINAL TEST chính thức
    official = artifacts.evaluation["final_test"]["ridge"]
    assert test_split["n_rows"] == official["n"]
    for key in ("MAE", "RMSE", "R2"):
        assert test_split["metrics_raw"][key] == official[key], (
            f"{key} thô trong kiểm toán khác FINAL TEST chính thức"
        )
    # và phải có thực sự dự báo âm (chứng minh policy không phải hình thức)
    assert test_split["n_raw_negative"] > 0
    assert test_split["min_raw_prediction"] < 0
    # không có NaN/inf
    assert test_split["any_nan_or_inf"] is False

    report = (config.ROOT_DIR / "reports" / "final_report.md").read_text(encoding="utf-8")

    def vi(v, d=2):
        s = f"{v:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")
        return s

    def in_report(s: str) -> bool:
        # báo cáo dùng dấu trừ Unicode (−) ở nhiều chỗ
        return s in report or s.replace("-", "−") in report

    assert in_report(vi(test_split["n_raw_negative"], 0)), (
        f"báo cáo thiếu số dòng dự báo âm: {test_split['n_negative_raw']}"
    )
    assert in_report(vi(test_split["min_raw_prediction"]))
    assert in_report(vi(test_split["metrics_deployed"]["MAE"]))


@requires_artifacts
def test_api_reports_clipped_flag_for_known_negative_case(artifacts):
    """Có input tạo ra dự báo âm thì API phải báo clipped_to_zero = true và trả 0."""
    # đêm 25/12/2018 là đêm lễ — vùng mô hình dự báo âm
    out = predict(
        artifacts,
        _payload(date_time=datetime(2018, 12, 25, 2, 0, 0), temperature_celsius=5.0, weather=["Clear"]),
    )
    assert out["raw_model_output"] < 0, "input này được kỳ vọng tạo dự báo thô âm"
    assert out["clipped_to_zero"] is True
    assert out["predicted_traffic_volume"] == 0.0
    assert out["holiday"]["is_holiday"] == 1


def test_build_scope_reports_dataset_window(artifacts):
    scope, warnings = build_scope(artifacts, datetime(2018, 6, 15, 8, 0, 0))
    assert scope["dataset_coverage_start"].startswith("2012-10-02")
    assert scope["dataset_coverage_end"].startswith("2018-09-30")
    assert scope["unit"] == config.TARGET_UNIT_VI
    assert warnings == []

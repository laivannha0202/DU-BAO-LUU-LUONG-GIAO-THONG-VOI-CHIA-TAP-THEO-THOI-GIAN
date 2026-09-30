"""
app/serving.py

Tầng "serving": biến JSON của người dùng -> feature -> dự báo.

BA NGUYÊN TẮC BẤT DI BẤT DỊCH
1. Dùng CHUNG `src.features.build_features` với lúc huấn luyện
   -> không thể có train-serving skew.
2. CHỈ predict trên pipeline đã đóng băng. Không fit, không tune, không đọc target.
3. Không sửa âm thầm input sai (validation đã loại trước ở tầng Pydantic);
   mọi cảnh báo về phạm vi đều được TRẢ VỀ cho người dùng dưới dạng `warnings`.
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.registry import Artifacts

_ROOT = config.ROOT_DIR
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import FEATURE_COLUMNS_ALL, build_features  # noqa: E402
from src.holidays import (  # noqa: E402
    HOLIDAY_NONE,
    KNOWN_YEARS,
    holiday_name,
    is_holiday as is_holiday_day,
    is_year_known,
)
from src.weather import (  # noqa: E402
    WEATHER_MAIN_CATEGORIES,
    WEATHER_FAMILY_SEVERITY,
    aggregate_weather_group,
    severity_of_weather_main,
)

STATE_FAIR_HOLIDAY_NAME = "State Fair"

#: Tên thứ trong tuần (0 = Thứ 2) — chỉ để hiển thị, không ảnh hưởng mô hình
DAY_OF_WEEK_NAMES: tuple[str, ...] = (
    "Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật",
)


class ServingError(RuntimeError):
    """Lỗi khi dự báo (không phải lỗi input) — sẽ trả HTTP 500 kèm giải thích."""


# ---------------------------------------------------------------------------
# 1. Đơn vị nhiệt độ
# ---------------------------------------------------------------------------
def celsius_to_kelvin(celsius: float) -> float:
    """°C -> K. Đây là phép biến đổi DUY NHẤT backend dùng cho nhiệt độ."""
    return float(celsius) + config.KELVIN_OFFSET


# ---------------------------------------------------------------------------
# 2. Ánh xạ weather_main -> weather_family lúc serving
# ---------------------------------------------------------------------------
# Lúc huấn luyện, `weather_family` được suy ra từ CỐT `weather_description`
# (xem src/weather.classify_weather_description). Ở thời điểm dự báo, API chỉ
# nhận `weather_main` (mức khái quát mà bảng điều khiển giao thông thực tế có
# sẵn). Vì vậy cần một quy tắc TẤT ĐỊNH nối main -> family.
#
# Quy tắc này là hằng số do nhóm định nghĩa, KHÔNG học từ dữ liệu, KHÔNG fit gì.
# Nhiều hiện tượng cùng lúc -> lấy family nghiêm trọng nhất, đúng như
# `src.weather.aggregate_weather_description_group` đang làm khi collapse dữ liệu.
WEATHER_MAIN_TO_FAMILY: dict[str, str] = {
    "Clear": "clear",
    "Clouds": "partly_cloudy",
    "Drizzle": "rain",
    "Fog": "fog",
    "Haze": "haze_smoke",
    "Mist": "mist",
    "Rain": "rain",
    "Smoke": "haze_smoke",
    "Snow": "snow",
    "Squall": "thunder",
    "Thunderstorm": "thunder",
}
assert set(WEATHER_MAIN_TO_FAMILY) == set(WEATHER_MAIN_CATEGORIES)


def weather_family_for(weather_main: list[str]) -> str:
    families = sorted({WEATHER_MAIN_TO_FAMILY[m] for m in weather_main})
    return max(families, key=lambda f: (WEATHER_FAMILY_SEVERITY[f], f))


# ---------------------------------------------------------------------------
# 3. Dựng khung feature dùng CHUNG với huấn luyện
# ---------------------------------------------------------------------------
def build_serving_frame(
    *,
    date_time: datetime,
    temperature_celsius: float,
    rain_1h_mm: float,
    snow_1h_mm: float,
    clouds_all: int,
    weather: list[str],
) -> pd.DataFrame:
    """Dựng DataFrame 1 dòng ở đúng định dạng dữ liệu sạch, rồi gọi
    `src.features.build_features` — hàm DUY NHẤT dùng lúc train.

    `build_features` tự lo: is_holiday (từ lịch tất định), hour, day_of_week,
    month, hour_dow, is_weekend, cờ thời tiết cực đoan.
    """
    weather_sorted = sorted(set(weather))
    agg = aggregate_weather_group(weather_sorted)

    frame = pd.DataFrame(
        [
            {
                "date_time": pd.Timestamp(date_time),
                # `holiday` gốc KHÔNG được dùng để tính is_holiday
                # (xong src/features.py) — chỉ đặt sentinel cho đúng schema.
                "holiday": HOLIDAY_NONE,
                "temp": celsius_to_kelvin(temperature_celsius),
                "rain_1h": float(rain_1h_mm),
                "snow_1h": float(snow_1h_mm),
                "clouds_all": int(clouds_all),
                "weather_main_mode": agg["weather_main_mode"],
                "weather_main_set": agg["weather_main_set"],
                "weather_main_n": agg["weather_main_n"],
                "weather_severity": agg["weather_severity"],
                "weather_family": weather_family_for(weather_sorted),
                **{col: agg[col] for col in agg if col.startswith("wm_")},
            }
        ]
    )
    return frame


def engineer_features(payload: dict[str, Any]) -> dict[str, Any]:
    """payload (dict thô) -> dict feature cuối cùng mà pipeline nhận.

    Hàm thuần (không I/O) -> test được dễ, và chính là hàm mà endpoint dùng.
    """
    weather = sorted(set(payload["weather"]))
    raw_frame = build_serving_frame(
        date_time=payload["date_time"],
        temperature_celsius=payload["temperature_celsius"],
        rain_1h_mm=payload.get("rain_1h_mm", 0.0),
        snow_1h_mm=payload.get("snow_1h_mm", 0.0),
        clouds_all=payload["clouds_all"],
        weather=weather,
    )
    featured = build_features(raw_frame)
    row = featured.iloc[0]

    fair_override: date | None = payload.get("state_fair_start_date")
    if fair_override is not None and not is_year_known(fair_override.year):
        # Người dùng CỐ TÌNH cung cấp ngày khai mạc State Fair cho năm chưa có
        # trong bảng lịch. Đây là dữ liệu do người dùng đưa vào một cách tường minh,
        # không phải backend đoán.
        if row["date_time"].date() == fair_override:
            row["is_holiday"] = 1
            row["holiday_name"] = STATE_FAIR_HOLIDAY_NAME

    features = {col: _jsonable(row[col]) for col in FEATURE_COLUMNS_ALL}
    features["_derived"] = {
        "hour": int(row["hour"]),
        "day_of_week": int(row["day_of_week"]),
        "month": int(row["month"]),
        "is_weekend": int(row["is_weekend"]),
        "is_extreme_weather": int(row["is_extreme_weather"]),
        "weather_main_set": row["weather_main_set"],
        "weather_main_n": int(row["weather_main_n"]),
        "holiday_name": str(row["holiday_name"]),
    }
    return features


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(float(value), 4)
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


# ---------------------------------------------------------------------------
# 4. Phạm vi sử dụng & cảnh báo trung thực
# ---------------------------------------------------------------------------
def _parse_ts(text: str | None):
    if not text:
        return None
    try:
        return pd.Timestamp(text)
    except Exception:  # noqa: BLE001
        return None


def build_scope(art: Artifacts, date_time: datetime) -> tuple[dict[str, Any], list[str]]:
    """Thông tin phạm vi + DANH SÁCH CẢNH BÁO. Không bao giờ đoán ngầm."""
    warnings: list[str] = []
    coverage = art.coverage
    start = _parse_ts(coverage.get("start"))
    end = _parse_ts(coverage.get("end"))
    ts = pd.Timestamp(date_time)
    in_range = bool(start is not None and end is not None and start <= ts <= end)

    if not in_range:
        warnings.append(
            f"Ngoài phạm vi quan sát của dataset ({coverage.get('start')} → {coverage.get('end')}). "
            "Mô hình vẫn chạy nhưng độ tin cậy ngoài phạm vi này KHÔNG được đảm bảo — "
            "xem mục Hạn chế trong Model Card."
        )

    year_known = is_year_known(date_time.year)
    if not year_known:
        warnings.append(
            f"Năm {date_time.year} KHÔNG nằm trong bảng lịch State Fair "
            f"({min(KNOWN_YEARS)}–{max(KNOWN_YEARS)}) của src/holidays.py. "
            "Backend KHÔNG tự đoán ngày khai mạc: 10 ngày lễ liên bang vẫn tính đúng, "
            "nhưng State Fair có thể bị bỏ sót. "
            "Muốn tính cả State Fair, hãy truyền thêm `state_fair_start_date`."
        )

    scope = {
        "dataset_coverage_start": coverage.get("start"),
        "dataset_coverage_end": coverage.get("end"),
        "in_dataset_range": in_range,
        "state_fair_calendar_known": year_known,
        "state_fair_known_years": [min(KNOWN_YEARS), max(KNOWN_YEARS)],
        "model_trained_on": "2012-10-02 .. 2016-12-31 (TRAIN)",
        "model_validated_on": "2017 (VALIDATION)",
        "final_test": "2018-01-01 .. 2018-09-30 (FINAL TEST, không dùng để chọn mô hình)",
        "unit": config.TARGET_UNIT_VI,
    }
    return scope, warnings


# ---------------------------------------------------------------------------
# 5. Dự báo
# ---------------------------------------------------------------------------
def predict(art: Artifacts, payload: dict[str, Any]) -> dict[str, Any]:
    """Trả về dict khớp `app.schemas.ForecastResponse`."""
    date_time: datetime = payload["date_time"]
    weather = sorted(set(payload["weather"]))

    features = engineer_features(payload)
    derived = features.pop("_derived")
    missing = [c for c in FEATURE_COLUMNS_ALL if c not in features]
    if missing:  # pragma: no cover - phòng thủ
        raise ServingError(f"Không tạo được feature: {missing}")

    X = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS_ALL}])
    try:
        raw = float(np.asarray(art.pipeline.predict(X)).ravel()[0])
    except Exception as exc:  # noqa: BLE001
        raise ServingError(f"Pipeline dự báo lỗi: {exc}") from exc

    if not math.isfinite(raw):
        raise ServingError(
            "Mô hình trả về giá trị không hữu hạn (NaN/inf) — có thể do artifact hỏng. "
            "Kiểm tra lại bằng `python src/train.py`."
        )

    # --- RAW MODEL -> DEPLOYED PREDICTOR -----------------------------------
    # RAW MODEL        = Ridge(alpha, lsqr)          -> raw_model_output
    # DEPLOYED PREDICTOR = max(0, ·) ∘ Ridge        -> predicted_traffic_volume
    #
    # Chính sách được ĐÓNG BĂNG trên TRAIN + VALIDATION bởi
    # `src/freeze_serving_policy.py` (xem `models/serving_policy.json`).
    # Nó KHÔNG phải siêu tham số, không được chọn bằng FINAL TEST, và
    # căn cứ mạnh nhất là lập luận toán học + miền giá trị của `traffic_volume`.
    project_to_floor = art.apply_non_negative_projection()
    clipped = raw < 0.0 and project_to_floor
    final = max(0.0, raw) if project_to_floor else raw

    policy = art.serving_policy_info()

    scope, warnings = build_scope(art, date_time)
    day: date = date_time.date()

    model_meta = art.metadata
    warnings = list(warnings)
    if not policy.get("policy_artifact_present", False):
        warnings.append(
            "Thiếu artifact `models/serving_policy.json` — đang dùng mặc định "
            "`max(0, ·)`. Chạy `python src/freeze_serving_policy.py` để sinh policy "
            "đã đóng băng và nạp lại server."
        )
    return {
        "predicted_traffic_volume": round(final, 2),
        "unit": config.TARGET_UNIT_EN,
        "unit_vi": config.TARGET_UNIT_VI,
        "predictor": {
            "raw_model": "Ridge(alpha=%s, solver=%s)" % (
                art.alpha, (model_meta.get("hyperparameters") or {}).get("solver")
            ),
            "raw_model_output": round(raw, 4),
            "deployed_predictor": (
                "max(0, Ridge) — phép chiếu về sàn miền giá trị, KHÔNG phải mô hình khác"
                if project_to_floor
                else "Ridge (không hậu xử lý) — negative prediction được ghi là limitation"
            ),
            "deployed_prediction": round(final, 2),
            "note": (
                "`predicted_traffic_volume` là giá trị của DEPLOYED PREDICTOR. "
                "Metric của mô hình trong báo cáo là giá trị của RAW MODEL "
                "(tính trên `raw_model_output`)."
            ),
        },
        "raw_model_output": round(raw, 4),
        "clipped_to_zero": clipped,
        "postprocess_policy": policy.get("rule", config.NON_NEGATIVE_POLICY_ID),
        "serving_policy": policy,
        "model": {
            "name": str(model_meta.get("model_name", "ridge_pipeline")),
            "version": model_meta.get("model_version"),
            "alpha": art.alpha,
            "solver": (model_meta.get("hyperparameters") or {}).get("solver"),
            "target": config.TARGET_NAME,
            "unit": config.TARGET_UNIT_EN,
            "unit_vi": config.TARGET_UNIT_VI,
            "n_features_out": model_meta.get("n_features_out"),
            "artifact": "models/ridge_pipeline.joblib",
        },
        "input_echo": {
            "date_time": date_time.isoformat(),
            "temperature_celsius": float(payload["temperature_celsius"]),
            "rain_1h_mm": float(payload.get("rain_1h_mm", 0.0)),
            "snow_1h_mm": float(payload.get("snow_1h_mm", 0.0)),
            "clouds_all": int(payload["clouds_all"]),
            "weather": weather,
        },
        "derived_features": features,
        "calendar": {
            "hour": int(derived["hour"]),
            "day_of_week": int(derived["day_of_week"]),
            "day_of_week_name": DAY_OF_WEEK_NAMES[int(derived["day_of_week"])],
            "month": int(derived["month"]),
            "is_weekend": bool(derived["is_weekend"]),
            "is_extreme_weather": bool(derived["is_extreme_weather"]),
            "hour_dow": features["hour_dow"],
            "weather_main_set": derived["weather_main_set"],
            "weather_main_n": int(derived["weather_main_n"]),
            "source": "src/features.py (hàm thuần của date_time)",
        },
        "holiday": {
            "is_holiday": int(features["is_holiday"]),
            "holiday_name": derived["holiday_name"],
            "source": "src/holidays.py (lịch tất định, không đọc dataset)",
            "is_holiday_recomputed": int(is_holiday_day(day)),
            "state_fair_calendar_known": is_year_known(date_time.year),
        },
        "weather": {
            "requested": weather,
            "weather_main_mode": features["weather_main_mode"],
            "weather_family": features["weather_family"],
            "weather_severity": int(features["weather_severity"]),
            "max_severity_in_hour": max(severity_of_weather_main(w) for w in weather),
            "multi_hot": {k: int(v) for k, v in features.items() if k.startswith("wm_")},
            "encoding": "multi-hot (nhiều hiện tượng cùng lúc được giữ đủ)",
        },
        "scope": scope,
        "warnings": warnings,
    }

__all__ = [
    "ServingError",
    "WEATHER_MAIN_TO_FAMILY",
    "build_scope",
    "build_serving_frame",
    "celsius_to_kelvin",
    "engineer_features",
    "predict",
    "weather_family_for",
]

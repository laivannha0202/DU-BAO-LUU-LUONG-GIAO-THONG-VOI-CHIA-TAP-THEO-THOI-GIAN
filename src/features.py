"""
src/features.py

Feature engineering + time-based split cho Metro Interstate Traffic Volume.

Nguyên tắc chống rò rỉ (bất biến xuyên suốt dự án):
- Mọi phép biến đổi ở module này là QUY TẮC TẤT ĐỊNH, không học thống kê từ dữ liệu.
  Vì vậy chạy trên TRAIN / VALIDATION / TEST / dữ liệu API đều cho cùng kết quả
  -> không có train-serving skew.
- KHÔNG có `interpolate`, KHÔNG có `drop_duplicates(keep='first')` ở đây.
- Thống kê học được (imputer, scaler, encoder) CHỈ nằm trong sklearn Pipeline của
  `src/train.py` và CHỈ fit trên TRAIN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.holidays import HOLIDAY_NONE, holiday_name, is_holiday  # noqa: E402
from src.weather import WEATHER_MAIN_CATEGORIES, weather_main_columns  # noqa: E402

# ---------------------------------------------------------------------------
# Hằng số ngữ nghĩa
# ---------------------------------------------------------------------------
#: giá trị cột `holiday` mang nghĩa "KHÔNG phải ngày lễ" (không phải missing)
#: (nhập lại từ src.holidays để giữ đúng một nguồn sự thật)

#: mốc thời gian chia tập
TRAIN_END = "2016-12-31 23:59:59"  # train: 2012-01-01 .. 2016-12-31
VAL_END = "2017-12-31 23:59:59"  # val:   2017-01-01 .. 2017-12-31
# test: 2018-01-01 .. hết (không dùng để chọn mô hình)

TARGET_COLUMN = "traffic_volume"

# --- danh mục feature ---
FEATURE_COLUMNS_CATEGORICAL = ["hour_dow", "month", "weather_main_mode", "weather_family"]
FEATURE_COLUMNS_NUMERIC = ["temp", "rain_1h", "snow_1h", "clouds_all", "weather_severity"]
FEATURE_COLUMNS_BINARY = ["is_holiday", *weather_main_columns()]
FEATURE_COLUMNS_ALL = FEATURE_COLUMNS_CATEGORICAL + FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_BINARY

#: cột bị đoạn trùng / giá trị vô lý đã xử lý ở src/data.py — dùng cho kiểm tra
OUTLIER_HANDLED_COLUMNS = ["temp", "rain_1h"]

#: nhóm thời tiết cực đoan dùng cho phân tích lỗi (mỗi timestamp có thể thuộc nhiều)
EXTREME_WEATHER_CATEGORIES = ["Snow", "Thunderstorm", "Squall"]


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """Nạp dữ liệu ĐÃ qua src/data.py (đã collapse + đánh dấu giá trị vô lý).

    Cố tình KHÔNG tự collapse/dedup ở đây: nếu không, sẽ tồn tại hai nguồn sự thật
    khác nhau về cách xử lý trùng lặp giữa data.py và train.py.
    """
    if path is None:
        path = _ROOT / "data" / "processed" / "traffic_clean.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}.\n"
            "Chạy `python src/data.py` trước — đó là bước tạo dữ liệu đã collapse."
        )
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["date_time"] = pd.to_datetime(df["date_time"])
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume", "weather_severity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in weather_main_columns():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df.sort_values("date_time").reset_index(drop=True)


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Suy ra cờ `is_holiday` cho CẢ NGÀY lịch từ LỊCH TẤT ĐỊNH (`src/holidays.py`).

    Vì sao KHÔNG nhóm theo ngày trên chính bảng dữ liệu:
      1. Cột `holiday` chỉ ghi tên ở giờ 00:00, nên phải suy diễn mới đúng cho cả ngày.
      2. Suy diễn kiểu đó đòi hỏi phải có sẵn toàn bộ bảng dữ liệu. Ở thời điểm dự báo
         (Web/API) ta chỉ có MỘT ngày — không thể quét các dòng khác. Đó là train-serving skew.
      3. Cách đó còn có thể rò rỉ: để biết ngày X có phải lễ không, ta đã phải "nhìn" dữ liệu
         của chính ngày X.

    `src/holidays.py` tính ngày lễ từ ngày tháng bằng quy tắc lịch, nên:
      - tái tạo được với bất kỳ ngày nào, kể cả ngoài phạm vi dataset;
      - không cần đọc dữ liệu nào;
      - cho cùng kết quả trên TRAIN / VALIDATION / TEST / API / Web.

    Đối chiếu với dataset: lịch khớp 53/53 ngày lễ mà dataset ghi nhận, bỏ sót 0 ngày
    (xem `reports/figures/data_quality_report.md`).

    ⚠️ Điều kiện để dùng hợp lệ: lịch ngày lễ là thông tin CÔNG CỘNG, biết trước tại
    thời điểm dự báo — ta luôn biết 4/7 là lễ trước khi nó tới. Vì vậy đây KHÔNG phải rò rỉ.

    ⚠️ Dùng lịch, KHÔNG dùng `.notna()`: cột được đọc bằng `keep_default_na=False` nên
    `"None"` là chuỗi hợp lệ; `.notna().any()` sẽ trả True cho MỌI ngày.
    """
    df = df.copy()
    df["holiday"] = df["holiday"].astype("string").fillna(HOLIDAY_NONE)

    # Lịch tất định: thuần tuần theo NGÀY, không phụ thuộc bất kỳ dòng dữ liệu nào
    days = df["date_time"].dt.date
    df["is_holiday"] = [is_holiday(d) for d in days]
    # Cột phục vụ báo cáo/hiển thị: tên lễ tính được từ lịch
    df["holiday_name"] = [holiday_name(d) for d in days]
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng lịch — đều là hàm thuần của `date_time`, không học gì từ dữ liệu."""
    df = df.copy()
    dt = df["date_time"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0 = Monday
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Tương tác hour x day_of_week — đúng độ chi tiết mà baseline dùng, để Ridge học
    # được cùng effect đó cộng thêm weather/month.
    df["hour_dow"] = dt.dt.hour.astype(str) + "_" + dt.dt.dayofweek.astype(str)
    return df


def add_extreme_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Cờ thời tiết cực đoan, suy ra từ multi-hot (một giờ có thể có nhiều hiện tượng)."""
    df = df.copy()
    for cat in EXTREME_WEATHER_CATEGORIES:
        df[f"is_extreme_{cat.lower()}"] = df[f"wm_{cat.lower()}"].astype(int)
    df["is_extreme_weather"] = df[
        [f"is_extreme_{c.lower()}" for c in EXTREME_WEATHER_CATEGORIES]
    ].max(axis=1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dữ liệu sạch -> khung feature hoàn chỉnh. Không học thống kê, không nội suy."""
    df = df.sort_values("date_time").reset_index(drop=True)
    df = add_holiday_flag(df)
    df = add_calendar_features(df)
    df = add_extreme_weather_flags(df)

    missing = [c for c in FEATURE_COLUMNS_ALL if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu feature column: {missing}")
    return df


def _describe_split(name: str, part: pd.DataFrame) -> dict:
    return {
        "name": name,
        "n_rows": int(len(part)),
        "start": str(part["date_time"].min()),
        "end": str(part["date_time"].max()),
        "n_years": sorted(part["year"].unique().tolist()),
    }


def time_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia theo THỜI GIAN, không xáo trộn. Có assert chặn overlap.

    train = 2012-01-01 .. 2016-12-31
    val   = 2017-01-01 .. 2017-12-31
    test  = 2018-01-01 .. hết
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = df[df["date_time"] <= train_end_ts].reset_index(drop=True)
    val = df[(df["date_time"] > train_end_ts) & (df["date_time"] <= val_end_ts)].reset_index(drop=True)
    test = df[df["date_time"] > val_end_ts].reset_index(drop=True)

    assert_time_split_disjoint(train, val, test)
    return train, val, test


def assert_time_split_disjoint(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Chặn cứng: max(train) < min(val) và max(val) < min(test), không timestamp trùng."""
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        if len(part) == 0:
            raise AssertionError(f"Tập `{name}` rỗng — kiểm tra lại mốc thời gian.")

    max_train, min_val = train["date_time"].max(), val["date_time"].min()
    max_val, min_test = val["date_time"].max(), test["date_time"].min()

    if not max_train < min_val:
        raise AssertionError(f"Train/Validation bị chồng: max(train)={max_train} >= min(val)={min_val}")
    if not max_val < min_test:
        raise AssertionError(f"Validation/Test bị chồng: max(val)={max_val} >= min(test)={min_test}")

    sets = {
        "train": set(train["date_time"]),
        "validation": set(val["date_time"]),
        "test": set(test["date_time"]),
    }
    for a, b in [("train", "validation"), ("validation", "test"), ("train", "test")]:
        overlap = sets[a] & sets[b]
        if overlap:
            raise AssertionError(f"{a} & {b} chung {len(overlap)} timestamp.")

    return {
        "train": _describe_split("train", train),
        "validation": _describe_split("validation", val),
        "test": _describe_split("test", test),
        "assertions": {
            "max_train_lt_min_validation": True,
            "max_validation_lt_min_test": True,
            "no_shared_timestamps": True,
        },
    }


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    return assert_time_split_disjoint(train, val, test)


def random_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CHỈ dùng để MINH HỌA mức độ lạc quan do rò rễ — KHÔNG dùng để kết luận chính thức."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = shuffled.iloc[:n_train].reset_index(drop=True)
    val = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = shuffled.iloc[n_train + n_val :].reset_index(drop=True)
    return train, val, test


__all__ = [
    "HOLIDAY_NONE",
    "TARGET_COLUMN",
    "FEATURE_COLUMNS_ALL",
    "FEATURE_COLUMNS_BINARY",
    "FEATURE_COLUMNS_CATEGORICAL",
    "FEATURE_COLUMNS_NUMERIC",
    "EXTREME_WEATHER_CATEGORIES",
    "WEATHER_MAIN_CATEGORIES",
    "add_calendar_features",
    "add_extreme_weather_flags",
    "add_holiday_flag",
    "assert_time_split_disjoint",
    "build_features",
    "load_clean",
    "random_split",
    "split_summary",
    "time_split",
    "weather_main_columns",
]

"""
tests/test_features.py

Test cho src/features.py: ngữ nghĩa holiday, đặc trưng lịch, cờ thời tiết cực đoan,
và time split không chồng lấn.

Các test không phụ thuộc file dữ liệu thật: chúng dựng dataframe nhỏ trong bộ nhớ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    FEATURE_COLUMNS_ALL,
    FEATURE_COLUMNS_BINARY,
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    HOLIDAY_NONE,
    TARGET_COLUMN,
    add_calendar_features,
    add_extreme_weather_flags,
    add_holiday_flag,
    build_features,
    time_split,
)
from src.weather import weather_main_columns  # noqa: E402


def make_frame(start: str, n_hours: int, holiday: str = HOLIDAY_NONE, name_only_at_midnight: bool = False) -> pd.DataFrame:
    """Dựng khung dữ liệu giả.

    `name_only_at_midnight=True` mô phỏng đúng hành vi của dataset thật: cột `holiday`
    chỉ ghi tên ở giờ 00:00, các giờ còn lại của cùng ngày mang chuỗi "None".
    """
    ts = pd.date_range(start, periods=n_hours, freq="h")
    holidays = [holiday] * n_hours
    if name_only_at_midnight:
        holidays = [holiday if t.hour == 0 else HOLIDAY_NONE for t in ts]
    return pd.DataFrame(
        {
            "date_time": ts,
            "holiday": holidays,
            "temp": 280.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "traffic_volume": 3000,
            "weather_main_mode": "Clear",
            "weather_main_set": "Clear",
            "weather_main_n": 1,
            "weather_desc_mode": "sky is clear",
            "weather_desc_set": "sky is clear",
            "weather_desc_n": 1,
            "weather_family": "clear",
            "weather_severity": 0,
            "n_weather_obs": 1,
            **{c: 0 for c in weather_main_columns()},
        }
    )


# ===========================================================================
# holiday semantics
# ===========================================================================
def test_holiday_flag_uses_calendar_day_not_row_value():
    """Lịch tất định đánh dấu cả ngày, kể cả các giờ mà cột gốc chỉ ghi "None"."""
    # 2013-01-01 là Thứ Ba -> New Year's Day không bị dời, dùng ngày này cho rõ ràng
    df = make_frame("2013-01-01 00:00:00", 24, holiday="New Years Day", name_only_at_midnight=True)
    out = add_holiday_flag(df)
    # chỉ dòng 00:00 mang tên, 23 giờ còn lại là "None"
    assert (out["holiday"] == HOLIDAY_NONE).sum() == 23
    assert (out["holiday"] != HOLIDAY_NONE).sum() == 1
    assert out["is_holiday"].eq(1).all(), "Cả 24 giờ của ngày lễ phải được đánh dấu"
    assert out["holiday_name"].eq("New Years Day").all()


def test_holiday_none_string_day_is_not_holiday():
    df = make_frame("2013-01-02 00:00:00", 24, holiday=HOLIDAY_NONE)
    out = add_holiday_flag(df)
    assert out["is_holiday"].eq(0).all()


def test_holiday_flag_ignores_holiday_column_entirely():
    """YÊU CẦU 3.1 #4: cờ phải đến từ LỊCH, không từ cột `holiday`.

    Nếu ta gán tên ngày lễ sai cho một ngày thường, cờ vẫn phải là 0 — vì lịch tất định
    là nguồn sự thật, và đó chính là điều cho phép Web/API tái tạo feature khi
    không có bất kỳ dòng dữ liệu nào.
    """
    df = make_frame("2013-01-02 00:00:00", 24, holiday="New Years Day", name_only_at_midnight=True)
    out = add_holiday_flag(df)
    assert (out["holiday"] != HOLIDAY_NONE).sum() == 1, "fixture cố tình gán tên sai"
    assert out["is_holiday"].eq(0).all(), "Cờ phải theo LỊCH, không theo cột dữ liệu"


def test_holiday_observed_day_shifting():
    """Quy ước 'observed day': 01/01/2017 là Chủ nhật -> lễ dời sang 02/01/2017."""
    from src.holidays import is_holiday
    from datetime import date

    # 01/01/2017 là Chủ nhật -> lễ được QUAN SÁT vào 02/01, ngày 01/01 không còn là ngày lễ
    # (đúng như dataset đã ghi nhận: nhãn "New Years Day" nằm ở 2017-01-02)
    assert is_holiday(date(2017, 1, 1)) == 0
    assert is_holiday(date(2017, 1, 2)) == 1
    assert is_holiday(date(2017, 1, 3)) == 0
    # 04/07/2015 là thứ Bảy -> lễ được quan sát vào 03/07
    assert is_holiday(date(2015, 7, 3)) == 1
    assert is_holiday(date(2015, 7, 4)) == 0
    assert is_holiday(date(2015, 7, 5)) == 0
    # 25/12/2016 là Chủ nhật -> dời sang 26/12
    assert is_holiday(date(2016, 12, 26)) == 1
    assert is_holiday(date(2016, 12, 24)) == 0
    # 2014: 04/07 rơi vào thứ Sáu -> giữ nguyên
    assert is_holiday(date(2014, 7, 4)) == 1


def test_holiday_calendar_rule_dates():
    """Các ngày lễ tính bằng quy tắc phải đúng với lịch thật."""
    from src.holidays import holiday_name
    from datetime import date

    cases = {
        date(2013, 1, 21): "Martin Luther King Jr Day",   # thứ Hai thứ 3 tháng 1
        date(2013, 2, 18): "Washingtons Birthday",        # thứ Hai thứ 3 tháng 2
        date(2013, 5, 27): "Memorial Day",                # thứ Hai cuối tháng 5
        date(2013, 9, 2): "Labor Day",                    # thứ Hai thứ nhất tháng 9
        date(2013, 10, 14): "Columbus Day",                # thứ Hai thứ 2 tháng 10
        date(2013, 11, 28): "Thanksgiving Day",           # thứ Năm thứ 4 tháng 11
        date(2013, 11, 11): "Veterans Day",
        date(2013, 8, 22): "State Fair",                  # từ bảng công bố
        date(2013, 7, 4): "Independence Day",
        date(2013, 12, 25): "Christmas Day",
    }
    for d, expected in cases.items():
        assert holiday_name(d) == expected, f"{d} -> {holiday_name(d)}, cần {expected}"


def test_notna_would_be_catastrophically_wrong():
    """Chứng minh tại sao `.notna()` là cái bẫy: nó đánh dấu MỌI ngày."""
    df = make_frame("2013-01-02 00:00:00", 24, holiday=HOLIDAY_NONE)
    by_day_notna = df.groupby(df["date_time"].dt.normalize())["holiday"].apply(
        lambda s: s.notna().any()
    )
    by_day_correct = df.groupby(df["date_time"].dt.normalize())["holiday"].apply(
        lambda s: (s != HOLIDAY_NONE).any()
    )
    assert by_day_notna.all(), "notna() coi mọi ngày là ngày lễ"
    assert not by_day_correct.any(), "Cách đúng: không ngày nào là ngày lễ"


def test_holiday_flag_mixed_days():
    d1 = make_frame("2013-01-01 00:00:00", 24, holiday="New Years Day", name_only_at_midnight=True)
    d2 = make_frame("2013-01-02 00:00:00", 24, holiday=HOLIDAY_NONE)
    out = add_holiday_flag(pd.concat([d1, d2], ignore_index=True))
    assert out.loc[:23, "is_holiday"].eq(1).all()
    assert out.loc[24:, "is_holiday"].eq(0).all()


# ===========================================================================
# Calendar features
# ===========================================================================
def test_calendar_features():
    df = make_frame("2017-03-06 00:00:00", 24 * 8)  # đủ để tới 2017-03-11 (thứ Bảy)
    out = add_calendar_features(df)
    assert out["date_time"].dt.hour.iloc[0] == 0
    assert out["day_of_week"].iloc[0] == 0, "Thứ Hai phải là 0"
    assert out["month"].iloc[0] == 3
    assert out["year"].iloc[0] == 2017
    assert out["is_weekend"].iloc[0] == 0
    # 2017-03-11 là thứ Bảy -> hour_dow = "12_5"
    sat = out[out["date_time"] == pd.Timestamp("2017-03-11 12:00:00")].iloc[0]
    assert sat["hour_dow"] == "12_5"
    assert sat["is_weekend"] == 1


def test_calendar_features_are_row_local():
    """Đặc trưng lịch là hàm thuần của date_time -> không rò rỉ giữa các tập."""
    df = make_frame("2016-05-10 00:00:00", 48)
    full = add_calendar_features(df)
    part = add_calendar_features(df.iloc[10:14].copy())
    for col in ["hour", "day_of_week", "month", "year", "is_weekend", "hour_dow"]:
        assert full[col].iloc[10:14].tolist() == part[col].tolist()


# ===========================================================================
# Extreme weather flags
# ===========================================================================
def test_extreme_weather_flags_are_multi_label():
    df = make_frame("2018-01-05 00:00:00", 2)
    df.loc[0, "wm_snow"] = 1
    df.loc[0, "wm_thunderstorm"] = 1
    df.loc[1, "wm_squall"] = 1
    out = add_extreme_weather_flags(df)
    assert out.loc[0, "is_extreme_weather"] == 1
    assert out.loc[0, "is_extreme_snow"] == 1
    assert out.loc[0, "is_extreme_thunderstorm"] == 1
    assert out.loc[1, "is_extreme_weather"] == 1
    assert out.loc[1, "is_extreme_snow"] == 0


# ===========================================================================
# build_features
# ===========================================================================
def test_build_features_produces_all_required_columns():
    df = make_frame("2016-01-04 00:00:00", 48, holiday="New Years Day")
    df.loc[5, "temp"] = float("nan")
    out = build_features(df)
    missing = [c for c in FEATURE_COLUMNS_ALL if c not in out.columns]
    assert missing == [], f"Thiếu feature: {missing}"
    assert TARGET_COLUMN in out.columns
    # NaN phải được giữ nguyên, KHÔNG bị interpolate
    assert out["temp"].isna().sum() == 1


def test_build_features_does_not_impute():
    """build_features KHÔNG được điền NaN — việc đó thuộc SimpleImputer trong pipeline."""
    df = make_frame("2016-01-04 00:00:00", 10)
    df.loc[2, "rain_1h"] = float("nan")
    out = build_features(df)
    assert out["rain_1h"].isna().sum() == 1


def test_feature_column_groups_are_disjoint_and_complete():
    all_cols = FEATURE_COLUMNS_CATEGORICAL + FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_BINARY
    assert sorted(all_cols) == sorted(FEATURE_COLUMNS_ALL)
    assert len(set(all_cols)) == len(all_cols), "Không được trùng cột giữa các nhóm"
    assert TARGET_COLUMN not in FEATURE_COLUMNS_ALL, "Target không được làm feature"
    for c in FEATURE_COLUMNS_BINARY:
        assert c in ["is_holiday"] or c.startswith("wm_")


# ===========================================================================
# Time split
# ===========================================================================
def test_time_split_boundaries():
    df = build_features(
        pd.concat(
            [
                make_frame("2015-12-31 20:00:00", 10),   # sang 2016
                make_frame("2016-12-31 20:00:00", 10),   # sang 2017
                make_frame("2017-12-31 20:00:00", 10),   # sang 2018
                make_frame("2018-06-01 00:00:00", 10),
            ],
            ignore_index=True,
        )
    )
    train, val, test = time_split(df)

    assert train["date_time"].max() == pd.Timestamp("2016-12-31 23:00:00")
    assert val["date_time"].min() == pd.Timestamp("2017-01-01 00:00:00")
    assert val["date_time"].max() == pd.Timestamp("2017-12-31 23:00:00")
    assert test["date_time"].min() == pd.Timestamp("2018-01-01 00:00:00")
    assert len(train) + len(val) + len(test) == len(df)


def test_time_split_has_no_overlap():
    df = build_features(
        pd.concat(
            [
                make_frame("2014-01-01 00:00:00", 24 * 40),
                make_frame("2017-01-01 00:00:00", 24 * 40),
                make_frame("2018-01-01 00:00:00", 24 * 40),
            ],
            ignore_index=True,
        )
    )
    train, val, test = time_split(df)

    assert max(train["date_time"]) < min(val["date_time"])
    assert max(val["date_time"]) < min(test["date_time"])
    assert set(train["date_time"]).isdisjoint(val["date_time"])
    assert set(val["date_time"]).isdisjoint(test["date_time"])
    assert set(train["date_time"]).isdisjoint(test["date_time"])


def test_time_split_is_not_shuffled():
    df = build_features(
        pd.concat(
            [make_frame("2015-01-01 00:00:00", 24 * 40), make_frame("2017-01-01 00:00:00", 24 * 40), make_frame("2018-01-01 00:00:00", 100)],
            ignore_index=True,
        )
    )
    train, val, test = time_split(df)
    assert len(val) > 0 and len(test) > 0
    for part in (train, val, test):
        assert part["date_time"].is_monotonic_increasing, "Time split không được xáo trộn"


def test_time_split_raises_on_empty_split():
    df = build_features(make_frame("2016-01-01 00:00:00", 100))
    with pytest.raises(AssertionError, match="rỗng"):
        time_split(df)


def test_time_split_raises_on_overlap():
    """Nếu ranh giới bị cấu hình sai, assert phải chặn được."""
    df = build_features(
        pd.concat(
            [make_frame("2015-01-01 00:00:00", 24 * 40), make_frame("2017-01-01 00:00:00", 24 * 40), make_frame("2018-01-01 00:00:00", 100)],
            ignore_index=True,
        )
    )
    # truyền mốc sai: val lấy tới 2016 -> tập validation rỗng
    with pytest.raises(AssertionError):
        time_split(df, train_end="2016-12-31 23:59:59", val_end="2016-12-31 23:59:59")


def test_time_split_custom_boundaries_do_not_overlap():
    df = build_features(
        pd.concat(
            [
                make_frame("2012-01-01 00:00:00", 24 * 366),
                make_frame("2013-01-01 00:00:00", 24 * 365),
                make_frame("2014-01-01 00:00:00", 24 * 365),
            ],
            ignore_index=True,
        )
    )
    train, val, test = time_split(df, train_end="2012-12-31 23:59:59", val_end="2013-12-31 23:59:59")
    assert train["date_time"].max() < val["date_time"].min() < test["date_time"].min()
    assert train["date_time"].max() == pd.Timestamp("2012-12-31 23:00:00")
    assert val["date_time"].max() == pd.Timestamp("2013-12-31 23:00:00")
    assert test["date_time"].min() == pd.Timestamp("2014-01-01 00:00:00")

"""
tests/test_data.py

Unit test cho src/data.py và src/weather.py, dùng fixture tổng hợp
(tests/fixtures/sample_traffic.csv) để chạy độc lập với dữ liệu thật.

Fixture cố tình mô phỏng đúng các lỗi đã audit trên dataset thật:
  - cột `holiday` chứa chuỗi "None" (không phải NaN)
  - tên ngày lễ chỉ nằm ở giờ 00:00 của ngày đó
  - timestamp trùng với nhiều hiện tượng/mô tả thời tiết khác nhau
  - `temp == 0` K và `rain_1h == 9831.3` (giá trị vô lý)
  - một giờ bị THIẾU hẳn để kiểm tra phát hiện khoảng trống

Chạy: pytest tests/test_data.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    EXPECTED_COLUMNS,
    INVARIANT_COLUMNS,
    RAIN_1H_SENTINEL_INVALID,
    TEMP_K_INVALID_MAX,
    assert_timestamp_invariants,
    audit_duplicates,
    audit_holiday,
    audit_holiday_calendar,
    build_clean_dataset,
    clean_and_report,
    collapse_duplicates,
    flag_invalid_measurements,
    load_raw,
    validate_schema,
)
from src.weather import (  # noqa: E402
    WEATHER_MAIN_CATEGORIES,
    aggregate_weather_group,
    classify_weather_description,
    deterministic_mode,
    weather_main_columns,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_traffic.csv"


def load_fixture() -> pd.DataFrame:
    return load_raw(FIXTURE)


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return build_clean_dataset(load_fixture())[0]


# ===========================================================================
# Schema
# ===========================================================================
def test_fixture_has_expected_columns():
    df = load_fixture()
    assert validate_schema(df) == [], "Fixture phải khớp đúng schema kỳ vọng"
    assert list(df.columns) == EXPECTED_COLUMNS


# ===========================================================================
# holiday: "None" == KHÔNG phải ngày lễ, KHÔNG phải missing
# ===========================================================================
def test_holiday_none_is_not_converted_to_nan():
    """`keep_default_na=False` phải giữ chuỗi "None" nguyên vẹn."""
    df = load_fixture()
    assert df["holiday"].isna().sum() == 0, '"None" không được đọc thành NaN'
    assert (df["holiday"] == "None").sum() > 0


def test_holiday_none_means_non_holiday():
    """Chuỗi "None" KHÔNG BAO GIỜ được tự coi là ngày lễ.

    Lưu ý ngữ nghĩa: một dòng có holiday == "None" vẫn CÓ THỂ nằm trong ngày lễ,
    vì cột gốc chỉ ghi tên lễ ở giờ 00:00. Vì vậy test này kiểm tra trên fixture
    ở các giờ KHÔNG phải 00:00 của ngày lễ — chúng phải là ngày thường.
    """
    from src.features import HOLIDAY_NONE, add_holiday_flag

    df = add_holiday_flag(load_fixture())
    none_rows = df[df["holiday"] == HOLIDAY_NONE]
    assert len(none_rows) > 0

    # nhóm "None" nằm ngoài các ngày lễ => is_holiday phải bằng 0
    holiday_dates = set(
        df.loc[df["holiday"] != HOLIDAY_NONE, "date_time"].dt.normalize()
    )
    outside = none_rows[~none_rows["date_time"].dt.normalize().isin(holiday_dates)]
    assert len(outside) > 0
    assert outside["is_holiday"].sum() == 0, '"None" bị coi nhầm là ngày lễ!'


def test_named_holiday_is_holiday():
    """Tên ngày lễ thật => is_holiday = 1 (lan toả cho cả ngày lịch)."""
    from src.features import HOLIDAY_NONE, add_holiday_flag

    df = add_holiday_flag(load_fixture())
    named = df[df["holiday"] != HOLIDAY_NONE]
    assert len(named) > 0
    assert named["is_holiday"].eq(1).all()


def test_notna_would_be_wrong_for_holiday():
    """Bảo vệ khỏi cái bẫy đã nêu: `.notna().any()` SAI khi "None" là chuỗi."""
    from src.features import HOLIDAY_NONE

    df = load_fixture()
    # Bẫy: nếu dùng notna() thì MỌI ngày đều thành ngày lễ
    wrong = df.groupby(df["date_time"].dt.normalize())["holiday"].apply(lambda s: s.notna().any())
    assert wrong.all(), "Fixture phải phản ánh đúng cái bẫy notna()"

    # Cách đúng: so sánh với hằng số "None"
    right = df.groupby(df["date_time"].dt.normalize())["holiday"].apply(
        lambda s: (s != HOLIDAY_NONE).any()
    )
    assert right.sum() < len(right), "Cách đúng phải chỉ ra phần lớn ngày KHÔNG phải lễ"


def test_holiday_audit_counts():
    audit = audit_holiday(load_fixture())
    assert audit["n_holiday_none_string"] > 0
    assert audit["n_holiday_named"] > 0
    # Trong file gốc chỉ giờ 00:00 mới ghi tên lễ => số dòng có tên < số giờ của ngày lễ
    assert audit["n_holiday_dates"] == 1
    assert audit["n_holiday_named"] == 1
    assert audit["n_holiday_as_nan_if_default_read"] == 0
    assert "New Years Day" in audit["holiday_name_counts"]


def test_holiday_distribution_is_reasonable(clean: pd.DataFrame):
    """Phân bố holiday sau collapse phải hợp lý: phần lớn là 'None'."""
    counts = clean["holiday"].value_counts()
    n_none = int(counts.get("None", 0))
    n_named = int(len(clean) - n_none)
    assert n_none > 0
    assert 0 < n_named < len(clean), "Không thể toàn bộ là ngày lễ hoặc toàn bộ là ngày thường"
    assert n_none / len(clean) > 0.5, "Phần lớn số giờ phải là ngày thường"
    assert clean["holiday"].notna().all(), "Không được có NaN trong holiday sau xử lý"


def test_holiday_flag_covers_whole_calendar_day(clean: pd.DataFrame):
    from src.features import add_holiday_flag

    df = add_holiday_flag(clean)
    holiday_days = df.loc[df["is_holiday"] == 1, "date_time"].dt.normalize().unique()
    assert len(holiday_days) > 0
    for day in holiday_days:
        rows = df[df["date_time"].dt.normalize() == day]
        assert rows["is_holiday"].eq(1).all(), f"Ngày {day} bị lan toả chưa đủ"


# ===========================================================================
# Duplicate timestamp: KHÔNG drop_duplicates keep='first'
# ===========================================================================
def test_fixture_contains_duplicate_timestamps():
    df = load_fixture()
    audit = audit_duplicates(df)
    assert audit["n_duplicate_groups"] == 2
    assert audit["n_rows_in_duplicate_groups"] == 7
    assert audit["duplicate_group_size_max"] == 4
    assert audit["n_unique_timestamps"] == 64
    assert audit["n_raw_rows"] == 69


def test_collapse_gives_one_row_per_timestamp(clean: pd.DataFrame):
    assert clean["date_time"].is_unique, "Sau collapse mỗi timestamp phải có đúng 1 dòng"
    assert len(clean) == 64
    assert clean["date_time"].is_monotonic_increasing


def test_duplicates_are_reported_with_conflicting_columns():
    audit = audit_duplicates(load_fixture())
    # traffic/holiday/snow bất biến; weather khác nhau
    assert audit["invariant_in_duplicate_groups"]["traffic_volume"] == 2
    assert audit["invariant_in_duplicate_groups"]["holiday"] == 2
    assert audit["invariant_in_duplicate_groups"]["snow_1h"] == 2
    assert audit["varying_in_duplicate_groups"]["weather_main"] == 2
    assert audit["varying_in_duplicate_groups"]["weather_description"] == 2
    assert audit["varying_in_duplicate_groups"]["temp"] == 2
    assert audit["varying_in_duplicate_groups"]["rain_1h"] == 1
    assert audit["varying_in_duplicate_groups"]["clouds_all"] == 2


def test_traffic_target_invariant_across_duplicates():
    """Gộp KHÔNG được làm đổi giá trị lưu lượng."""
    raw = load_fixture()
    nunique = raw.groupby("date_time")["traffic_volume"].nunique()
    assert nunique[nunique > 1].empty, "traffic_volume phải bất biến trong mọi nhóm trùng"

    collapsed = collapse_duplicates(raw)
    expected = (
        raw[["date_time", "traffic_volume"]]
        .drop_duplicates()
        .sort_values("date_time")
        .reset_index(drop=True)
    )
    assert collapsed["traffic_volume"].tolist() == expected["traffic_volume"].tolist()


def test_holiday_invariant_across_duplicates(clean: pd.DataFrame):
    raw = load_fixture()
    dup_ts = audit_duplicates(raw)["n_duplicate_groups"]
    assert dup_ts > 0
    nunique = raw.groupby("date_time")["holiday"].nunique()
    assert nunique[nunique > 1].empty, "holiday phải bất biến trong nhóm trùng"


def test_snow_invariant_across_duplicates():
    raw = load_fixture()
    nunique = raw.groupby("date_time")["snow_1h"].nunique()
    assert nunique[nunique > 1].empty, "snow_1h phải bất biến trong nhóm trùng"


def test_numeric_duplicate_aggregation_uses_median():
    """temp/rain/clouds gộp bằng median, không phải dòng đầu."""
    raw = load_fixture()
    collapsed = collapse_duplicates(raw).set_index("date_time")

    # 2013-01-01 06:00: temp = [309, 280, 290] -> median 290
    assert collapsed.loc["2013-01-01 06:00:00", "temp"] == 290.0
    # clouds = [50, 20, 90] -> median 50
    assert collapsed.loc["2013-01-01 06:00:00", "clouds_all"] == 50.0

    # 2013-01-01 07:00: temp = [310, 290, 310, 296] -> median (296+310)/2 = 303.0
    assert collapsed.loc["2013-01-01 07:00:00", "temp"] == 303.0
    # clouds = [60, 20, 95, 50] -> median (50+60)/2 = 55
    assert collapsed.loc["2013-01-01 07:00:00", "clouds_all"] == 55.0
    # rain = [0, 0, 55.63, 0] -> median 0 (median không bị phép đo lớn kéo lệch)
    assert collapsed.loc["2013-01-01 07:00:00", "rain_1h"] == 0.0

    # median KHÔNG phụ thuộc thứ tự dòng
    shuffled = raw.sample(frac=1.0, random_state=7).reset_index(drop=True)
    collapsed_shuffled = collapse_duplicates(shuffled).set_index("date_time")
    assert collapsed_shuffled.loc["2013-01-01 07:00:00", "temp"] == 303.0
    assert collapsed_shuffled.loc["2013-01-01 07:00:00", "clouds_all"] == 55.0


def test_weather_multi_hot_preserves_all_categories():
    """weather_main KHÔNG được chọn dòng đầu — phải giữ mọi hiện tượng."""
    raw = load_fixture()
    collapsed = collapse_duplicates(raw).set_index("date_time")

    # 06:00: dòng đầu = Clear, nhưng 2/3 bản ghi là Thunderstorm
    first_row_main = (
        raw[raw["date_time"] == "2013-01-01 06:00:00"].iloc[0]["weather_main"]
    )
    assert first_row_main == "Clear"
    row = collapsed.loc["2013-01-01 06:00:00"]
    assert row["weather_main_mode"] == "Thunderstorm", "mode phải theo đa số, không phải dòng đầu"
    assert row["weather_main_set"] == "Clear; Thunderstorm"
    assert row["weather_main_n"] == 2
    assert row["wm_clear"] == 1 and row["wm_thunderstorm"] == 1
    assert row["weather_severity"] == 4

    # 07:00: dòng đầu = Clouds, nhưng 2/4 bản ghi là Mist
    assert raw[raw["date_time"] == "2013-01-01 07:00:00"].iloc[0]["weather_main"] == "Clouds"
    row15 = collapsed.loc["2013-01-01 07:00:00"]
    assert row15["weather_main_mode"] == "Mist"
    assert row15["weather_main_set"] == "Clouds; Mist; Squall"
    assert row15["weather_main_n"] == 3
    # severity = mức nghiêm trọng nhất trong giờ (Squall = 4)
    assert row15["weather_severity"] == 4


def test_weather_aggregation_is_order_independent():
    raw = load_fixture()
    a = collapse_duplicates(raw).set_index("date_time")
    b = collapse_duplicates(
        raw.sample(frac=1.0, random_state=11).reset_index(drop=True)
    ).set_index("date_time")
    for ts in a.index:
        assert a.loc[ts, "weather_main_set"] == b.loc[ts, "weather_main_set"]
        assert a.loc[ts, "weather_main_mode"] == b.loc[ts, "weather_main_mode"]
        assert a.loc[ts, "weather_severity"] == b.loc[ts, "weather_severity"]
        for col in weather_main_columns():
            assert a.loc[ts, col] == b.loc[ts, col]


def test_deterministic_mode_tiebreak_is_alphabetical():
    assert deterministic_mode(["Rain", "Clear"]) == "Clear"
    assert deterministic_mode(["Clear", "Rain"]) == "Clear"
    assert deterministic_mode(["Clouds", "Clouds", "Rain"]) == "Clouds"


def test_assert_timestamp_invariants_raises_when_target_varies():
    """Guard phải thật sự chặn được khi target không bất biến."""
    raw = load_fixture()
    bad = raw.copy()
    idx = bad.index[bad["date_time"] == "2013-01-01 06:00:00"][0]
    bad.loc[idx, "traffic_volume"] = 999999
    audit = audit_duplicates(bad)
    with pytest.raises(AssertionError, match="traffic_volume"):
        assert_timestamp_invariants(bad, audit)


# ===========================================================================
# Outlier: giá trị đo vô lý -> NaN, KHÔNG interpolate
# ===========================================================================
def test_temp_zero_becomes_nan():
    raw = load_fixture()
    assert (raw["temp"] == 0).sum() == 1
    collapsed = collapse_duplicates(raw)
    flagged, stats = flag_invalid_measurements(collapsed)
    assert stats["n_temp_invalid"] == 1
    assert flagged["temp"].isna().sum() == 1
    # giá trị 0 KHÔNG còn trong cột
    assert not (flagged["temp"] == TEMP_K_INVALID_MAX).any()


def test_rain_extreme_becomes_nan():
    raw = load_fixture()
    assert (raw["rain_1h"] == 9831.3).sum() == 1
    collapsed = collapse_duplicates(raw)
    flagged, stats = flag_invalid_measurements(collapsed)
    assert stats["n_rain_invalid"] == 1
    assert flagged["rain_1h"].isna().sum() == 1
    # các giá trị hợp lệ khác phải được giữ nguyên, không bị sửa
    assert flagged["rain_1h"].max() == 12.5


def test_rain_rule_is_exact_match_not_threshold():
    """YÊU CẦU 3.1: KHÔNG được dùng ngưỡng hậu nghiệm cho rain_1h.

    Một giá trị lớn nhưng KHÔNG phải sentinel phải được giữ lại — vì bất kỳ ngưỡng nào
    suy ra từ "max còn lại của tập dữ liệu" đều là hậu nghiệm và sẽ xoá nhầm dữ liệu hợp lệ.
    """
    raw = load_fixture()
    # chèn một giá trị rất lớn nhưng hợp lệ về mặt quy tắc (không phải sentinel)
    idx = raw.index[raw["date_time"] == "2013-01-03 00:00:00"][0]
    raw.loc[idx, "rain_1h"] = 500.0

    flagged, stats = flag_invalid_measurements(collapse_duplicates(raw))
    # chỉ sentinel mới bị loại; 500.0 phải được giữ lại
    assert stats["n_rain_invalid"] == 1
    assert flagged["rain_1h"].isna().sum() == 1
    assert 500.0 in set(flagged["rain_1h"].dropna()), "Giá trị hợp lệ phải được giữ nguyên"


def test_rain_rule_declares_it_is_not_a_threshold():
    raw = load_fixture()
    _df, stats = flag_invalid_measurements(collapse_duplicates(raw))
    assert stats["rain_rule_is_threshold"] is False
    assert "KHÔNG dùng ngưỡng" in stats["rain_invalid_rule"]
    assert 9831.3 in RAIN_1H_SENTINEL_INVALID


def test_no_rain_threshold_constant_exists():
    """Chặn hồi quy: hằng số ngưỡng rain đã bị gỡ bỏ khỏi source."""
    import src.data as data_mod

    assert not hasattr(data_mod, "RAIN_1H_INVALID_MIN"), (
        "Không được tồn tại ngưỡng rain_1h — đó là rule hậu nghiệm"
    )
    source = (ROOT / "src" / "data.py").read_text(encoding="utf-8")
    assert "RAIN_1H_INVALID_MIN" not in source


def test_temp_rule_is_physics_based():
    """temp giữ ngưỡng vì có căn cứ vật lý (0 K = nhiệt độ tuyệt đối), không phải vì thống kê."""
    raw = load_fixture()
    _df, stats = flag_invalid_measurements(collapse_duplicates(raw))
    assert "vật lý" in stats["temp_invalid_rule"]


def test_no_interpolation_fills_nan():
    """NaN phải được giữ nguyên để SimpleImputer xử lý, KHÔNG bị interpolate."""
    raw = load_fixture()
    _df, _report, audit = build_clean_dataset(raw)
    assert audit["invalid_measurements"]["n_temp_nan_after"] == 1
    assert audit["invalid_measurements"]["n_rain_nan_after"] == 1
    collapsed = collapse_duplicates(raw)
    flagged, _ = flag_invalid_measurements(collapsed)
    assert flagged["temp"].isna().sum() == 1


def test_outlier_nan_reaches_clean_dataset(clean: pd.DataFrame):
    assert clean["temp"].isna().sum() == 1
    assert clean["rain_1h"].isna().sum() == 1


# ===========================================================================
# weather_description -> nhóm gia đình
# ===========================================================================
def test_weather_description_families():
    assert classify_weather_description("thunderstorm with heavy rain") == "thunder"
    assert classify_weather_description("SQUALLS") == "thunder"
    assert classify_weather_description("light snow") == "snow"
    assert classify_weather_description("very heavy rain") == "rain"
    assert classify_weather_description("heavy intensity drizzle") == "rain"
    assert classify_weather_description("overcast clouds") == "overcast"
    assert classify_weather_description("broken clouds") == "partly_cloudy"
    assert classify_weather_description("Sky is Clear") == "clear"  # khác hoa/thường
    assert classify_weather_description("sky is clear") == "clear"
    assert classify_weather_description("") == "other"


def test_weather_family_keeps_most_severe(clean: pd.DataFrame):
    row = clean.set_index("date_time").loc["2013-01-01 07:00:00"]
    # trong giờ đó: "overcast clouds", "mist" x2, "SQUALLS" -> thunder là nghiêm trọng nhất
    assert row["weather_family"] == "thunder"
    assert row["weather_desc_n"] == 3


def test_all_weather_main_categories_have_binary_columns(clean: pd.DataFrame):
    for col in weather_main_columns():
        assert col in clean.columns
        assert set(clean[col].unique()) <= {0, 1}


def test_aggregate_weather_group_multi_hot():
    out = aggregate_weather_group(["Clouds", "Clouds", "Rain"])
    assert out["wm_clouds"] == 1
    assert out["wm_rain"] == 1
    assert out["wm_snow"] == 0
    assert out["weather_main_mode"] == "Clouds"
    assert out["weather_main_set"] == "Clouds; Rain"
    assert out["weather_severity"] == 3  # Rain


# ===========================================================================
# Khoảng trống + báo cáo
# ===========================================================================
def test_detects_missing_hour():
    _, report, _audit = build_clean_dataset(load_fixture())
    assert "Khoảng trống theo giờ" in report
    assert "Số giờ bị THIẾU hẳn dòng" in report


def test_report_contains_key_numbers():
    raw = load_fixture()
    _df, report, audit = build_clean_dataset(raw)
    assert "Trùng lặp" in report
    assert "median" in report
    assert "multi-hot" in report
    assert "keep_default_na=False" in report
    assert f"{audit['n_raw_rows']}" in report
    assert "KHÔNG nội suy" in report or "KHÔNG interpolate" in report
    # báo cáo phải nêu rõ KHÔNG dùng drop_duplicates keep first
    assert "drop_duplicates" in report


def test_clean_and_report_backward_compatible():
    df_clean, report = clean_and_report(load_fixture())
    assert len(df_clean) == 64
    assert isinstance(report, str) and len(report) > 0


def test_hour_gap_reported_in_coverage():
    _df, _report, audit = build_clean_dataset(load_fixture())
    cov = audit["coverage"]
    assert cov["n_missing_hours"] == 2
    assert cov["n_present_hours"] == 64


def test_audit_json_is_serialisable():
    import json

    _df, _report, audit = build_clean_dataset(load_fixture())
    text = json.dumps(audit, ensure_ascii=False, allow_nan=False)
    assert json.loads(text)["n_rows_after_collapse"] == 64


# ===========================================================================
# Lịch tất định (yêu cầu 3.1 #4)
# ===========================================================================
def test_holiday_calendar_never_misses_a_dataset_holiday(clean: pd.DataFrame):
    """Lịch tất định phải bắt được MỌI ngày lễ dataset biết (FN = 0)."""
    audit = audit_holiday_calendar(clean)
    assert audit["n_data_holidays_missed_by_calendar"] == 0, (
        f"Lịch bỏ sót: {audit['missed_by_calendar']}"
    )
    assert audit["n_agreements"] == audit["n_holiday_dates_in_data"]
    assert audit["n_mismatched_name"] == 0


def test_holiday_flag_works_without_dataset_rows():
    """is_holiday phải tính được từ NGÀY, không cần bảng dữ liệu."""
    from src.holidays import holiday_name, is_holiday
    from datetime import date

    # 2017-07-04 là Independence Day
    assert is_holiday(date(2017, 7, 4)) == 1
    assert holiday_name(date(2017, 7, 4)) == "Independence Day"
    # ngày thường
    assert is_holiday(date(2017, 7, 5)) == 0
    assert holiday_name(date(2017, 7, 5)) == "None"
    # ngoài phạm vi dataset vẫn tính được
    assert is_holiday(date(2020, 12, 25)) == 1
    assert is_holiday(date(1999, 1, 1)) == 1


def test_no_nan_in_traffic_target(clean: pd.DataFrame):
    assert clean["traffic_volume"].notna().all()
    assert np.isfinite(clean["traffic_volume"]).all()

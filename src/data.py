"""
src/data.py

Load -> validate -> collapse trùng `date_time` -> đánh dấu giá trị đo vô lý ->
sinh báo cáo chất lượng dữ liệu.

4 quy tắc bắt buộc (đã audit trên dữ liệu thật):
  1. `holiday` đọc bằng `keep_default_na=False`; chuỗi "None" = KHÔNG PHẢI ngày lễ,
     KHÔNG phải missing.
  2. KHÔNG `drop_duplicates(..., keep="first")`. Mỗi timestamp = 1 quan sát, gộp
     deterministic (median cho đo nhiều lần, multi-hot cho nhãn thời tiết).
  3. `temp <= 0 K` và `rain_1h` vượt ngưỡng lỗi cảm biến -> NaN (KHÔNG interpolate
     ở bước này; việc điền NaN do SimpleImputer trong pipeline, fit trên TRAIN).
  4. Không có bước nào học thống kê trên toàn bộ dữ liệu.

Usage:
    python src/data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Cho phép chạy cả `python src/data.py` lẫn `python -m src.data`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.weather import (  # noqa: E402
    WEATHER_MAIN_CATEGORIES,
    aggregate_weather_description_group,
    aggregate_weather_group,
    weather_main_columns,
)
from src.holidays import HOLIDAY_NONE, holiday_name  # noqa: E402

ROOT = _ROOT
RAW_PATH = ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
REPORT_PATH = ROOT / "reports" / "figures" / "data_quality_report.md"
CLEAN_PATH = ROOT / "data" / "processed" / "traffic_clean.csv"
AUDIT_PATH = ROOT / "data" / "processed" / "data_audit.json"

EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

# ---------------------------------------------------------------------------
# Ngưỡng / quy tắc bất biến vật lý
# ---------------------------------------------------------------------------
# temp: dataset có đúng 10 dòng temp == 0.0 K (≈ -273.15 °C). 0 K là nhiệt độ TUYỆT ĐỐI
# bằng 0 — về mặt vật lý không tồn tại ngoài trời và không thể đo được. Đây là quy tắc
# DỰA TRÊN VẬT LÝ (độc lập với bất kỳ bộ dữ liệu nào), KHÔNG phải ngưỡng rút ra từ
# phân bố quan sát được — nên giữ.
TEMP_K_INVALID_MAX = 0.0  # temp <= ngưỡng này => NaN

# rain_1h: KHÔNG dùng ngưỡng kiểu "rain_1h > X".
#
# Lý do: một ngưỡng lấy từ "giá trị lớn nhất còn lại sau khi lọc" (ví dụ 55.63 mm rồi
# đặt ngưỡng 100 mm) là ngưỡng HẬU NGHIỆM — nó được chọn bằng cách nhìn toàn bộ tập dữ
# liệu, nên không có giá trị như một quy tắc khái quát, và sẽ xoá nhầm các phép đo
# hợp lệ ở bất kỳ tập dữ liệu hay miền khác.
#
# Thay vào đó ta dùng khớp CHÍNH XÁC các giá trị đã được audit là lỗi nhập liệu/cảm biến
# trong đúng bản phát hành UCI này (9831.3 mm ≈ 386 inch — dấu hiệu nhập nhầm đơn vị).
# Khớp chính xác không bao giờ xoá một phép đo hợp lệ, bất kể phân bố ra sao.
RAIN_1H_SENTINEL_INVALID: frozenset[float] = frozenset({9831.3})

# Biến phải BẤT BIẾN trong mọi nhóm trùng `date_time` (audit: 5445/5445 nhóm).
INVARIANT_COLUMNS = ["traffic_volume", "holiday", "snow_1h"]

# Biến có thể có nhiều phép đo khác nhau trong cùng một giờ -> gộp median.
MEDIAN_AGGREGATED_COLUMNS = ["temp", "rain_1h", "clouds_all"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Đọc CSV thô.

    `keep_default_na=False` là BẮT BUỘC: mặc định pandas sẽ biến chuỗi "None" trong
    cột `holiday` thành NaN, tức biến "không phải ngày lễ" thành "dữ liệu thiếu".
    `na_values=[""]` giữ lại cơ chế nhận diện chuỗi rỗng thật sự.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}.\n"
            "Chạy `python src/download_data.py` (trên máy có Internet thường) "
            "hoặc tải thủ công theo hướng dẫn trong data/README.md, "
            "rồi đặt file vào đúng đường dẫn trên."
        )
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return coerce_types(df)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["holiday", "weather_main", "weather_description"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("")
    return df


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Trả về danh sách cảnh báo. Không raise — để vẫn tiếp tục ra được report."""
    warnings: list[str] = []
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_cols:
        warnings.append(f"THIẾU cột kỳ vọng: {missing_cols}")
    if extra_cols:
        warnings.append(f"Có cột lạ không nằm trong schema gốc: {extra_cols}")
    return warnings


def audit_duplicates(df: pd.DataFrame) -> dict:
    """Thống kê chi tiết các nhóm trùng `date_time` mà KHÔNG sửa gì."""
    grouped = df.groupby("date_time", sort=True)
    sizes = grouped.size()

    dup_sizes = sizes[sizes > 1]
    dup_ts = dup_sizes.index

    n_unique_ts = int(len(sizes))
    n_dup_groups = int(len(dup_sizes))
    n_rows_in_dup_groups = int(dup_sizes.sum())

    invariant_report: dict[str, int] = {}
    varying_report: dict[str, int] = {}
    for col in EXPECTED_COLUMNS:
        if col == "date_time":
            continue
        nunique = grouped[col].nunique(dropna=False).reindex(dup_ts)
        n_invariant = int((nunique == 1).sum())
        invariant_report[col] = n_invariant
        varying_report[col] = n_dup_groups - n_invariant

    # Nhóm mà thời tiết BỊ KHÁC nhau -> nguyên nhân trùng lặp
    weather_varying = {
        "weather_main": varying_report["weather_main"],
        "weather_description": varying_report["weather_description"],
    }

    return {
        "n_raw_rows": int(len(df)),
        "n_unique_timestamps": n_unique_ts,
        "n_duplicate_groups": n_dup_groups,
        "n_rows_in_duplicate_groups": n_rows_in_dup_groups,
        "n_singleton_groups": int((sizes == 1).sum()),
        "duplicate_group_size_min": int(dup_sizes.min()) if n_dup_groups else 0,
        "duplicate_group_size_max": int(dup_sizes.max()) if n_dup_groups else 0,
        "invariant_in_duplicate_groups": invariant_report,
        "varying_in_duplicate_groups": varying_report,
        "n_duplicate_groups_with_conflicting_weather": int(weather_varying["weather_main"]),
    }


def assert_timestamp_invariants(df_raw: pd.DataFrame, audit: dict) -> None:
    """Chặn cứng: mọi cột bắt buộc bất biến PHẢI thực sự bất biến trong nhóm trùng."""
    total = audit["n_duplicate_groups"]
    if total == 0:
        return
    for col in INVARIANT_COLUMNS:
        n_ok = audit["invariant_in_duplicate_groups"].get(col, 0)
        if n_ok != total:
            raise AssertionError(
                f"Cột `{col}` KHÔNG bất biến trong {total - n_ok}/{total} nhóm trùng date_time. "
                "Quy tắc collapse giả định cột này không đổi — dừng lại để kiểm tra lại dữ liệu."
            )


def _grouped_to_frame(series_grouped, func, index: pd.Index) -> pd.DataFrame:
    """Gom kết quả theo nhóm thành DataFrame, ổn định qua các phiên bản pandas.

    KHÔNG dùng `groupby.apply(func_trả_về_dict)`: pandas 3 trả về Series MultiIndex
    (group, key) và chỉ giữ lại MỘT giá trị cho mọi key -> âm thầm mất dữ liệu.
    Thay vào đó ta lấy list giá trị thô của từng nhóm rồi tính toán tường minh.
    """
    raw = series_grouped.agg(list)  # Series: date_time -> list[str]
    records = {key: func(list(values)) for key, values in raw.items()}
    frame = pd.DataFrame.from_dict(records, orient="index")
    return frame.reindex(index)


def collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """1 timestamp = 1 quan sát, gộp DETERMINISTIC.

    - `traffic_volume`, `holiday`, `snow_1h`: bất biến trong nhóm (đã assert ở trên)
      nên lấy giá trị đại diện.
    - `temp`, `rain_1h`, `clouds_all`: có thể nhiều phép đo -> MEDIAN.
      Median không nhạy với phép đo lỗi, giá trị luôn nằm trong tập phép đo
      (không bị "ảo" ra ngoài thực tế như mean), và không phụ thuộc thứ tự dòng.
    - `weather_main` / `weather_description`: KHÔNG lấy dòng đầu. Giữ toàn bộ thông
      tin qua multi-hot + nhãn đại diện có quy tắc mode→alphabet.
    """
    df = df.sort_values("date_time", kind="mergesort").reset_index(drop=True)
    grouped = df.groupby("date_time", sort=True)

    agg_map: dict[str, tuple[str, str]] = {}
    for col in INVARIANT_COLUMNS:
        agg_map[col] = (col, "first")
    for col in MEDIAN_AGGREGATED_COLUMNS:
        agg_map[col] = (col, "median")
    agg_map["n_weather_obs"] = ("weather_main", "size")

    out = grouped.agg(**agg_map)

    # --- thời tiết: multi-hot + nhãn đại diện deterministic ---
    weather_main_df = _grouped_to_frame(
        grouped["weather_main"], aggregate_weather_group, out.index
    )
    weather_desc_df = _grouped_to_frame(
        grouped["weather_description"], aggregate_weather_description_group, out.index
    )

    out = out.join(weather_main_df).join(weather_desc_df)

    ordered = [
        "date_time",
        "traffic_volume",
        "holiday",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main_mode",
        "weather_main_set",
        "weather_main_n",
        "weather_desc_mode",
        "weather_desc_set",
        "weather_desc_n",
        "weather_family",
        "weather_severity",
        "n_weather_obs",
        *weather_main_columns(),
    ]
    out.index.name = "date_time"
    out = out.reset_index()
    return out[ordered]


def flag_invalid_measurements(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Đánh dấu giá trị đo vô lý thành NaN bằng quy tắc DETERMINISTIC.

    KHÔNG nội suy ở bước này. NaN sẽ được xử lý bởi `SimpleImputer` nằm trong
    sklearn pipeline, và imputer CHỈ fit trên TRAIN.
    """
    df = df.copy()

    n_bad_temp = int((df["temp"] <= TEMP_K_INVALID_MAX).sum())
    df.loc[df["temp"] <= TEMP_K_INVALID_MAX, "temp"] = np.nan

    # Khớp CHÍNH XÁC giá trị sentinel, KHÔNG dùng ngưỡng
    rain_bad_mask = df["rain_1h"].isin(RAIN_1H_SENTINEL_INVALID)
    n_bad_rain = int(rain_bad_mask.sum())
    df.loc[rain_bad_mask, "rain_1h"] = np.nan

    stats = {
        "n_temp_invalid": n_bad_temp,
        "n_rain_invalid": n_bad_rain,
        "temp_invalid_rule": f"temp <= {TEMP_K_INVALID_MAX} K -> NaN (quy tắc vật lý: 0 K là nhiệt độ tuyệt đối)",
        "rain_invalid_rule": (
            f"rain_1h ∈ {sorted(RAIN_1H_SENTINEL_INVALID)} -> NaN "
            "(khớp chính xác giá trị lỗi đã audit trong bản phát hành này; KHÔNG dùng ngưỡng)"
        ),
        "rain_rule_is_threshold": False,
        "n_temp_nan_after": int(df["temp"].isna().sum()),
        "n_rain_nan_after": int(df["rain_1h"].isna().sum()),
    }
    return df, stats


def audit_holiday(df_raw: pd.DataFrame) -> dict:
    """Kiểm tra ngữ nghĩa cột `holiday` SAU KHI đọc bằng keep_default_na=False.

    Rất dễ sai: nếu dùng `s.notna().any()` thì chuỗi "None" cũng là chuỗi hợp lệ
    -> mọi ngày sẽ bị coi là ngày lễ. Ở đây so sánh với hằng số "None".
    """
    if "holiday" not in df_raw.columns:
        return {"error": "thiếu cột holiday"}
    values = df_raw["holiday"].astype("string").fillna("")
    n_total = int(len(values))
    n_none = int((values == "None").sum())
    n_named = n_total - n_none
    counts = values[values != "None"].value_counts().to_dict()
    # số ngày lịch duy nhất có tên lễ
    n_dates = 0
    if "date_time" in df_raw.columns and n_named:
        n_dates = int(df_raw.loc[values != "None", "date_time"].dt.normalize().nunique())
    return {
        "n_rows": n_total,
        "n_holiday_none_string": n_none,
        "n_holiday_named": n_named,
        "n_holiday_dates": n_dates,
        "n_distinct_holiday_names": len(counts),
        "holiday_name_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "n_holiday_as_nan_if_default_read": int(values.isna().sum()),
    }


def audit_holiday_calendar(df: pd.DataFrame) -> dict:
    """Đối chiếu lịch TẤT ĐỊNH trong `src/holidays.py` với cột `holiday` của dataset.

    Mục đích: chứng minh rằng `is_holiday` có thể tái tạo ở thời điểm dự báo mà KHÔNG cần
    đọc các dòng khác trong dataset. Nếu lịch không bỏ sót ngày lễ nào mà dataset biết
    (FN = 0), thì nó là nguồn đúng đắn và độc lập.
    """
    dates = df["date_time"].dt.date
    observed = set(dates.unique())
    data_named = {
        day: name
        for day, name in zip(dates, df["holiday"])
        if name != HOLIDAY_NONE
    }

    tp = fp = 0
    extra_calendar_only: list[str] = []
    for day in observed:
        cal_name = holiday_name(day)
        if cal_name == HOLIDAY_NONE:
            if day in data_named:
                fp += 1  # dataset nói là lễ nhưng lịch không biết -> KHÔNG được xảy ra
        elif day in data_named:
            if data_named[day] == cal_name:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
            extra_calendar_only.append(f"{day} ({cal_name})")

    fn = [f"{day} ({name})" for day, name in data_named.items() if holiday_name(day) == HOLIDAY_NONE]

    return {
        "n_observed_dates": len(observed),
        "n_holiday_dates_in_data": len(data_named),
        "n_agreements": tp,
        "n_data_holidays_missed_by_calendar": len(fn),
        "missed_by_calendar": fn,
        "n_calendar_only": len(extra_calendar_only),
        "calendar_only": sorted(extra_calendar_only)[:10],
        "n_mismatched_name": max(0, fp - len(extra_calendar_only)),
        "verdict": (
            "Lịch tất định khớp hoàn toàn và bổ sung cho dataset"
            if not fn
            else "LỊCH BỎ SÓT ngày lễ trong dataset — KHÔNG dùng được"
        ),
    }


def build_clean_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, str, dict]:
    """Pipeline đầy đủ: validate -> audit -> collapse -> flag giá trị vô lý."""
    n_raw = len(df_raw)
    schema_warnings = validate_schema(df_raw)

    df = df_raw.copy()
    if "date_time" in df.columns:
        n_bad_dates = int(df["date_time"].isna().sum())
        if n_bad_dates:
            df = df[df["date_time"].notna()].reset_index(drop=True)
    else:
        raise KeyError("Thiếu cột date_time — không thể xử lý.")

    dup_audit = audit_duplicates(df)
    assert_timestamp_invariants(df, dup_audit)

    df_collapsed = collapse_duplicates(df)
    n_after_collapse = int(len(df_collapsed))

    df_flagged, invalid_stats = flag_invalid_measurements(df_collapsed)
    holiday_audit = audit_holiday(df_flagged)
    calendar_audit = audit_holiday_calendar(df_flagged)

    gaps = audit_hour_gaps(df_flagged)

    audit = {
        "n_raw_rows": n_raw,
        "n_rows_after_drop_bad_date": int(len(df)),
        "n_rows_after_collapse": n_after_collapse,
        "n_rows_removed_by_collapse": n_raw - n_after_collapse,
        "schema_warnings": schema_warnings,
        "n_bad_dates_dropped": n_raw - len(df),
        "duplicates": dup_audit,
        "invalid_measurements": invalid_stats,
        "holiday": holiday_audit,
        "holiday_calendar": calendar_audit,
        "coverage": gaps,
    }

    report = render_report(audit, df_flagged)
    return df_flagged, report, audit


def audit_hour_gaps(df: pd.DataFrame) -> dict:
    full_range = pd.date_range(df["date_time"].min(), df["date_time"].max(), freq="h")
    present = pd.DatetimeIndex(df["date_time"])
    missing = full_range.difference(present)
    return {
        "start": str(df["date_time"].min()),
        "end": str(df["date_time"].max()),
        "n_expected_hours": int(len(full_range)),
        "n_present_hours": int(len(present)),
        "n_missing_hours": int(len(missing)),
        "missing_ratio": round(len(missing) / len(full_range), 6) if len(full_range) else 0.0,
    }


def render_report(audit: dict, df: pd.DataFrame) -> str:
    dup = audit["duplicates"]
    inv = audit["invalid_measurements"]
    hol = audit["holiday"]
    cov = audit["coverage"]
    train_like = df[df["date_time"].dt.year <= 2016]

    L: list[str] = []
    L.append("# Báo cáo chất lượng dữ liệu — Metro Interstate Traffic Volume")
    L.append("")
    L.append("> Sinh tự động bởi `python src/data.py`. Mọi con số dưới đây được đo trên file thật.")
    L.append("")

    L.append("## 1. Tổng quan")
    L.append(f"- Raw rows đọc được: **{audit['n_raw_rows']:,}**".replace(",", "."))
    L.append(f"- Rows sau khi loại `date_time` không hợp lệ: **{audit['n_rows_after_drop_bad_date']:,}**".replace(",", "."))
    L.append(f"- Rows sau khi collapse trùng `date_time`: **{audit['n_rows_after_collapse']:,}**".replace(",", "."))
    L.append(
        f"- Số dòng bị loại bởi collapse: **{audit['n_rows_removed_by_collapse']:,}**".replace(",", ".")
    )
    for w in audit["schema_warnings"]:
        L.append(f"- ⚠️ {w}")
    L.append("")

    L.append("## 2. Trùng lặp theo `date_time` — nguyên nhân và cách xử lý")
    L.append(f"- Số timestamp duy nhất: **{dup['n_unique_timestamps']:,}**".replace(",", "."))
    L.append(f"- Số nhóm trùng (>= 2 bản ghi): **{dup['n_duplicate_groups']:,}**".replace(",", "."))
    L.append(
        f"- Số dòng nằm trong các nhóm trùng: **{dup['n_rows_in_duplicate_groups']:,}** "
        f"({dup['n_rows_in_duplicate_groups'] / max(dup['n_raw_rows'], 1):.2%})".replace(",", ".")
    )
    L.append(
        f"- Kích thước nhóm trùng: từ {dup['duplicate_group_size_min']} đến {dup['duplicate_group_size_max']} bản ghi/timestamp"
    )
    L.append("")
    L.append("### Kiểm tra tính bất biến trong từng nhóm trùng")
    L.append("")
    L.append("| Cột | Nhóm bất biến | Nhóm khác nhau | Xử lý |")
    L.append("| --- | --- | --- | --- |")
    handlers = {
        "traffic_volume": "bất biến -> assert rồi lấy giá trị",
        "holiday": "bất biến -> assert rồi lấy giá trị",
        "snow_1h": "bất biến -> assert rồi lấy giá trị",
        "temp": "nhiều phép đo -> **median**",
        "rain_1h": "nhiều phép đo -> **median**",
        "clouds_all": "nhiều phép đo -> **median**",
        "weather_main": "nhiều hiện tượng -> **multi-hot** + nhãn đại diện mode→alphabet",
        "weather_description": "nhiều mô tả -> **multi-label** + nhãn gia đình thời tiết",
    }
    for col in EXPECTED_COLUMNS:
        if col == "date_time":
            continue
        n_ok = dup["invariant_in_duplicate_groups"].get(col, 0)
        n_bad = dup["varying_in_duplicate_groups"].get(col, 0)
        L.append(
            f"| `{col}` | {n_ok}/{dup['n_duplicate_groups']} | {n_bad} | {handlers.get(col, '—')} |"
        )
    L.append("")
    L.append(
        f"- Nguyên nhân trùng lặp: `weather_main` khác nhau trong "
        f"**{dup['varying_in_duplicate_groups']['weather_main']:,}**/{dup['n_duplicate_groups']} nhóm trùng, "
        f"`weather_description` khác nhau trong **{dup['varying_in_duplicate_groups']['weather_description']:,}** "
        f"nhóm — tức một giờ được ghi nhận nhiều hiện tượng/mô tả thời tiết.".replace(",", ".")
    )
    L.append(
        "- Vì `traffic_volume`, `holiday`, `snow_1h` bất biến trong **100%** nhóm trùng, "
        "việc gộp KHÔNG làm mất hay bóp méo thông tin lưu lượng hay ngày lễ."
    )
    L.append(
        "- ❌ KHÔNG dùng `drop_duplicates(subset=['date_time'], keep='first')`: cách đó vứt bỏ "
        "mọi hiện tượng thời tiết ngoài dòng đầu và phụ thuộc thứ tự dòng trong file (không deterministic về mặt nội dung)."
    )
    L.append(
        "- ✅ Dùng: median cho phép đo số, multi-hot/multi-label cho thời tiết, nhãn đại diện "
        "theo quy tắc (giá trị phổ biến nhất, hoà thì alphabet) — không phụ thuộc thứ tự dòng."
    )
    L.append("")

    L.append("## 3. Ngữ nghĩa cột `holiday`")
    L.append(
        f"- Đọc CSV bằng `keep_default_na=False` — bắt buộc, vì pandas mặc định biến chuỗi `\"None\"` thành NaN."
    )
    L.append(f"- Rows với `holiday == \"None\"` (KHÔNG phải ngày lễ): **{hol['n_holiday_none_string']:,}**".replace(",", "."))
    L.append(f"- Rows có tên ngày lễ thật: **{hol['n_holiday_named']:,}**".replace(",", "."))
    L.append(f"- Số ngày lễ (lịch) duy nhất: **{hol['n_holiday_dates']}**")
    L.append(f"- Số tên ngày lễ khác nhau: **{hol['n_distinct_holiday_names']}**")
    L.append("")
    L.append("| Tên ngày lễ | Số dòng (giờ 00:00 của ngày đó) |")
    L.append("| --- | --- |")
    for name, cnt in hol["holiday_name_counts"].items():
        L.append(f"| {name} | {cnt} |")
    L.append("")
    L.append(
        "- Trong file gốc, tên ngày lễ chỉ xuất hiện ở **giờ 00:00** của ngày lễ. Vì vậy cột này "
        "KHÔNG dùng được trực tiếp làm feature mà cũng không đủ tin cậy để tra cứu ở thời điểm dự báo."
    )
    L.append("")
    cal_a = audit.get("holiday_calendar", {})
    L.append("### Đối chiếu với lịch tất định (`src/holidays.py`)")
    L.append("")
    L.append(
        "`src/holidays.py` tính ngày lễ **từ ngày tháng bằng quy tắc lịch**, không đọc dữ liệu: "
        "MLK = thứ Hai thứ 3 tháng 1, Memorial = thứ Hai cuối tháng 5, Labor = thứ Hai thứ nhất "
        "tháng 9, Columbus = thứ Hai thứ 2 tháng 10, Thanksgiving = thứ Năm thứ 4 tháng 11, "
        "các ngày cố định có dời cuối tuần theo quy ước *observed day*, và "
        "`State Fair` lấy từ bảng ngày khai mạc do bang Minnesota công bố."
    )
    L.append("")
    L.append("| Kiểm tra | Kết quả |")
    L.append("| --- | --- |")
    L.append(f"| Số ngày lễ dataset ghi nhận | {cal_a.get('n_holiday_dates_in_data', 0)} |")
    L.append(f"| Lịch khớp đúng tên + đúng ngày | {cal_a.get('n_agreements', 0)} |")
    L.append(f"| Ngày lễ dataset có mà lịch BỎ SÓT | **{cal_a.get('n_data_holidays_missed_by_calendar', 0)}** |")
    L.append(f"| Ngày lịch có nhưng dataset không ghi (thiếu dòng 00:00) | {cal_a.get('n_calendar_only', 0)} |")
    L.append("")
    L.append(f"- Kết luận: **{cal_a.get('verdict', '—')}**")
    if cal_a.get("calendar_only"):
        L.append(
            "- Các ngày \"lịch có nhưng dataset không ghi\" là do dataset **thiếu hẳn dòng giờ 00:00** "
            "trong ngày đó, nên cột `holiday` không kịp ghi tên. Lịch tất định vẫn đánh dấu đúng — "
            "tức là lịch chính xác HƠN chính cột dữ liệu."
        )
    L.append(
        "- Vì vậy `is_holiday` được tính từ lịch này, **không** nhóm theo ngày trên bảng dữ liệu. "
        "Đây là thông tin lịch công cộng, biết trước tại thời điểm dự báo, nên dùng làm feature là hợp lệ; "
        "đồng thời Web/API chỉ cần nhận một ngày là tái tạo được, không cần quét bất kỳ dòng nào khác."
    )
    L.append(
        "- ❌ Cấm dùng `s.notna().any()` sau khi `\"None\"` đã là chuỗi — như vậy mọi ngày đều bị coi là ngày lễ."
    )
    L.append("")

    L.append("## 4. Giá trị đo vô lý (quy tắc tất định -> NaN, KHÔNG nội suy)")
    L.append("")
    L.append("### 4a. `temp` — quy tắc DỰA TRÊN VẬT LÝ")
    L.append(f"- Quy tắc: `{inv['temp_invalid_rule']}`")
    L.append(f"- Số dòng bị loại: **{inv['n_temp_invalid']}**")
    L.append(
        "- 0 K là nhiệt độ tuyệt đối — không tồn tại ngoài trời và không thể đo được. "
        "Đây là ngưỡng có căn cứ vật lý, độc lập với bất kỳ bộ dữ liệu cụ thể nào, "
        "nên áp dụng được một cách nguyên tắc cho mọi tập dữ liệu khác."
    )
    L.append("")
    L.append("### 4b. `rain_1h` — khớp CHÍNH XÁC giá trị lỗi, KHÔNG dùng ngưỡng")
    L.append(f"- Quy tắc: `{inv['rain_invalid_rule']}`")
    L.append(f"- Số dòng bị loại: **{inv['n_rain_invalid']}**")
    L.append(
        "- ❌ **Không dùng ngưỡng kiểu `rain_1h > 100`.** Một ngưỡng rút ra từ "
        "\"giá trị lớn nhất còn lại sau khi lọc\" (55,63 mm) là ngưỡng **hậu nghiệm**: "
        "nó được chọn bằng cách nhìn toàn bộ tập dữ liệu, nên không khái quát và sẽ xoá nhầm "
        "các phép đo hợp lệ trên bất kỳ tập dữ liệu hoặc miền nào khác. "
        "Tương tự, cũng không dùng kỷ lục mưa thế giới làm rule cho mô hình."
    )
    L.append(
        "- ✅ Dùng **khớp chính xác** các giá trị đã audit là lỗi nhập liệu trong đúng bản "
        "phát hành UCI này (9831,3 mm ≈ 386 inch — dấu hiệu nhập nhầm đơn vị). "
        "Khớp chính xác không bao giờ xoá một phép đo hợp lệ, bất kể phân bố dữ liệu thế nào."
    )
    L.append(
        "- Hệ quả trung thực: nếu sau này gặp một giá trị mưa lớn bất thường **khác** 9831,3, "
        "quy tắc này sẽ không bắt được. Đó là đánh đổi có ý thức giữa "
        "\"không xoá nhầm dữ liệu\" và \"bắt được mọi ngoại lệ\" — "
        "và trong bài toán này, xoá nhầm còn tệ hơn bỏ sót."
    )
    L.append(
        f"- Sau khi đánh dấu: `temp` còn {inv['n_temp_nan_after']} NaN, `rain_1h` còn {inv['n_rain_nan_after']} NaN"
    )
    L.append(
        "- ❌ KHÔNG chạy `interpolate(method='time', limit_direction='both')` trên toàn bộ dữ liệu: "
        "nội suy trước khi tách tập là rò rỉ thống kê từ validation/test vào train. "
        "NaN được xử lý bởi `SimpleImputer` nằm trong sklearn pipeline, **fit chỉ trên TRAIN**."
    )
    L.append("")

    L.append("## 5. Khoảng trống theo giờ & độ phủ thời gian (chỉ mang tính thông tin)")
    L.append(f"- Khoảng thời gian: {cov['start']} → {cov['end']}")
    L.append(f"- Số giờ lẽ ra phải có nếu liên tục tuyệt đối: {cov['n_expected_hours']:,}".replace(",", "."))
    L.append(f"- Số giờ có thật trong dữ liệu: {cov['n_present_hours']:,}".replace(",", "."))
    L.append(
        f"- Số giờ bị THIẾU hẳn dòng: {cov['n_missing_hours']:,} "
        f"({cov['missing_ratio']:.2%})".replace(",", ".")
    )
    L.append(
        "- 1 timestamp = 1 quan sát sau collapse. Nếu sau này tạo lag feature, phải reindex theo lưới giờ "
        "đầy đủ trước khi shift, nếu không `shift(1)` sẽ không còn là 'giờ trước'."
    )
    L.append("")

    L.append("## 6. Phân bố sau xử lý (toàn bộ, mô tả dữ liệu)")
    L.append("")
    L.append("| Cột | min | max | mean | n NaN |")
    L.append("| --- | --- | --- | --- | --- |")
    for col in ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_severity"]:
        s = pd.to_numeric(df[col], errors="coerce")
        L.append(
            f"| `{col}` | {s.min():.2f} | {s.max():.2f} | {s.mean():.2f} | {int(s.isna().sum())} |"
        )
    L.append("")
    L.append("### Phân bố `holiday` sau collapse")
    L.append("")
    L.append("| Giá trị | Số timestamp |")
    L.append("| --- | --- |")
    for name, cnt in df["holiday"].value_counts().items():
        L.append(f"| {name} | {int(cnt):,} |".replace(",", "."))
    L.append("")
    L.append("### `weather_main` sau collapse (multi-hot nên có thể >0 mã/cột)")
    L.append("")
    L.append("| weather_main | Số timestamp có hiện tượng này |")
    L.append("| --- | --- |")
    wm_sets = df["weather_main_set"].str.split("; ")
    for cat in WEATHER_MAIN_CATEGORIES:
        n = int(wm_sets.map(lambda xs: cat in xs).sum())
        L.append(f"| {cat} | {n:,} |".replace(",", "."))
    L.append("")
    L.append("### Nhãn đại diện `weather_main_mode` (dùng làm categorical feature)")
    L.append("")
    L.append("| weather_main_mode | Số timestamp |")
    L.append("| --- | --- |")
    for name, cnt in df["weather_main_mode"].value_counts().items():
        L.append(f"| {name} | {int(cnt):,} |".replace(",", "."))
    L.append("")
    L.append("### Số bản ghi thời tiết gộp lại mỗi giờ")
    L.append("")
    L.append("| n_weather_obs | Số timestamp |")
    L.append("| --- | --- |")
    for k, v in df["n_weather_obs"].value_counts().sort_index().items():
        L.append(f"| {int(k)} | {int(v):,} |".replace(",", "."))
    L.append("")
    L.append(f"- Kiểm tra chéo: rows trong TRAIN (≤2016) = {len(train_like):,}".replace(",", "."))
    L.append("")
    return "\n".join(L)


def clean_and_report(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Tương thích ngược: trả (df_clean, report)."""
    df_clean, report, _audit = build_clean_dataset(df_raw)
    return df_clean, report


def save_report(report: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # console Windows mặc định cp1252, không in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df_raw = load_raw()
    df_clean, report, audit = build_clean_dataset(df_raw)
    save_report(report)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(report)
    print(f"\nĐã lưu báo cáo : {REPORT_PATH}")
    print(f"Đã lưu audit JSON: {AUDIT_PATH}")
    print(f"Đã lưu dữ liệu sạch: {CLEAN_PATH}  ({len(df_clean):,} dòng)".replace(",", "."))


if __name__ == "__main__":
    main()

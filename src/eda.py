"""
src/eda.py

EDA CHỈ TRÊN TẬP TRAIN (2012-2016). Validation (2017) và Test (2018) bị chặn khỏi
mọi thống kê mô tả: nếu nhìn vào chúng trước khi đóng băng cấu hình thì việc chọn
feature / giả định phân bố sẽ bị nhiễm thông tin tương lai.

Usage:
    python src/eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import (  # noqa: E402
    TARGET_COLUMN,
    build_features,
    load_clean,
    time_split,
    weather_main_columns,
)
from src.weather import WEATHER_MAIN_CATEGORIES  # noqa: E402

FIG_DIR = _ROOT / "reports" / "figures"
REPORT_PATH = FIG_DIR / "eda_train_only.md"

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path.relative_to(_ROOT)).replace("\\", "/")


def plot_traffic_by_hour_dow(train: pd.DataFrame) -> str:
    pivot = train.pivot_table(
        index="hour", columns="day_of_week", values=TARGET_COLUMN, aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for dow in sorted(pivot.columns):
        ax.plot(pivot.index, pivot[dow], marker="o", markersize=3, label=DOW_LABELS[dow])
    ax.set_xlabel("Giờ trong ngày")
    ax.set_ylabel("Lưu lượng trung bình")
    ax.set_title("Lưu lượng trung bình theo giờ × thứ trong tuần (TRAIN 2012-2016)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.3)
    return _save(fig, "eda_traffic_by_hour_dow.png")


def plot_traffic_by_month_year(train: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for year, part in train.groupby("year"):
        monthly = part.groupby("month")[TARGET_COLUMN].mean().reindex(range(1, 13))
        ax.plot(range(1, 13), monthly.values, marker="o", label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Lưu lượng trung bình")
    ax.set_title("Lưu lượng trung bình theo tháng × năm (TRAIN 2012-2016)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "eda_traffic_by_month_year.png")


def plot_target_distribution(train: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train[TARGET_COLUMN], bins=60, color="#4C78A8")
    ax.set_xlabel("traffic_volume")
    ax.set_ylabel("Số giờ")
    ax.set_title("Phân bố target trên TRAIN")
    ax.grid(alpha=0.3)
    return _save(fig, "eda_target_distribution.png")


def plot_weather_distribution(train: pd.DataFrame) -> str:
    counts = [int(train[c].sum()) for c in weather_main_columns()]
    order = sorted(range(len(counts)), key=lambda i: counts[i])
    labels = [WEATHER_MAIN_CATEGORIES[i] for i in order]
    values = [counts[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color="#F58518")
    ax.set_xlabel("Số timestamp (multi-hot: một giờ có thể được tính ở nhiều cột)")
    ax.set_title("Tần suất weather_main trên TRAIN")
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:,}".replace(",", "."), va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "eda_weather_distribution.png")


def plot_weather_numeric(train: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, ["temp", "rain_1h", "snow_1h"]):
        s = train[col].dropna()
        ax.hist(s, bins=50, color="#54A24B")
        ax.set_title(f"{col} (n NaN sau xử lý: {int(train[col].isna().sum())})")
        ax.grid(alpha=0.3)
    fig.suptitle("Phân bố biến thời tiết số trên TRAIN")
    return _save(fig, "eda_weather_numeric.png")


def plot_holiday_effect(train: pd.DataFrame) -> str:
    hour = sorted(train["hour"].unique())
    normal = (
        train[train["is_holiday"] == 0].groupby("hour")[TARGET_COLUMN].mean().reindex(hour)
    )
    holiday = (
        train[train["is_holiday"] == 1].groupby("hour")[TARGET_COLUMN].mean().reindex(hour)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hour, normal.values, label="ngày thường", marker="o", markersize=3)
    ax.plot(hour, holiday.values, label="ngày lễ", marker="s", markersize=3)
    ax.set_xlabel("Giờ")
    ax.set_ylabel("Lưu lượng trung bình")
    ax.set_title(
        f"Ngày lễ vs ngày thường trong TRAIN (n_holiday_hours = {int(train['is_holiday'].sum())})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "eda_holiday_effect.png")


def render_eda_report(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[str, dict]:
    figs = {
        "hour_dow": plot_traffic_by_hour_dow(train),
        "month_year": plot_traffic_by_month_year(train),
        "target": plot_target_distribution(train),
        "weather_main": plot_weather_distribution(train),
        "weather_numeric": plot_weather_numeric(train),
        "holiday": plot_holiday_effect(train),
    }

    L: list[str] = []
    L.append("# EDA — CHỈ TRÊN TẬP TRAIN (2012-2016)")
    L.append("")
    L.append(
        "> Sinh tự động bởi `python src/eda.py`. Mọi thống kê dưới đây chỉ dùng dữ liệu TRAIN. "
        "Validation (2017) và Test (2018) không được nhìn vào trước khi cấu hình được đóng băng."
    )
    L.append("")

    L.append("## 1. Quy mô các tập (thống kê tối thiểu, chỉ để ghi nhận ranh giới thời gian)")
    L.append("")
    L.append("| Tập | Khoảng thời gian | Số dòng | Năm |")
    L.append("| --- | --- | --- | --- |")
    for name, part in [("TRAIN", train), ("VALIDATION", val), ("TEST", test)]:
        L.append(
            f"| {name} | {part['date_time'].min()} → {part['date_time'].max()} | "
            f"{_fmt(len(part))} | {', '.join(str(y) for y in sorted(part['year'].unique()))} |"
        )
    L.append("")
    L.append(
        "> Dòng trên chỉ ghi nhận mốc thời gian/số dòng để chứng minh split không chồng; "
        "không có phân bố nào của validation/test được tính ở đây."
    )
    L.append("")

    L.append("## 2. Target `traffic_volume` trên TRAIN")
    L.append("")
    s = train[TARGET_COLUMN]
    L.append(f"- Số quan sát: **{_fmt(len(train))}**")
    L.append(f"- min = {s.min():,.0f} | max = {s.max():,.0f} | mean = {s.mean():,.2f} | std = {s.std():,.2f}".replace(",", "."))
    L.append(f"- median = {s.median():,.0f} | p05 = {s.quantile(0.05):,.0f} | p95 = {s.quantile(0.95):,.0f}".replace(",", "."))
    L.append(f"- Số giờ có lưu lượng = 0: {int((s == 0).sum())}")
    L.append("")
    L.append(f"![Phân bố target]({figs['target']})")
    L.append("")

    L.append("### Theo giờ × thứ (trung bình)")
    L.append("")
    L.append(f"![Lưu lượng theo giờ x thứ]({figs['hour_dow']})")
    L.append("")
    L.append("| Giờ | " + " | ".join(DOW_LABELS) + " |")
    L.append("| --- |" + " --- |" * 7)
    pivot = train.pivot_table(index="hour", columns="day_of_week", values=TARGET_COLUMN, aggfunc="mean")
    for hour, row in pivot.iterrows():
        cells = [_fmt(row.get(d, float("nan"))) if pd.notna(row.get(d)) else "-" for d in range(7)]
        L.append(f"| {hour:02d} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(
        "- Hình dạng lưu lượng rất rõ: hai đỉnh buổi sáng (~08:00) và buổi tối (~17:00), "
        "đường cong ngày làm việc khác hẳn cuối tuần. Đây là lý do `hour_dow` là feature chính "
        "và cũng chính là cấu trúc mà baseline khai thác."
    )
    L.append("")

    L.append("### Theo tháng × năm")
    L.append("")
    L.append(f"![Lưu lượng theo tháng x năm]({figs['month_year']})")
    L.append("")
    L.append("| Năm | " + " | ".join(MONTH_LABELS) + " |")
    L.append("| --- |" + " --- |" * 12)
    for year, part in train.groupby("year"):
        monthly = part.groupby("month")[TARGET_COLUMN].mean().reindex(range(1, 13))
        cells = [_fmt(v) if pd.notna(v) else "-" for v in monthly.values]
        L.append(f"| {year} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. Ngày lễ trên TRAIN")
    L.append("")
    n_holiday_hours = int(train["is_holiday"].sum())
    n_holiday_days = int(train.loc[train["is_holiday"] == 1, "date_time"].dt.normalize().nunique())
    L.append(f"- Số giờ thuộc ngày lễ: **{_fmt(n_holiday_hours)}** ({n_holiday_hours / len(train):.2%} của TRAIN)")
    L.append(f"- Số ngày lễ: **{n_holiday_days}**")
    L.append(f"- Lưu lượng trung bình ngày thường: {_fmt(train.loc[train['is_holiday'] == 0, TARGET_COLUMN].mean())}")
    if n_holiday_hours:
        L.append(f"- Lưu lượng trung bình ngày lễ: {_fmt(train.loc[train['is_holiday'] == 1, TARGET_COLUMN].mean())}")
    L.append("")
    L.append(f"![Tác động ngày lễ]({figs['holiday']})")
    L.append("")

    L.append("## 4. Thời tiết trên TRAIN")
    L.append("")
    L.append(f"![Phân bố weather_main]({figs['weather_main']})")
    L.append("")
    L.append("| weather_main | Số timestamp (multi-hot) | Tỉ lệ |")
    L.append("| --- | --- | --- |")
    for cat, col in zip(WEATHER_MAIN_CATEGORIES, weather_main_columns()):
        n = int(train[col].sum())
        L.append(f"| {cat} | {_fmt(n)} | {n / len(train):.2%} |")
    L.append("")
    L.append(
        "- Đây là phân bố **multi-hot** (một giờ có thể thuộc nhiều hiện tượng), "
        "nên tổng các tỉ lệ có thể vượt 100%. Không dùng được `drop_duplicates(keep='first')` ở đây — "
        "nó sẽ chỉ giữ đúng 1 hiện tượng đầu tiên và làm mất phần còn lại."
    )
    L.append("")
    L.append(f"![Phân bố biến thời tiết số]({figs['weather_numeric']})")
    L.append("")
    L.append("| Biến | n NaN trong TRAIN | Ghi chú |")
    L.append("| --- | --- | --- |")
    for col in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
        n_nan = int(train[col].isna().sum())
        note = "sẽ do SimpleImputer (median, fit TRAIN) điền" if n_nan else "không thiếu"
        L.append(f"| `{col}` | {n_nan} | {note} |")
    L.append("")

    L.append("## 5. Mật độ quan sát theo năm (trên TRAIN) — cảnh báo về khoảng trống")
    L.append("")
    L.append("| Năm | Số giờ có dữ liệu | Số giờ lẽ ra có (nếu đủ 8760/8784) | Tỉ lệ phủ |")
    L.append("| --- | --- | --- | --- |")
    for year, part in train.groupby("year"):
        expected = int(pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00:00", freq="h").size)
        L.append(f"| {year} | {_fmt(len(part))} | {_fmt(expected)} | {len(part) / expected:.1%} |")
    L.append("")
    L.append(
        "- Năm 2015 đặc biệt thưa (chỉ từ 2015-06-11). Vì vậy kết luận " +
        "'mô hình generalize tốt qua các năm' phải dựa vào VALIDATION 2017 và TEST 2018, "
        "không dựa vào việc khớp tốt trên TRAIN."
    )
    L.append("")

    stats = {
        "n_train_rows": int(len(train)),
        "n_val_rows": int(len(val)),
        "n_test_rows": int(len(test)),
        "train_start": str(train["date_time"].min()),
        "train_end": str(train["date_time"].max()),
        "val_start": str(val["date_time"].min()),
        "val_end": str(val["date_time"].max()),
        "test_start": str(test["date_time"].min()),
        "test_end": str(test["date_time"].max()),
        "target_mean_train": float(train[TARGET_COLUMN].mean()),
        "n_holiday_hours_train": n_holiday_hours,
        "n_holiday_days_train": n_holiday_days,
        "target_holiday_mean_train": float(train.loc[train["is_holiday"] == 1, TARGET_COLUMN].mean())
        if n_holiday_hours
        else None,
        "target_normal_mean_train": float(train.loc[train["is_holiday"] == 0, TARGET_COLUMN].mean()),
        "weather_main_counts_train": {
            cat: int(train[col].sum())
            for cat, col in zip(WEATHER_MAIN_CATEGORIES, weather_main_columns())
        },
        "nan_counts_train": {
            col: int(train[col].isna().sum())
            for col in ["temp", "rain_1h", "snow_1h", "clouds_all"]
        },
        "rows_by_year_train": {str(y): int(len(p)) for y, p in train.groupby("year")},
        "figures": figs,
    }
    return "\n".join(L), stats


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    df = build_features(load_clean())
    train, val, test = time_split(df)
    report, _stats = render_eda_report(train, val, test)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nĐã lưu: {REPORT_PATH}")


if __name__ == "__main__":
    main()

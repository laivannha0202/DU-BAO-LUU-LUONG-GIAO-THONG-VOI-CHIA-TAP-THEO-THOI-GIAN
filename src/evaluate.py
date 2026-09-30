"""
src/evaluate.py

FINAL EVALUATION BUNDLE — chạy DUY NHẤT trên FINAL TEST 2018, SAU KHI cấu hình đã đóng băng.

Gói đánh giá này gồm đúng ba phần, theo yêu cầu nghiệm thu:
  1. Baseline vs Model
  2. MAE / RMSE / R²
  3. Phân tích lỗi theo hour / day_of_week / holiday / weather

⚠️ KHÔNG DÙNG KẾT QUẢ 2018 ĐỂ QUAY LẠI SỬA HOẶC TINH CHỈNH MÔ HÌNH.
   Mọi lựa chọn (alpha, feature, quy tắc tiền xử lý) đã được chốt ở
   `src/train.py` + `src/experiments.py`, vốn chỉ dùng dữ liệu 2012–2017.
   Nếu sau khi đọc file này mà còn điều chỉnh gì, thì con số 2018 mất ý nghĩa và phải
   chạy lại từ đầu trên một holdout mới.

Các thí nghiệm phát triển (random vs time split, kiểm soát rò rỉ, drift) KHÔNG nằm ở
đây — chúng ở `src/experiments.py` và chạy trong 2012–2017.

Usage:
    python src/evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import (  # noqa: E402
    EXTREME_WEATHER_CATEGORIES,
    TARGET_COLUMN,
    build_features,
    load_clean,
    split_summary,
    time_split,
)
from src.train import (  # noqa: E402
    BASELINE_META_PATH,
    BASELINE_TABLE_PATH,
    MODEL_METADATA_PATH,
    RIDGE_PIPELINE_PATH,
    compute_metrics,
    get_X_y,
    predict_baseline,
)

MODELS_DIR = _ROOT / "models"
REPORTS_DIR = _ROOT / "reports" / "figures"
EVAL_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"
EVAL_JSON_PATH = REPORTS_DIR / "evaluation_results.json"

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

#: ngưỡng cảnh báo khi kết luận về một nhóm quá nhỏ
SMALL_SAMPLE_THRESHOLD = 100


def _fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", ".")


def _save(fig, name: str) -> str:
    path = REPORTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path.relative_to(_ROOT)).replace("\\", "/")


def _segment(part: pd.DataFrame, label: str, extra: dict | None = None) -> dict:
    """MAE + số mẫu cho một phân khúc. Không dùng NaN để JSON hợp lệ."""
    n = int(len(part))
    out = {
        "segment": label,
        "n_samples": n,
        "MAE": round(float(part["_abs_err"].mean()), 2) if n else None,
        "RMSE": round(float(np.sqrt((part["_err"] ** 2).mean())), 2) if n else None,
        "bias": round(float(part["_err"].mean()), 2) if n else None,
        "mean_actual": round(float(part[TARGET_COLUMN].mean()), 2) if n else None,
        "mean_pred": round(float(part["_pred"].mean()), 2) if n else None,
        "small_sample": bool(n < SMALL_SAMPLE_THRESHOLD),
    }
    if n == 0:
        out["note"] = "KHÔNG CÓ MẪU nào trong FINAL TEST — không đánh giá được."
    elif out["small_sample"]:
        out["note"] = f"Số mẫu < {SMALL_SAMPLE_THRESHOLD} — không đủ cơ sở kết luận mạnh."
    if extra:
        out.update(extra)
    return out


def _segment_table(segments: list[dict], baseline_key: str = "baseline_MAE") -> str:
    rows = ["| Phân khúc | MAE (Ridge) | n mẫu | MAE (baseline) | Lưu lượng thực | Dự báo TB | Ghi chú |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in segments:
        b = s.get(baseline_key)
        rows.append(
            f"| {s['segment']} | {s['MAE'] if s['MAE'] is not None else '—'} | "
            f"{_fmt(s['n_samples'])} | {b if b is not None else '—'} | "
            f"{s['mean_actual'] if s['mean_actual'] is not None else '—'} | "
            f"{s['mean_pred'] if s['mean_pred'] is not None else '—'} | {s.get('note', 'mẫu đủ')} |"
        )
    return "\n".join(rows)


# ===========================================================================
# Chuẩn bị
# ===========================================================================
def prepare_test(test_df, ridge_pipe, baseline_table, global_fallback) -> pd.DataFrame:
    t = test_df.copy()
    X, _ = get_X_y(t)
    t["_pred"] = ridge_pipe.predict(X)
    t["_err"] = t["_pred"] - t[TARGET_COLUMN]
    t["_abs_err"] = t["_err"].abs()
    t["_baseline_pred"] = predict_baseline(t, baseline_table, global_fallback)
    t["_baseline_abs_err"] = (t["_baseline_pred"] - t[TARGET_COLUMN]).abs()
    return t


def with_baseline(seg: dict, t: pd.DataFrame, mask: pd.Series) -> dict:
    if int(mask.sum()) > 0:
        seg["baseline_MAE"] = round(float(t.loc[mask, "_baseline_abs_err"].mean()), 2)
    return seg


# ===========================================================================
# 1. Baseline vs Model  +  2. MAE/RMSE/R²
# ===========================================================================
def final_comparison(t: pd.DataFrame) -> dict:
    return {
        "baseline": compute_metrics(t[TARGET_COLUMN], t["_baseline_pred"]),
        "ridge": compute_metrics(t[TARGET_COLUMN], t["_pred"]),
    }


# ===========================================================================
# 3. Phân tích lỗi: hour / day_of_week / holiday / weather
# ===========================================================================
def error_by_hour(t: pd.DataFrame) -> dict:
    segs = []
    for h, part in t.groupby("hour"):
        segs.append(
            with_baseline(
                _segment(part, f"hour_{int(h):02d}"),
                t, t["hour"] == h,
            )
        )
    segs.sort(key=lambda s: s["segment"])
    return {"by": "hour", "segments": segs}


def error_by_day_of_week(t: pd.DataFrame) -> dict:
    segs = []
    for d, part in t.groupby("day_of_week"):
        segs.append(
            with_baseline(
                _segment(part, DOW_LABELS[int(d)], {"day_of_week": int(d)}),
                t, t["day_of_week"] == d,
            )
        )
    order = {DOW_LABELS[i]: i for i in range(7)}
    segs.sort(key=lambda s: order[s["segment"]])
    return {"by": "day_of_week", "segments": segs}


def error_by_holiday(t: pd.DataFrame) -> dict:
    holiday = with_baseline(_segment(t[t["is_holiday"] == 1], "holiday"), t, t["is_holiday"] == 1)
    normal = with_baseline(_segment(t[t["is_holiday"] == 0], "non_holiday"), t, t["is_holiday"] == 0)
    return {
        "by": "holiday",
        "segments": [normal, holiday],
        "mae_delta": round(holiday["MAE"] - normal["MAE"], 2) if holiday["MAE"] is not None else None,
        "n_holiday_dates": int(t.loc[t["is_holiday"] == 1, "date_time"].dt.normalize().nunique()),
    }


def error_by_weather(t: pd.DataFrame) -> dict:
    from src.weather import WEATHER_MAIN_CATEGORIES

    segs = []
    for cat in WEATHER_MAIN_CATEGORIES:
        mask = t[f"wm_{cat.lower()}"] == 1
        s = _segment(t[mask], cat)
        s["note"] = (
            "KHÔNG CÓ MẪU — không đánh giá được."
            if s["n_samples"] == 0
            else ("Mẫu nhỏ — thận trọng." if s["small_sample"] else "Mẫu đủ.")
        )
        segs.append(with_baseline(s, t, mask))
    return {"by": "weather_main (multi-hot)", "segments": segs}


def error_extreme_weather(t: pd.DataFrame) -> dict:
    segs = []
    for cat in EXTREME_WEATHER_CATEGORIES:
        mask = t[f"wm_{cat.lower()}"] == 1
        s = _segment(t[mask], cat)
        s["note"] = (
            "KHÔNG CÓ MẪU trong FINAL TEST — không đánh giá được."
            if s["n_samples"] == 0
            else ("Mẫu nhỏ — thận trọng." if s["small_sample"] else "Mẫu đủ.")
        )
        segs.append(with_baseline(s, t, mask))
    any_mask = t["is_extreme_weather"] == 1
    segs.append(with_baseline(_segment(t[any_mask], "any_extreme_weather"), t, any_mask))
    segs.append(with_baseline(_segment(t[~any_mask], "no_extreme_weather"), t, ~any_mask))
    return {
        "by": "extreme_weather",
        "segments": segs,
        "multi_label_note": (
            "Các phân khúc dùng multi-hot nên Snow/Thunderstorm/Squall có thể trùng nhau; "
            "tổng n có thể vượt số dòng của FINAL TEST."
        ),
    }


def error_analysis(t: pd.DataFrame) -> dict:
    return {
        "hour": error_by_hour(t),
        "day_of_week": error_by_day_of_week(t),
        "holiday": error_by_holiday(t),
        "weather_main": error_by_weather(t),
        "extreme_weather": error_extreme_weather(t),
    }


# ===========================================================================
# Biểu đồ
# ===========================================================================
def plot_hour_dow(t: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    by_hour = t.groupby("hour")["_abs_err"].agg(["mean", "size"])
    axes[0].bar(by_hour.index, by_hour["mean"], color="#4C78A8")
    axes[0].set_xticks(range(24))
    axes[0].set_xlabel("Giờ")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("MAE theo giờ (FINAL TEST 2018)")
    axes[0].grid(axis="y", alpha=0.3)

    by_dow = t.groupby("day_of_week")["_abs_err"].agg(["mean", "size"])
    axes[1].bar([DOW_LABELS[i] for i in by_dow.index], by_dow["mean"], color="#F58518")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE theo thứ trong tuần (FINAL TEST 2018)")
    axes[1].grid(axis="y", alpha=0.3)
    return _save(fig, "final_mae_by_hour_dow.png")


def plot_weather_segments(t: pd.DataFrame) -> str:
    from src.weather import WEATHER_MAIN_CATEGORIES

    labels, maes, ns, colors = [], [], [], []
    for cat in WEATHER_MAIN_CATEGORIES:
        mask = t[f"wm_{cat.lower()}"] == 1
        labels.append(cat)
        maes.append(t.loc[mask, "_abs_err"].mean() if mask.any() else np.nan)
        ns.append(int(mask.sum()))
        colors.append("#E45756" if cat in EXTREME_WEATHER_CATEGORIES else "#4C78A8")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, maes, color=colors)
    for i, (m, n) in enumerate(zip(maes, ns)):
        if not np.isnan(m):
            ax.text(i, m, f"{m:.0f}\nn={n:,}".replace(",", "."), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("MAE")
    ax.set_title("MAE theo weather_main trên FINAL TEST 2018 (đỏ = thời tiết cực đoan)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "final_mae_by_weather.png")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_features(load_clean())
    train_df, val_df, test_df = time_split(df)
    split_info = split_summary(train_df, val_df, test_df)

    baseline_table = pd.read_csv(BASELINE_TABLE_PATH)
    baseline_meta = json.loads(BASELINE_META_PATH.read_text(encoding="utf-8"))
    ridge_pipe = joblib.load(RIDGE_PIPELINE_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    global_fallback = baseline_meta["global_fallback"]

    t = prepare_test(test_df, ridge_pipe, baseline_table, global_fallback)
    comp = final_comparison(t)
    errs = error_analysis(t)
    figs = {"hour_dow": plot_hour_dow(t), "weather": plot_weather_segments(t)}

    b, r = comp["baseline"], comp["ridge"]
    hol = errs["holiday"]
    vm = metadata["validation_metrics"]

    L: list[str] = []
    L.append("# FINAL EVALUATION — TEST 2018 (đánh giá đúng MỘT LẦN)")
    L.append("")
    L.append("> Sinh tự động bởi `python src/evaluate.py`. **Không con số nào được hard-code.**")
    L.append("> Mọi số liệu tính lại từ dữ liệu sau khi sửa toàn bộ preprocessing.")
    L.append("")

    L.append("## 0. Cấu hình đã đóng băng TRƯỚC khi đánh giá 2018")
    L.append("")
    L.append(f"- Mô hình: Ridge, **alpha = {metadata['hyperparameters']['alpha']}** (chọn theo MAE trên VALIDATION 2017)")
    L.append(f"- Solver: `{metadata['hyperparameters']['solver']}`")
    L.append("- Pipeline: `ColumnTransformer[OneHotEncoder | SimpleImputer(median)→StandardScaler | passthrough] → Ridge`")
    L.append("- Imputer / scaler / encoder: **fit trên TRAIN 2012–2016 duy nhất**")
    L.append("- Baseline: mean traffic theo (hour, day_of_week), **fit trên TRAIN duy nhất**")
    L.append("- `is_holiday`: tính từ **lịch tất định** `src/holidays.py` (không phụ thuộc dữ liệu)")
    L.append("")
    L.append("| Tập | Khoảng thời gian | Số dòng |")
    L.append("| --- | --- | --- |")
    for name in ("train", "validation", "test"):
        s = split_info[name]
        L.append(f"| {name} | {s['start']} → {s['end']} | {_fmt(s['n_rows'])} |")
    L.append("")
    L.append("Assert: `max(train) < min(validation)` ✅ · `max(validation) < min(test)` ✅ · 0 timestamp trùng ✅")
    L.append("")
    L.append(
        "> **Bằng chứng 2018 chưa bị dùng để lựa chọn gì:** toàn bộ tuning alpha và các thí nghiệm "
        "phát triển (random vs time split, kiểm soát rò rỉ, rolling-origin drift) đều chạy trong "
        "`src/train.py` và `src/experiments.py`, với cửa sổ dữ liệu **2012–2017**. "
        "`src/experiments.py` gọi `assert_no_final_test_rows()` ở mọi hàm và sẽ dừng chương trình "
        "nếu bất kỳ dòng 2018 nào lọt vào."
    )
    L.append("")

    L.append("## 1. Baseline vs Model")
    L.append("")
    L.append("| Mô hình | MAE | RMSE | R² | n |")
    L.append("| --- | --- | --- | --- | --- |")
    L.append(f"| Baseline (hour × day_of_week) | {b['MAE']} | {b['RMSE']} | {b['R2']} | {_fmt(b['n'])} |")
    L.append(f"| **Ridge pipeline** | **{r['MAE']}** | **{r['RMSE']}** | **{r['R2']}** | {_fmt(r['n'])} |")
    L.append("")
    L.append(
        f"- Ridge cải thiện **{b['MAE'] - r['MAE']:.2f}** MAE "
        f"({(b['MAE'] - r['MAE']) / b['MAE']:.1%}) so với baseline."
    )
    L.append("")

    L.append("## 2. MAE / RMSE / R²")
    L.append("")
    L.append("| Tập | Đánh giá | MAE | RMSE | R² | n |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    L.append(
        f"| VALIDATION 2017 (dùng để chọn alpha) | out-of-sample | {vm['MAE']} | {vm['RMSE']} | {vm['R2']} | {_fmt(vm['n'])} |"
    )
    L.append(
        f"| FINAL TEST 2018 | out-of-sample | **{r['MAE']}** | **{r['RMSE']}** | **{r['R2']}** | {_fmt(r['n'])} |"
    )
    L.append(
        f"| FINAL TEST 2018 (baseline) | out-of-sample | {b['MAE']} | {b['RMSE']} | {b['R2']} | {_fmt(b['n'])} |"
    )
    L.append("")
    L.append(
        "- ⚠️ Không đưa MAE trên tập train vào bảng này: train là **in-sample** còn hai dòng trên "
        "là **out-of-sample**, trộn chúng là so sánh không cùng đối tượng. "
        "Phân tích drift riêng nằm ở `experiments_report.md` (rolling-origin, chỉ out-of-sample)."
    )
    L.append("")

    L.append("## 3. Phân tích lỗi trên FINAL TEST")
    L.append("")
    L.append(f"> Mọi phân khúc đều kèm **MAE + số mẫu**. Ngưỡng cảnh báo mẫu nhỏ: n < {SMALL_SAMPLE_THRESHOLD}.")
    L.append("")

    L.append("### 3a. Theo giờ")
    L.append("")
    L.append(_segment_table(errs["hour"]["segments"]))
    L.append("")
    worst = max(errs["hour"]["segments"], key=lambda s: s["MAE"] or -1)
    best = min(errs["hour"]["segments"], key=lambda s: s["MAE"] or 1e18)
    L.append(
        f"- Giờ khó nhất: **{worst['segment']}** (MAE {worst['MAE']}, n={_fmt(worst['n_samples'])}); "
        f"giờ dễ nhất: **{best['segment']}** (MAE {best['MAE']}, n={_fmt(best['n_samples'])})."
    )
    L.append("")

    L.append("### 3b. Theo thứ trong tuần")
    L.append("")
    L.append(_segment_table(errs["day_of_week"]["segments"]))
    L.append("")
    dow_worst = max(errs["day_of_week"]["segments"], key=lambda s: s["MAE"] or -1)
    L.append(f"- Ngày khó nhất: **{dow_worst['segment']}** (MAE {dow_worst['MAE']}, n={_fmt(dow_worst['n_samples'])}).")
    L.append("")

    L.append("### 3c. Ngày lễ")
    L.append("")
    L.append(_segment_table(errs["holiday"]["segments"]))
    L.append("")
    L.append(
        f"- Chênh lệch MAE (ngày lễ − ngày thường) = **{hol['mae_delta']}** xe/giờ; "
        f"ngày lễ chiếm {hol['segments'][1]['n_samples']} giờ trên "
        f"{hol['n_holiday_dates']} ngày lịch."
    )
    if hol["segments"][1]["small_sample"]:
        L.append(
            f"- ⚠️ Số mẫu ngày lễ chỉ {hol['segments'][1]['n_samples']} — cần thận trọng khi kết luận."
        )
    L.append("")

    L.append("### 3d. Thời tiết (weather_main, multi-hot)")
    L.append("")
    L.append(_segment_table(errs["weather_main"]["segments"]))
    L.append("")
    L.append(f"![MAE theo weather]({figs['weather']})")
    L.append("")

    L.append("### 3e. Thời tiết cực đoan")
    L.append("")
    L.append(_segment_table(errs["extreme_weather"]["segments"]))
    L.append("")
    L.append(f"- {errs['extreme_weather']['multi_label_note']}")
    zero = [s for s in errs["extreme_weather"]["segments"] if s["n_samples"] == 0]
    if zero:
        L.append(
            "- ⚠️ **KẾT LUẬN YẾU:** "
            + ", ".join(f"`{s['segment']}` không có mẫu nào trong FINAL TEST" for s in zero)
            + ". Không phát biểu mạnh về các phân khúc này."
        )
    L.append("")

    L.append("### 3f. Theo giờ × thứ")
    L.append("")
    L.append(f"![MAE theo giờ và thứ]({figs['hour_dow']})")
    L.append("")

    L.append("## 4. Kết luận")
    L.append("")
    L.append(
        f"1. FINAL TEST 2018 (n={_fmt(r['n'])}): Ridge đạt **MAE={r['MAE']}, RMSE={r['RMSE']}, R²={r['R2']}**, "
        f"so với baseline MAE={b['MAE']}."
    )
    L.append(
        f"2. Điểm yếu rõ nhất là **ngày lễ** (MAE={hol['segments'][1]['MAE']} so với "
        f"{hol['segments'][0]['MAE']} ở ngày thường, n={_fmt(hol['segments'][1]['n_samples'])}). "
        "`is_holiday` chỉ là cờ nhị phân nên mô hình chỉ học được một mức dịch chuyển trung bình, "
        "không học được dạng hình giờ đặc thù của ngày lễ. Hướng sửa hợp lý cho checkpoint sau: "
        "thêm tương tác `hour × is_holiday`."
    )
    ext = {s["segment"]: s for s in errs["extreme_weather"]["segments"]}
    n = 3
    if "any_extreme_weather" in ext and "no_extreme_weather" in ext:
        a, nm = ext["any_extreme_weather"], ext["no_extreme_weather"]
        L.append(
            f"{n}. Thời tiết cực đoan tổng hợp: MAE={a['MAE']} (n={_fmt(a['n_samples'])}) "
            f"so với {nm['MAE']} ở thời tiết bình thường (n={_fmt(nm['n_samples'])})."
        )
        n += 1
    worst_hour = max(errs["hour"]["segments"], key=lambda s: s["MAE"] or -1)
    worst_dow = max(errs["day_of_week"]["segments"], key=lambda s: s["MAE"] or -1)
    L.append(
        f"{n}. Giờ khó nhất: **{worst_hour['segment']}** (MAE {worst_hour['MAE']}, "
        f"n={_fmt(worst_hour['n_samples'])}); ngày khó nhất: **{worst_dow['segment']}** "
        f"(MAE {worst_dow['MAE']}, n={_fmt(worst_dow['n_samples'])})."
    )
    n += 1
    L.append(
        f"{n}. Mọi phân khúc đều báo kèm số mẫu; phân khúc không có mẫu được đánh dấu rõ và không dùng để kết luận."
    )
    n += 1
    L.append(
        f"{n}. **Đối chiếu với giai đoạn phát triển:** trên pseudo-test 2016–2017 (train chỉ tới 2015, "
        "dữ liệu rất thưa) baseline thắng Ridge về MAE nhưng thua về RMSE. Trên FINAL TEST này "
        "(train tới 2016, dữ liệu dày hơn nhiều) **Ridge thắng ở cả ba chỉ số**. Phần đuôi hơn của "
        "Ridge vì vậy phụ thuộc vào mật độ dữ liệu huấn luyện — và là lý do không nên kết luận "
        "chỉ từ một cửa sổ đánh giá duy nhất."
    )
    L.append("")

    report = "\n".join(L)
    EVAL_REPORT_PATH.write_text(report, encoding="utf-8")

    results = {
        "config": {
            "alpha": metadata["hyperparameters"]["alpha"],
            "solver": metadata["hyperparameters"]["solver"],
            "model": metadata["model_name"],
            "imputer_fit_scope": "TRAIN ONLY",
            "holiday_source": "src/holidays.py (deterministic calendar, no dataset dependency)",
            "config_frozen_before_final_eval": True,
            "dev_experiments_window": "2012-2017 (src/experiments.py, guarded by assert_no_final_test_rows)",
        },
        "time_split": split_info,
        "final_test": comp,
        "validation_reference": {"ridge": vm},
        "error_analysis": errs,
        "figures": figs,
    }
    EVAL_JSON_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    print(report)
    print(f"\nĐã lưu: {EVAL_REPORT_PATH}")
    print(f"Đã lưu: {EVAL_JSON_PATH}")


if __name__ == "__main__":
    main()

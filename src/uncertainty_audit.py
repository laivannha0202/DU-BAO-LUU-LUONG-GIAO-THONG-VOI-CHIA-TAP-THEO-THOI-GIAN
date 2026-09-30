"""
src/uncertainty_audit.py

KIỂM TOÁN BẤT ĐỊNH của phép so sánh Ridge vs baseline — CHẠY SAU `src/evaluate.py`.

Vì sao script này TÁCH RIÊNG khỏi `src/evaluate.py` và `src/freeze_serving_policy.py`
-------------------------------------------------------------------------------
- `evaluate.py` **đánh giá** FINAL TEST 2018 và ghi kết quả chính thức.
- `uncertainty_audit.py` (file này) chỉ **đo lại độ bất định** quanh những con số đó:
  khoảng tin cậy, tỉ lệ bootstrap mà Ridge thắng, và tính nhất quán theo từng tháng.
  Nó KHÔNG quyết định gì và KHÔNG dùng 2018 để chọn mô hình / feature / alpha /
  serving policy.

Nguyên tắc bất di bất dịch
---------------------------
1. **CHẠY SAU `evaluate.py`** — script từ chối chạy nếu chưa có
   `reports/figures/evaluation_results.json`.
2. **CHỈ ĐO, KHÔNG QUYẾT ĐỊNH.** Không ghi đè `evaluation_results.json`, không ghi
   vào `models/`, không train lại mô hình đã đóng băng.
3. Metric dùng ở đây là **RAW MODEL** (Ridge trả về trực tiếp), đúng quy ước đã
   đóng băng ở `src/freeze_serving_policy.py`.
4. **Bootstrap theo khối ngày lịch.** Các quan sát trong cùng một ngày lịch KHÔNG độc
   lập với nhau (cùng thời tiết, cùng ngày làm việc), nên lấy mẫu lại từng dòng sẽ
   tạo khoảng tin cậy **hẹp hơn thực tế**. Vì vậy ta lấy mẫu lại theo **ngày**.
5. Phần dữ liệu dev (pseudo-test 2016–2017 và các fold rolling-origin) được bảo vệ
   bằng `assert_no_final_test_rows()` y hệt `src/experiments.py`.

Usage:
    python src/evaluate.py           # TRƯỚC: đánh giá FINAL TEST 2018
    python src/uncertainty_audit.py  # SAU: đo bất định quanh kết quả đó
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

from src.experiments import (  # noqa: E402
    MONTH_LABELS,
    NIGHT_HOURS,
    ROLLING_FOLDS,
    assert_no_final_test_rows,
    dev_window,
)
from src.features import (  # noqa: E402
    TARGET_COLUMN,
    build_features,
    load_clean,
    time_split,
)
from src.train import (  # noqa: E402
    BASELINE_META_PATH,
    BASELINE_TABLE_PATH,
    MODEL_METADATA_PATH,
    REPORTS_DIR,
    RIDGE_PIPELINE_PATH,
    compute_metrics,
    get_X_y,
    make_ridge_pipeline,
    predict_baseline,
    train_baseline,
)

EVAL_JSON_PATH = REPORTS_DIR / "evaluation_results.json"
AUDIT_JSON_PATH = REPORTS_DIR / "uncertainty_audit.json"
AUDIT_MD_PATH = REPORTS_DIR / "uncertainty_audit.md"
FIGURE_PATH = REPORTS_DIR / "uncertainty_bootstrap.png"

#: Số lần lấy mẫu lại. >= 2000 theo yêu cầu đề.
N_BOOTSTRAP = 4000

#: Seed CỐ ĐỊNH — kết quả phải tái lập được y hệt mỗi lần chạy.
BOOTSTRAP_SEED = 20240501

#: Mức tin cậy của khoảng bootstrap phần trăm.
CI_LEVEL = 95.0

#: Ngưỡng cảnh báo khi một phân khúc theo tháng quá nhỏ để kết luận.
SMALL_SEGMENT = 100


# ===========================================================================
# Bootstrap theo khối ngày lịch
# ===========================================================================
def _day_index(ts: pd.Series) -> np.ndarray:
    """Mã số nguyên cho mỗi ngày lịch (để gom nhóm)."""
    return ts.dt.normalize().astype("int64").to_numpy()


def paired_block_bootstrap(
    part: pd.DataFrame,
    err_baseline: np.ndarray,
    err_ridge: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    ci: float = CI_LEVEL,
) -> dict:
    """Bootstrap CẶP theo khối ngày lịch cho hiệu MAE và hiệu RMSE.

    Thống kê: ``baseline − Ridge`` (dương = Ridge tốt hơn).
    Cùng một tập ngày được lấy mẫu lại cho **cả hai** mô hình — đây là bootstrap
    cặp, nên nó loại bỏ phần sai số chung do điều kiện thời tiết/ngày.
    """
    day = _day_index(part["date_time"])
    uniq, inverse = np.unique(day, return_inverse=True)
    n_days = int(uniq.size)

    # Tổng và số đếm theo từng ngày -> gom trước rồi mới lấy mẫu lại (nhanh và chính xác)
    abs_b = np.abs(err_baseline)
    abs_r = np.abs(err_ridge)
    sq_b = (err_baseline ** 2).astype(float)
    sq_r = (err_ridge ** 2).astype(float)
    sum_abs_b = np.bincount(inverse, weights=abs_b, minlength=n_days)
    sum_abs_r = np.bincount(inverse, weights=abs_r, minlength=n_days)
    sum_sq_b = np.bincount(inverse, weights=sq_b, minlength=n_days)
    sum_sq_r = np.bincount(inverse, weights=sq_r, minlength=n_days)
    count = np.bincount(inverse, minlength=n_days).astype(float)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_days, size=(n_boot, n_days))
    n_sel = count[picks].sum(axis=1)

    mae_b = sum_abs_b[picks].sum(axis=1) / n_sel
    mae_r = sum_abs_r[picks].sum(axis=1) / n_sel
    rmse_b = np.sqrt(sum_sq_b[picks].sum(axis=1) / n_sel)
    rmse_r = np.sqrt(sum_sq_r[picks].sum(axis=1) / n_sel)
    d_mae = mae_b - mae_r
    d_rmse = rmse_b - rmse_r

    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0

    def _summary(point: float, draws: np.ndarray, unit: str) -> dict:
        return {
            "unit": unit,
            "point_estimate": round(float(point), 2),
            "ci_low": round(float(np.percentile(draws, lo_q)), 2),
            "ci_high": round(float(np.percentile(draws, hi_q)), 2),
            "ci_level_pct": ci,
            "boot_sd": round(float(draws.std(ddof=0)), 2),
            "share_ridge_better_pct": round(float((draws > 0).mean() * 100), 2),
            "ci_excludes_zero": bool(
                np.percentile(draws, lo_q) > 0 or np.percentile(draws, hi_q) < 0
            ),
        }

    point_mae = float(abs_b.mean() - abs_r.mean())
    point_rmse = float(np.sqrt((sq_b.mean())) - np.sqrt((sq_r.mean())))
    return {
        "n_rows": int(len(part)),
        "n_blocks_days": n_days,
        "n_bootstrap": int(n_boot),
        "block_unit": "calendar_day",
        "difference_convention": "baseline - ridge (positive = Ridge better)",
        "MAE": _summary(point_mae, d_mae, "vehicles/hour"),
        "RMSE": _summary(point_rmse, d_rmse, "vehicles/hour"),
        "_draws_mae": d_mae,
    }


def by_month_table(part: pd.DataFrame, err_baseline: np.ndarray, err_ridge: np.ndarray) -> dict:
    """MAE theo từng tháng của cửa sổ đánh giá, và đếm số tháng Ridge thắng."""
    work = pd.DataFrame({
        "month": part["month"].to_numpy(),
        "abs_b": np.abs(err_baseline),
        "abs_r": np.abs(err_ridge),
    })
    rows = []
    n_win = 0
    for month, grp in work.groupby("month"):
        mae_b = float(grp["abs_b"].mean())
        mae_r = float(grp["abs_r"].mean())
        ridge_wins = mae_r < mae_b
        n_win += int(ridge_wins)
        rows.append({
            "month": int(month),
            "month_label": MONTH_LABELS[int(month) - 1],
            "n_samples": int(len(grp)),
            "MAE_baseline": round(mae_b, 2),
            "MAE_ridge": round(mae_r, 2),
            "delta_MAE": round(mae_b - mae_r, 2),
            "ridge_better": bool(ridge_wins),
            "small_sample": bool(len(grp) < SMALL_SEGMENT),
        })
    return {
        "by_month": rows,
        "n_months": len(rows),
        "n_months_ridge_better": n_win,
        "n_months_baseline_better": len(rows) - n_win,
    }


# ===========================================================================
# Chuẩn bị dữ liệu cho từng cửa sổ đánh giá
# ===========================================================================
def _errors(pipe, baseline_table, fallback, part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = part[TARGET_COLUMN].to_numpy(dtype=float)
    pred_r = np.asarray(pipe.predict(get_X_y(part)[0]), dtype=float)
    pred_b = predict_baseline(part, baseline_table, fallback)
    return pred_b - y, pred_r - y


def night_hour_analysis(part: pd.DataFrame, err_baseline: np.ndarray, err_ridge: np.ndarray) -> dict:
    """MAE theo giờ trên FINAL TEST — MÔ TẢ, không dùng để quyết định gì.

    Phần kiểm chứng trên dữ liệu dev nằm ở `src/experiments.py`
    (`experiment_night_hour_failure`). Ở đây chỉ đo lại trên 2018 để báo cáo.
    """
    work = pd.DataFrame({
        "hour": part["hour"].to_numpy(),
        "abs_b": np.abs(err_baseline),
        "abs_r": np.abs(err_ridge),
        "y": part[TARGET_COLUMN].to_numpy(dtype=float),
    })
    by_hour = []
    for hour, grp in work.groupby("hour"):
        mae_r = float(grp["abs_r"].mean())
        mae_b = float(grp["abs_b"].mean())
        mean_actual = float(grp["y"].mean())
        by_hour.append({
            "hour": int(hour),
            "n_samples": int(len(grp)),
            "MAE_ridge": round(mae_r, 2),
            "MAE_baseline": round(mae_b, 2),
            "mean_actual": round(mean_actual, 2),
            "rel_MAE_ridge": round(mae_r / mean_actual, 4) if mean_actual > 0 else None,
            "rel_MAE_baseline": round(mae_b / mean_actual, 4) if mean_actual > 0 else None,
            "ridge_worse": bool(mae_r > mae_b),
        })
    by_hour.sort(key=lambda r: r["hour"])
    n_worse = sum(1 for r in by_hour if r["ridge_worse"])
    night = work[work["hour"].isin(NIGHT_HOURS)]
    day = work[~work["hour"].isin(NIGHT_HOURS)]
    return {
        "n_hours": len(by_hour),
        "n_hours_ridge_worse": n_worse,
        "n_hours_ridge_better": len(by_hour) - n_worse,
        "night_hours": list(NIGHT_HOURS),
        "night": {
            "n_samples": int(len(night)),
            "MAE_ridge": round(float(night["abs_r"].mean()), 2) if len(night) else None,
            "MAE_baseline": round(float(night["abs_b"].mean()), 2) if len(night) else None,
            "rel_MAE_ridge": (
                round(float(night["abs_r"].mean() / night["y"].mean()), 4) if len(night) else None
            ),
            "rel_MAE_baseline": (
                round(float(night["abs_b"].mean() / night["y"].mean()), 4) if len(night) else None
            ),
        },
        "daytime": {
            "n_samples": int(len(day)),
            "MAE_ridge": round(float(day["abs_r"].mean()), 2) if len(day) else None,
            "MAE_baseline": round(float(day["abs_b"].mean()), 2) if len(day) else None,
            "rel_MAE_ridge": (
                round(float(day["abs_r"].mean() / day["y"].mean()), 4) if len(day) else None
            ),
            "rel_MAE_baseline": (
                round(float(day["abs_b"].mean() / day["y"].mean()), 4) if len(day) else None
            ),
        },
        "by_hour": by_hour,
    }


def final_test_window(df: pd.DataFrame, frozen_pipe, baseline_table, fallback) -> dict:
    """FINAL TEST 2018 — dùng đúng artifact ĐÃ ĐÓNG BĂNG, không fit lại."""
    _train, _val, test_df = time_split(df)
    err_b, err_r = _errors(frozen_pipe, baseline_table, fallback, test_df)
    boot = paired_block_bootstrap(test_df, err_b, err_r)
    out = {
        "name": "FINAL TEST 2018",
        "kind": "final_test",
        "n_rows": int(len(test_df)),
        "range": [str(test_df["date_time"].min()), str(test_df["date_time"].max())],
        "metrics": {
            "baseline": compute_metrics(
                test_df[TARGET_COLUMN], test_df[TARGET_COLUMN] + err_b
            ),
            "ridge": compute_metrics(test_df[TARGET_COLUMN], test_df[TARGET_COLUMN] + err_r),
        },
        "bootstrap": {k: v for k, v in boot.items() if not k.startswith("_")},
        "months": by_month_table(test_df, err_b, err_r),
        "night_hour": night_hour_analysis(test_df, err_b, err_r),
        "_draws_mae": boot["_draws_mae"],
    }
    return out


def dev_pseudo_test_window(dev: pd.DataFrame, alpha: float) -> dict:
    """Pseudo-test 2016–2017 (train chỉ tới 2015) — dùng để xem hiệu ứng có bền không."""
    assert_no_final_test_rows(dev, "uncertainty:pseudo_test")
    tr = dev[dev["date_time"] <= pd.Timestamp("2015-12-31 23:59:59")]
    te = dev[dev["date_time"] > pd.Timestamp("2015-12-31 23:59:59")]
    pipe = make_ridge_pipeline(alpha)
    pipe.fit(*get_X_y(tr))
    err_b, err_r = _errors(pipe, train_baseline(tr), float(tr[TARGET_COLUMN].mean()), te)
    boot = paired_block_bootstrap(te, err_b, err_r, seed=BOOTSTRAP_SEED + 1)
    return {
        "name": "pseudo-test 2016-2017 (train <= 2015)",
        "kind": "dev_pseudo_test",
        "n_rows": int(len(te)),
        "range": [str(te["date_time"].min()), str(te["date_time"].max())],
        "metrics": {
            "baseline": compute_metrics(te[TARGET_COLUMN], te[TARGET_COLUMN] + err_b),
            "ridge": compute_metrics(te[TARGET_COLUMN], te[TARGET_COLUMN] + err_r),
        },
        "bootstrap": {k: v for k, v in boot.items() if not k.startswith("_")},
        "months": by_month_table(te, err_b, err_r),
        "_draws_mae": boot["_draws_mae"],
    }


def dev_rolling_folds(dev: pd.DataFrame, alpha: float) -> list[dict]:
    """Từng fold rolling-origin — CHỈ dữ liệu dev, dùng đúng định nghĩa fold như
    `src/experiments.py` (dùng chung hằng số `ROLLING_FOLDS`)."""
    out = []
    for i, spec in enumerate(ROLLING_FOLDS):
        assert_no_final_test_rows(dev, f"uncertainty:fold{spec['fold']}")
        train_end = pd.Timestamp(spec["train_end"])
        tr = dev[dev["date_time"] <= train_end]
        te = dev[
            (dev["date_time"] > train_end) & (dev["year"] == spec["test_year"])
        ]
        if len(tr) == 0 or len(te) == 0:
            continue
        pipe = make_ridge_pipeline(alpha)
        pipe.fit(*get_X_y(tr))
        err_b, err_r = _errors(pipe, train_baseline(tr), float(tr[TARGET_COLUMN].mean()), te)
        boot = paired_block_bootstrap(te, err_b, err_r, seed=BOOTSTRAP_SEED + 10 + i)
        out.append({
            "name": f"fold {spec['fold']} — test year {spec['test_year']}",
            "kind": "dev_rolling_fold",
            "test_year": int(spec["test_year"]),
            "n_rows": int(len(te)),
            "range": [str(te["date_time"].min()), str(te["date_time"].max())],
            "metrics": {
                "baseline": compute_metrics(te[TARGET_COLUMN], te[TARGET_COLUMN] + err_b),
                "ridge": compute_metrics(te[TARGET_COLUMN], te[TARGET_COLUMN] + err_r),
            },
            "bootstrap": {k: v for k, v in boot.items() if not k.startswith("_")},
            "months": by_month_table(te, err_b, err_r),
            "_draws_mae": boot["_draws_mae"],
        })
    return out


# ===========================================================================
# Kết luận — TÍNH TỪ SỐ LIỆU, không viết trước
# ===========================================================================
def build_verdict(final: dict, dev_windows: list[dict]) -> dict:
    """Sinh kết luận từ chính khoảng tin cậy và tỉ lệ thắng theo tháng."""
    f_mae = final["bootstrap"]["MAE"]
    f_rmse = final["bootstrap"]["RMSE"]
    f_months = final["months"]

    dev_consistent = all(w["bootstrap"]["MAE"]["share_ridge_better_pct"] > 50.0 for w in dev_windows)
    dev_signs = [w["bootstrap"]["MAE"]["point_estimate"] > 0 for w in dev_windows]

    if f_mae["ci_excludes_zero"] and f_mae["share_ridge_better_pct"] > 50.0:
        strength_2018 = (
            "Ridge tốt hơn baseline trên 2018, và khoảng tin cậy 95 % của hiệu MAE "
            "không chứa 0."
        )
    elif f_mae["share_ridge_better_pct"] > 50.0:
        strength_2018 = (
            "Điểm ước lượng nghiêng về Ridge, nhưng khoảng tin cậy 95 % **có chứa 0** — "
            "chưa đủ bằng chứng để kết luận thắng có ý nghĩa."
        )
    else:
        strength_2018 = (
            "Baseline tốt hơn Ridge trên 2018 theo cả điểm ước lượng và tỉ lệ bootstrap."
        )

    return {
        "final_test_MAE": {
            "point_estimate": f_mae["point_estimate"],
            "ci95": [f_mae["ci_low"], f_mae["ci_high"]],
            "share_ridge_better_pct": f_mae["share_ridge_better_pct"],
            "ci_excludes_zero": f_mae["ci_excludes_zero"],
        },
        "final_test_RMSE": {
            "point_estimate": f_rmse["point_estimate"],
            "ci95": [f_rmse["ci_low"], f_rmse["ci_high"]],
            "share_ridge_better_pct": f_rmse["share_ridge_better_pct"],
            "ci_excludes_zero": f_rmse["ci_excludes_zero"],
        },
        "months_ridge_better": f"{f_months['n_months_ridge_better']}/{f_months['n_months']}",
        "dev_windows_with_ridge_better": f"{sum(dev_signs)}/{len(dev_signs)}",
        "dev_direction_consistent": bool(dev_consistent),
        "verdict_2018": strength_2018,
        "conditions_required": [
            "Kết luận này chỉ nói về FINAL TEST 2018 (01/01–30/09), không suy rộng ra năm khác.",
            "Nó là so sánh trên cùng một tập đánh giá và cùng một tập huấn luyện 2012–2016.",
            "Metric là RAW MODEL; DEPLOYED PREDICTOR (max(0,·)) được báo riêng ở postprocess_audit.",
        ],
    }


# ===========================================================================
# Biểu đồ
# ===========================================================================
def plot_audit(final: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ax = axes[0]
    ax.hist(final["_draws_mae"], bins=60, color="#4C78A8", alpha=0.85, edgecolor="none")
    point = final["bootstrap"]["MAE"]["point_estimate"]
    lo = final["bootstrap"]["MAE"]["ci_low"]
    hi = final["bootstrap"]["MAE"]["ci_high"]
    ax.axvline(0.0, color="#B03A2E", lw=2, label="hòa nhau (0)")
    ax.axvline(point, color="#1B3A57", lw=2, label=f"ước lượng {point:.2f}")
    ax.axvspan(lo, hi, color="#F58518", alpha=0.18,
               label=f"CI 95 % [{lo:.2f}; {hi:.2f}]")
    ax.set_xlabel("Hiệu MAE = baseline - Ridge  (dương = Ridge tốt hơn)")
    ax.set_ylabel("Số lần lấy mẫu lại")
    ax.set_title("Block bootstrap theo ngày — FINAL TEST 2018")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    months = final["months"]["by_month"]
    labels = [m["month_label"] for m in months]
    xs = np.arange(len(months))
    width = 0.38
    ax.bar(xs - width / 2, [m["MAE_baseline"] for m in months], width,
           label="Baseline", color="#BAB0AC")
    ax.bar(xs + width / 2, [m["MAE_ridge"] for m in months], width,
           label="Ridge", color="#4C78A8")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("MAE")
    ax.set_title(
        f"MAE theo tháng 2018 — Ridge thắng ở "
        f"{final['months']['n_months_ridge_better']}/{final['months']['n_months']} tháng"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)
    return FIGURE_PATH.relative_to(_ROOT).as_posix()


# ===========================================================================
# Báo cáo Markdown
# ===========================================================================
def _fmt(n) -> str:
    if n is None:
        return "—"
    return f"{n:,}".replace(",", ".")


def render_markdown(result: dict) -> str:
    v = result["verdict"]
    L: list[str] = []
    L.append("# KIỂM TOÁN BẤT ĐỊNH — so sánh Ridge vs baseline")
    L.append("")
    L.append("> Sinh tự động bởi `python src/uncertainty_audit.py` — **chạy SAU**")
    L.append("> `python src/evaluate.py`.")
    L.append("")
    L.append("> **Script này KHÔNG quyết định gì.** Nó chỉ đo độ bất định quanh những con số")
    L.append("> đã có. Không ghi đè `evaluation_results.json`, không ghi vào `models/`,")
    L.append("> không train lại mô hình đã đóng băng.")
    L.append("")
    L.append(f"- Số lần lấy mẫu lại: **{result['settings']['n_bootstrap']}** · "
             f"đơn vị khối: **{result['settings']['block_unit']}** · seed: "
             f"**{result['settings']['seed']}** · mức tin cậy: **{result['settings']['ci_level_pct']} %**")
    L.append(f"- Metric: **{result['settings']['metric_convention']}**")
    L.append("- Quy ước hiệu: **baseline − Ridge**; dương = Ridge tốt hơn.")
    L.append("")

    L.append("## 1. FINAL TEST 2018 — hiệu MAE và hiệu RMSE")
    L.append("")
    f = result["final_test"]
    m, r = f["bootstrap"]["MAE"], f["bootstrap"]["RMSE"]
    L.append(f"- Cửa sổ: {f['range'][0]} → {f['range'][1]}, n = {_fmt(f['n_rows'])}, "
             f"{f['bootstrap']['n_blocks_days']} khối ngày lịch")
    L.append(f"- Baseline: MAE {f['metrics']['baseline']['MAE']} · "
             f"Ridge: MAE {f['metrics']['ridge']['MAE']}")
    L.append("")
    L.append("| Chỉ số | Ước lượng | CI 95 % | Độ lệch bootstrap | % lần lấy mẫu Ridge thắng | CI có chứa 0? |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for label, s in (("Hiệu MAE", m), ("Hiệu RMSE", r)):
        L.append(
            f"| {label} | {s['point_estimate']:+} | "
            f"[{s['ci_low']}; {s['ci_high']}] | {s['boot_sd']} | "
            f"{s['share_ridge_better_pct']} % | "
            f"{'**không**' if s['ci_excludes_zero'] else 'có'} |"
        )
    L.append("")

    L.append("## 2. MAE theo từng tháng của 2018")
    L.append("")
    L.append("| Tháng | n | MAE baseline | MAE Ridge | Hiệu MAE | Ridge thắng |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for row in f["months"]["by_month"]:
        L.append(
            f"| {row['month_label']} | {_fmt(row['n_samples'])} | {row['MAE_baseline']} | "
            f"{row['MAE_ridge']} | {row['delta_MAE']:+} | "
            f"{'✓' if row['ridge_better'] else '✗'} |"
        )
    L.append("")
    L.append(
        f"- **Ridge thắng ở {f['months']['n_months_ridge_better']}/"
        f"{f['months']['n_months']} tháng** của FINAL TEST 2018."
    )
    L.append("")

    L.append("## 3. Mô hình kém ở giờ nào trên 2018? (mô tả, không quyết định)")
    L.append("")
    nh = f["night_hour"]
    n, d = nh["night"], nh["daytime"]
    L.append(f"- Ridge kém hơn baseline ở **{nh['n_hours_ridge_worse']}/{nh['n_hours']} giờ**.")
    L.append(f"- Giờ ban đêm {nh['night_hours']}: MAE Ridge {n['MAE_ridge']} so với baseline "
             f"{n['MAE_baseline']} (n = {n['n_samples']}).")
    L.append(f"- **MAE tương đối** (MAE / lưu lượng thực TB) ban đêm: Ridge {n['rel_MAE_ridge']} "
             f"so với baseline {n['rel_MAE_baseline']}; ban ngày: Ridge {d['rel_MAE_ridge']} "
             f"so với baseline {d['rel_MAE_baseline']}.")
    L.append("")
    L.append("| Giờ | n | Lưu lượng thực TB | MAE Ridge | MAE baseline | MAE tương đối R / B |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for row in nh["by_hour"]:
        L.append(
            f"| {row['hour']:02d}:00 | {_fmt(row['n_samples'])} | {row['mean_actual']} | "
            f"{row['MAE_ridge']} | {row['MAE_baseline']} | "
            f"{row['rel_MAE_ridge']} / {row['rel_MAE_baseline']} |"
        )
    L.append("")
    L.append(
        "> Phần **kiểm chứng trên dữ liệu dev 2012–2017** (các fold rolling-origin) và phần "
        "thử log-target nằm ở `experiments_report.md` mục 4 và 5 — vì 2018 không được dùng để "
        "quyết định bất cứ điều gì."
    )
    L.append("")

    L.append("## 4. Có nhất quán qua các năm khác không? (chỉ dữ liệu dev 2012–2017)")
    L.append("")
    L.append("| Cửa sổ | n | MAE baseline | MAE Ridge | Hiệu MAE | CI 95 % | % Ridge thắng | Tháng Ridge thắng |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for w in result["dev_windows"]:
        mm = w["bootstrap"]["MAE"]
        L.append(
            f"| {w['name']} | {_fmt(w['n_rows'])} | {w['metrics']['baseline']['MAE']} | "
            f"{w['metrics']['ridge']['MAE']} | {mm['point_estimate']:+} | "
            f"[{mm['ci_low']}; {mm['ci_high']}] | {mm['share_ridge_better_pct']} % | "
            f"{w['months']['n_months_ridge_better']}/{w['months']['n_months']} |"
        )
    L.append("")

    L.append("## 5. Kết luận — sinh từ số liệu trên")
    L.append("")
    L.append(f"- **Trên 2018:** {v['verdict_2018']}")
    L.append(
        f"- **Số tháng Ridge thắng:** {v['months_ridge_better']} · "
        f"**số cửa sổ dev mà Ridge thắng:** {v['dev_windows_with_ridge_better']} · "
        f"**chiều nhất quán qua các cửa sổ dev:** {v['dev_direction_consistent']}"
    )
    L.append(f"- **Kết luận tổng:** {v['verdict_overall']}")
    L.append("")
    L.append("**Điều kiện đi kèm bắt buộc khi trích dẫn kết luận này:**")
    for c in v["conditions_required"]:
        L.append(f"- {c}")
    L.append("")
    L.append(f"![Bootstrap và MAE theo tháng]({result['figure']})")
    L.append("")
    L.append("*Hình — nguồn: `src/uncertainty_audit.py`.*")
    L.append("")
    return "\n".join(L)


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Bắt buộc: FINAL TEST 2018 phải đã được đánh giá trước.
    if not EVAL_JSON_PATH.exists():
        raise SystemExit(
            "Chưa có kết quả FINAL TEST. Chạy TRƯỚC:\n"
            "    py src\\evaluate.py\n"
            "Script này chỉ đo bất định quanh kết quả đó, nên phải chạy SAU."
        )

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    alpha = float(metadata["hyperparameters"]["alpha"])
    frozen_pipe = joblib.load(RIDGE_PIPELINE_PATH)
    baseline_table = pd.read_csv(BASELINE_TABLE_PATH)
    baseline_meta = json.loads(BASELINE_META_PATH.read_text(encoding="utf-8"))

    df = build_features(load_clean())
    dev = dev_window(df)

    print("=== KIỂM TOÁN BẤT ĐỊNH (chỉ đo, không quyết định) ===")
    final = final_test_window(df, frozen_pipe, baseline_table, baseline_meta["global_fallback"])
    dev_windows = [dev_pseudo_test_window(dev, alpha)] + dev_rolling_folds(dev, alpha)

    for w in [final] + dev_windows:
        mm = w["bootstrap"]["MAE"]
        print(
            f"  {w['name'][:44]:44s} n={w['n_rows']:>6,} "
            f"dMAE={mm['point_estimate']:>7} "
            f"CI95=[{mm['ci_low']:>7}; {mm['ci_high']:>7}] "
            f"Ridge thang {mm['share_ridge_better_pct']:>6} %"
        )

    verdict = build_verdict(final, dev_windows)
    verdict["verdict_overall"] = _overall_text(final, dev_windows, verdict)
    result = {
        "report_type": "uncertainty audit (CHỈ ĐO — không quyết định, không train)",
        "trains_or_tunes": False,
        "overwrites_evaluation_results": False,
        "settings": {
            "n_bootstrap": N_BOOTSTRAP,
            "block_unit": "calendar_day",
            "seed": BOOTSTRAP_SEED,
            "ci_level_pct": CI_LEVEL,
            "metric_convention": "RAW MODEL (Ridge trả về trực tiếp)",
            "difference_convention": "baseline - ridge (positive = Ridge better)",
            "why_block_by_day": (
                "Các giờ trong cùng một ngày lịch không độc lập (cùng thời tiết, cùng ngày "
                "làm việc) nên lấy mẫu lại theo dòng sẽ cho khoảng tin cậy hẹp hơn thực tế."
            ),
        },
        "final_test": final,
        "dev_windows": dev_windows,
        "verdict": verdict,
        "figure": plot_audit(final),
    }

    AUDIT_JSON_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    AUDIT_MD_PATH.write_text(render_markdown(result), encoding="utf-8")
    print("\nKết luận (sinh từ số liệu):")
    print(f"  {verdict['verdict_overall']}")
    print(f"\nĐã lưu: {AUDIT_JSON_PATH}")
    print(f"Đã lưu: {AUDIT_MD_PATH}")
    print("Lưu ý: evaluation_results.json và models/ KHÔNG bị ghi đè.")


def _overall_text(final: dict, dev_windows: list[dict], verdict: dict) -> str:
    m = final["bootstrap"]["MAE"]
    months = final["months"]
    dev_pts = [w["bootstrap"]["MAE"]["point_estimate"] for w in dev_windows]
    dev_better = sum(1 for x in dev_pts if x > 0)
    if m["ci_excludes_zero"]:
        head = (
            f"Trên FINAL TEST 2018, Ridge vượt baseline {m['point_estimate']} MAE "
            f"(CI 95 % [{m['ci_low']}; {m['ci_high']}], không chứa 0) "
            f"và thắng ở {months['n_months_ridge_better']}/{months['n_months']} tháng."
        )
    elif m["share_ridge_better_pct"] > 50.0:
        head = (
            f"Trên FINAL TEST 2018, ước lượng nghiêng về Ridge ({m['point_estimate']} MAE) "
            f"nhưng CI 95 % [{m['ci_low']}; {m['ci_high']}] **có chứa 0**, nên chưa đủ bằng "
            f"chứng để kết luận thắng có ý nghĩa; Ridge thắng ở "
            f"{months['n_months_ridge_better']}/{months['n_months']} tháng."
        )
    else:
        head = (
            f"Trên FINAL TEST 2018, baseline tốt hơn Ridge "
            f"({m['point_estimate']} MAE, CI 95 % [{m['ci_low']}; {m['ci_high']}])."
        )
    tail = (
        f" Trên dữ liệu dev 2012–2017, Ridge thắng ở {dev_better}/{len(dev_windows)} cửa sổ "
        f"(pseudo-test 2016–2017 và {len(dev_windows) - 1} fold rolling-origin), nên "
        + (
            "chiều ủng hộ nhất quán giữa các cửa sổ."
            if dev_better == len(dev_windows)
            else (
                "chiều **không nhất quán** giữa các cửa sổ — vì vậy không được nói chung "
                "'mô hình vượt baseline' một cách tuyệt đối."
            )
        )
    )
    return head + tail


if __name__ == "__main__":
    main()

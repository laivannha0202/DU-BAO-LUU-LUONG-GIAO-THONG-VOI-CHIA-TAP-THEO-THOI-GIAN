"""
src/experiments.py

BA THÍ NGHIỆM PHÁT TRIỂN — CHẠY HOÀN TOÀN TRONG CỬA SỔ 2012–2017.

⚠️ RÀNG BUỘC TUYỆT ĐỐI
---------------------
Module này KHÔNG BAO GIỜ được đọc, huấn luyện, chia tập hay tinh chỉnh trên bất kỳ dòng
nào của năm 2018. Năm 2018 là FINAL TEST và chỉ được chạm vào đúng một lần, trong
`src/evaluate.py`, SAU KHI cấu hình đã đóng băng.

Lý do: nếu ta nhìn 2018 để thiết kế thí nghiệm, rồi lại sửa mô hình theo kết quả đó,
thì 2018 không còn là holdout nữa — con số báo cáo cuối sẽ bị lạm dụng (test-set
overfitting), dù mọi pipeline vẫn "đúng kỹ thuật".

Cách thực hiện thay thế
----------------------
Tạo một **pseudo-test (historical holdout) nằm trong 2012–2017** và dùng nó để trả lời
cùng những câu hỏi mà lẽ ra phải dùng 2018:
  - Thí nghiệm 1  : random split vs time split (chỉ minh hoạ)
  - Thí nghiệm 1b: kiểm soát rò rễ thời gian trên MỘT tập test cố định
  - Thí nghiệm 3  : rolling-origin evaluation (chỉ cửa sổ out-of-sample)

Có `assert_no_final_test_rows()` ở mọi hàm để chặn nếu ai đó vô tình lọt 2018 vào.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import (  # noqa: E402
    TARGET_COLUMN,
    random_split,
)
from src.train import compute_metrics, get_X_y, make_ridge_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Mốc thời gian
# ---------------------------------------------------------------------------
SEED = 42

#: Mọi thí nghiệm phát triển dừng trước 2018.
DEV_WINDOW_END = pd.Timestamp("2017-12-31 23:59:59")

#: Năm của FINAL TEST — KHÔNG BAO GIỜ dùng ở đây.
FINAL_TEST_START = pd.Timestamp("2018-01-01 00:00:00")

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


class FinalTestLeakError(AssertionError):
    """Ném ra khi một dòng của FINAL TEST 2018 lọt vào giai đoạn phát triển."""


def assert_no_final_test_rows(df: pd.DataFrame, where: str) -> None:
    """Chặn cứng: DataFrame này không được chứa bất kỳ dòng 2018 nào."""
    if df is None or len(df) == 0:
        return
    ts = df["date_time"]
    n_bad = int((ts >= FINAL_TEST_START).sum())
    if n_bad:
        first = ts[ts >= FINAL_TEST_START].min()
        raise FinalTestLeakError(
            f"{where}: phát hiện {n_bad} dòng FINAL TEST (>= {FINAL_TEST_START}), "
            f"dòng đầu tiên {first}. Giai đoạn phát triển KHÔNG được dùng 2018."
        )


def dev_window(df: pd.DataFrame) -> pd.DataFrame:
    """Lọc dữ liệu về 2012–2017 và assert là không còn dòng 2018."""
    dev = df[df["date_time"] <= DEV_WINDOW_END].reset_index(drop=True)
    assert_no_final_test_rows(dev, "dev_window")
    return dev


# ===========================================================================
# Thí nghiệm 1 — Random split vs Time split (CHỈ MINH HỌA)
# ===========================================================================
def experiment_random_vs_time_split(dev: pd.DataFrame, alpha: float) -> dict:
    """So sánh hai cách chia trên CïNG một cửa sổ phát triển 2012–2017.

    Cả hai tập test đều thuộc 2012–2017. Tuy nhiên chúng **khác nhau về thành phần năm**,
    nên chênh lệch không thể quy hết cho rò rỉ — đó chính là lý do cần Thí nghiệm 1b.
    """
    assert_no_final_test_rows(dev, "exp1")

    # --- Arm A: time split trên cửa sổ phát triển ---
    # train 2012-2015, test 2016-2017
    train_t = dev[dev["date_time"] <= pd.Timestamp("2015-12-31 23:59:59")]
    test_t = dev[dev["date_time"] > pd.Timestamp("2015-12-31 23:59:59")]

    # --- Arm B: random split trên cùng cửa sổ ---
    train_r, _val_r, test_r = random_split(dev, train_frac=0.7, val_frac=0.15, seed=SEED)

    for name, part in [("time_train", train_t), ("time_test", test_t),
                       ("rand_train", train_r), ("rand_test", test_r)]:
        assert_no_final_test_rows(part, f"exp1:{name}")

    def fit_eval(tr, te):
        pipe = make_ridge_pipeline(alpha)
        X_tr, y_tr = get_X_y(tr)
        pipe.fit(X_tr, y_tr)
        X_te, y_te = get_X_y(te)
        return compute_metrics(y_te, pipe.predict(X_te))

    metrics_time = fit_eval(train_t, test_t)
    metrics_random = fit_eval(train_r, test_r)

    return {
        "window": {
            "start": str(dev["date_time"].min()),
            "end": str(dev["date_time"].max()),
            "n_rows": int(len(dev)),
            "contains_2018": False,
        },
        "time_split": {
            **metrics_time,
            "train_range": [str(train_t["date_time"].min()), str(train_t["date_time"].max())],
            "test_range": [str(test_t["date_time"].min()), str(test_t["date_time"].max())],
            "test_years": sorted(test_t["year"].unique().tolist()),
        },
        "random_split": {
            **metrics_random,
            "test_years": sorted(test_r["year"].unique().tolist()),
        },
        "delta_mae_random_minus_time": round(metrics_random["MAE"] - metrics_time["MAE"], 2),
        "caveat": (
            "Hai tập test KHÁC thành phần năm (time split test = 2016-2017 còn nguyên; "
            "random split test = mẫu ngẫu nhiên rải rác 2012-2017), nên chênh lệch KHÔNG "
            "chứng minh được rò rỉ. Thí nghiệm 1b mới là phép so sánh công bằng."
        ),
    }


# ===========================================================================
# Thí nghiệm 1b — Kiểm soát rò rễ thời gian trên MỘT tập test cố định
# ===========================================================================
def experiment_leakage_controlled(dev: pd.DataFrame, alpha: float) -> dict:
    """Minh hoạ rò rễ thời gian một cách công bằng, trong 2012–2017.

    Thiết kế: một tập test CỐ ĐỊNH duy nhất, dùng chung cho MỌI arm, rồi chỉ thay đổi
    duy nhất thứ "tập train có nhìn thấy dữ liệu gần test bao xa".

    Tập test chung: **nửa còn lại của năm 2017** (chia bằng seed cố định).
    Ba arm, mức gần thời gian tăng dần:
      A) train ≤ 2015            -> cách test 2 năm
      B) train ≤ 2016            -> cách test 1 năm
      C) train ≤ 2016 + nửa 2017 -> chung năm với test, chỉ khác 'hàng xóm giờ'
    """
    assert_no_final_test_rows(dev, "exp1b")

    y2017 = dev[dev["date_time"] >= pd.Timestamp("2017-01-01 00:00:00")].reset_index(drop=True)
    y2017 = y2017.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_half = int(len(y2017) * 0.5)
    neighbor_part = y2017.iloc[:n_half]  # chỉ arm C được thấy
    shared_test = y2017.iloc[n_half:]  # dùng CHUNG cho cả 3 arm

    assert_no_final_test_rows(shared_test, "exp1b:shared_test")
    assert_no_final_test_rows(neighbor_part, "exp1b:neighbor")

    arms = {
        "A_train_le_2015": dev[dev["date_time"] <= pd.Timestamp("2015-12-31 23:59:59")],
        "B_train_le_2016": dev[dev["date_time"] <= pd.Timestamp("2016-12-31 23:59:59")],
        "C_train_le_2016_plus_half_2017": pd.concat(
            [dev[dev["date_time"] <= pd.Timestamp("2016-12-31 23:59:59")], neighbor_part],
            ignore_index=True,
        ),
    }

    X_te, y_te = get_X_y(shared_test)
    results: dict[str, dict] = {}
    for name, tr in arms.items():
        assert_no_final_test_rows(tr, f"exp1b:{name}")
        pipe = make_ridge_pipeline(alpha)
        X_tr, y_tr = get_X_y(tr)
        pipe.fit(X_tr, y_tr)
        results[name] = {
            **compute_metrics(y_te, pipe.predict(X_te)),
            "n_train": int(len(tr)),
            "train_end": str(tr["date_time"].max()),
            "gap_to_test_years": 2017 - tr["date_time"].max().year,
        }

    mae_a = results["A_train_le_2015"]["MAE"]
    mae_c = results["C_train_le_2016_plus_half_2017"]["MAE"]
    return {
        "shared_test": {
            "range": [str(shared_test["date_time"].min()), str(shared_test["date_time"].max())],
            "n": int(len(shared_test)),
            "note": "Dùng CHUNG cho cả 3 arm — đây là điểm làm phép so sánh công bằng.",
        },
        "arms": results,
        "mae_improvement_closest_vs_farthest": round(mae_a - mae_c, 2),
        "interpretation": (
            "Càng đưa dữ liệu sát thời điểm dự báo vào train, MAE càng giảm — nhưng phần "
            "giảm đó KHÔNG đến từ năng lực mô hình mà từ việc mô hình đã nhìn thấy 'hàng xóm' "
            "của chính dòng cần dự báo. Đây chính là rò rễ mà time split loại bỏ, và là lý do "
            "kết luận trên 2018 phải được công bố từ một mô hình chưa từng thấy năm 2018."
        ),
    }


# ===========================================================================
# Thí nghiệm 3 — Rolling-origin evaluation (CHỈ cửa sổ out-of-sample)
# ===========================================================================
def rolling_origin_evaluation(dev: pd.DataFrame, alpha: float) -> dict:
    """Đánh giá tiến trình theo thời gian, mọi cửa sổ đều OUT-OF-SAMPLE.

    Expanding window: mỗi fold huấn luyện trên một đoạn lịch sử, rồi dự báo **năm kế
    tiếp** — đúng tình huống sử dụng thật. Nhờ vậy mọi con số trong bảng đều cùng
    loại (ngoài mẫu), nên so sánh giữa các fold là công bằng.

    KHÔNG dùng MAE trên tập train (in-sample) để kết luận drift — trộn in-sample với
    out-of-sample là so sánh không cùng đối tượng.
    """
    assert_no_final_test_rows(dev, "exp3:rolling")

    folds = [
        {"fold": 1, "train_end": "2014-12-31 23:59:59", "test_year": 2015},
        {"fold": 2, "train_end": "2015-12-31 23:59:59", "test_year": 2016},
        {"fold": 3, "train_end": "2016-12-31 23:59:59", "test_year": 2017},
    ]

    results = []
    pooled_parts = []
    for spec in folds:
        train_end = pd.Timestamp(spec["train_end"])
        tr = dev[dev["date_time"] <= train_end]
        te = dev[
            (dev["date_time"] > train_end)
            & (dev["date_time"].dt.year == spec["test_year"])
        ]
        assert_no_final_test_rows(tr, f"exp3:fold{spec['fold']}:train")
        assert_no_final_test_rows(te, f"exp3:fold{spec['fold']}:test")
        if len(tr) == 0 or len(te) == 0:
            continue

        pipe = make_ridge_pipeline(alpha)
        X_tr, y_tr = get_X_y(tr)
        pipe.fit(X_tr, y_tr)
        X_te, y_te = get_X_y(te)
        m = compute_metrics(y_te, pipe.predict(X_te))
        results.append(
            {
                "fold": spec["fold"],
                "test_year": spec["test_year"],
                "train_range": [str(tr["date_time"].min()), str(tr["date_time"].max())],
                "test_range": [str(te["date_time"].min()), str(te["date_time"].max())],
                "evaluation_type": "out_of_sample",
                **m,
            }
        )
        part = te.copy()
        X_all, _ = get_X_y(part)
        part["_pred"] = pipe.predict(X_all)
        part["_abs_err"] = (part["_pred"] - part[TARGET_COLUMN]).abs()
        pooled_parts.append(part)

    maes = [r["MAE"] for r in results]
    pooled = pd.concat(pooled_parts, ignore_index=True)

    by_month = pooled.groupby("month")["_abs_err"].agg(["mean", "size"])
    pooled_month = {int(m): {"MAE": round(float(r["mean"]), 2), "n": int(r["size"])}
                    for m, r in by_month.iterrows()}
    pooled["_season"] = pooled["month"].map(SEASON_OF_MONTH)
    by_season = pooled.groupby("_season")["_abs_err"].agg(["mean", "size"])
    pooled_season = {str(s): {"MAE": round(float(r["mean"]), 2), "n": int(r["size"])}
                     for s, r in by_season.iterrows()}

    return {
        "design": "expanding-window rolling origin; mọi fold đều out-of-sample",
        "folds": results,
        "mae_trend": {
            "years": [r["test_year"] for r in results],
            "maes": maes,
            "mae_change_first_to_last": round(maes[-1] - maes[0], 2) if len(maes) > 1 else None,
            "verdict": _trend_verdict(maes),
        },
        "pooled_out_of_sample": {
            "years": sorted(pooled["year"].unique().tolist()),
            "n": int(len(pooled)),
            "MAE": round(float(pooled["_abs_err"].mean()), 2),
            "by_month": pooled_month,
            "by_season": pooled_season,
        },
    }


def _trend_verdict(maes: list[float]) -> str:
    if len(maes) < 2:
        return "Không đủ fold để kết luận."
    delta = maes[-1] - maes[0]
    if delta > 0.10 * maes[0]:
        return (
            f"MAE tăng {delta:.1f} ({delta / maes[0]:.1%}) từ fold đầu đến fold cuối — "
            "có dấu hiệu suy giảm khi dự báo xa hơn điểm cắt huấn luyện."
        )
    if delta < -0.10 * maes[0]:
        return (
            f"MAE giảm {abs(delta):.1f} ({abs(delta) / maes[0]:.1%}) — chất lượng được cải thiện, "
            "có thể do năm gần nhất (2017) dễ hơn, không nhất thiết là mô hình tiến bộ."
        )
    return (
        f"MAE ổn định (thay đổi {delta:+.1f}, {delta / maes[0]:+.1%}) qua các cửa sổ "
        "out-of-sample — chưa thấy suy giảm rõ rệt trong 2012–2017."
    )


def covariate_drift(dev: pd.DataFrame) -> dict:
    """Đo DRIFT DỮ LIỆU (phân bố), tách biệt hoàn toàn với drift hiệu năng.

    So sánh phân bố các biến đầu vào giữa năm gốc (2013, năm TRAIN đầy đủ nhất) và các
    năm sau. Dùng Population Stability Index (PSI) — ngưỡng diễn giải quy ước:
    < 0.10 ổn định, 0.10–0.25 dịch chuyển vừa, >= 0.25 dịch chuyểch lớn.
    """
    assert_no_final_test_rows(dev, "exp3:covariate")

    base = dev[dev["year"] == 2013]
    out: dict = {
        "reference_year": 2013,
        "psi_thresholds": {"stable": 0.10, "moderate": 0.25},
        "traffic_volume_by_year": {},
        "psi_vs_reference": {},
    }
    for y, part in dev.groupby("year"):
        out["traffic_volume_by_year"][int(y)] = {
            "mean": round(float(part[TARGET_COLUMN].mean()), 2),
            "median": round(float(part[TARGET_COLUMN].median()), 2),
            "std": round(float(part[TARGET_COLUMN].std()), 2),
            "n": int(len(part)),
        }

    for col in [TARGET_COLUMN, "temp", "clouds_all"]:
        bins = _reference_bins(base[col], n_bins=10)
        out["psi_vs_reference"][col] = {
            int(y): round(_psi(base[col], part[col], bins), 4)
            for y, part in dev.groupby("year")
        }
    return out


def _reference_bins(series: pd.Series, n_bins: int = 10) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.array([0.0, 1.0])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(s.quantile(qs).to_numpy(dtype=float))
    if len(edges) < 2:
        edges = np.array([edges[0], edges[0] + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _psi(base: pd.Series, other: pd.Series, edges: np.ndarray, eps: float = 1e-6) -> float:
    b = pd.to_numeric(base, errors="coerce").dropna()
    o = pd.to_numeric(other, errors="coerce").dropna()
    if b.empty or o.empty:
        return float("nan")
    b_counts = np.histogram(b, bins=edges)[0].astype(float)
    o_counts = np.histogram(o, bins=edges)[0].astype(float)
    b_p = np.clip(b_counts / max(b_counts.sum(), 1), eps, None)
    o_p = np.clip(o_counts / max(o_counts.sum(), 1), eps, None)
    return float(np.sum((o_p - b_p) * np.log(o_p / b_p)))


# ===========================================================================
# Baseline tham chiếu (chỉ trong cửa sổ phát triển)
# ===========================================================================
def dev_baseline_comparison(dev: pd.DataFrame, alpha: float) -> dict:
    """Baseline vs Ridge trên pseudo-test 2016–2017, cùng alpha đã chọn."""
    from src.train import predict_baseline, train_baseline

    assert_no_final_test_rows(dev, "exp_baseline")
    tr = dev[dev["date_time"] <= pd.Timestamp("2015-12-31 23:59:59")]
    te = dev[dev["date_time"] > pd.Timestamp("2015-12-31 23:59:59")]

    table = train_baseline(tr)
    fallback = float(tr[TARGET_COLUMN].mean())
    X_te, y_te = get_X_y(te)
    pipe = make_ridge_pipeline(alpha).fit(*get_X_y(tr))
    return {
        "pseudo_test_range": [str(te["date_time"].min()), str(te["date_time"].max())],
        "train_range": [str(tr["date_time"].min()), str(tr["date_time"].max())],
        "baseline": compute_metrics(y_te, predict_baseline(te, table, fallback)),
        "ridge": compute_metrics(y_te, pipe.predict(X_te)),
    }


# ===========================================================================
# Main
# ===========================================================================
def run_all(df: pd.DataFrame, alpha: float) -> dict:
    dev = dev_window(df)
    results = {
        "dev_window": {
            "start": str(dev["date_time"].min()),
            "end": str(dev["date_time"].max()),
            "n_rows": int(len(dev)),
            "excluded": "2018 (FINAL TEST) — chưa được chạm vào ở giai đoạn này",
        },
        "baseline_vs_ridge_on_pseudo_test": dev_baseline_comparison(dev, alpha),
        "experiment_1_random_vs_time_split": experiment_random_vs_time_split(dev, alpha),
        "experiment_1b_leakage_controlled": experiment_leakage_controlled(dev, alpha),
        "experiment_3_rolling_origin": rolling_origin_evaluation(dev, alpha),
        "experiment_3b_covariate_drift": covariate_drift(dev),
    }
    return results


def render_report(results: dict) -> str:
    L: list[str] = []
    dw = results["dev_window"]
    L.append("# Thí nghiệm phát triển — cửa sổ 2012–2017 (FINAL TEST 2018 CHƯA BỊ CHẠM)")
    L.append("")
    L.append("> Sinh tự động bởi `python src/experiments.py`.")
    L.append("")
    L.append(
        f"- Cửa sổ phát triển: {dw['start']} → {dw['end']} — **{dw['n_rows']:,}** dòng".replace(",", ".")
    )
    L.append(f"- {dw['excluded']}")
    L.append(
        "- Mọi hàm trong module này đều gọi `assert_no_final_test_rows()`; nếu một dòng 2018 "
        "lọt vào thì chương trình dừng ngay với `FinalTestLeakError`."
    )
    L.append("")

    L.append("## 0. Baseline vs Ridge trên pseudo-test 2016–2017")
    L.append("")
    b = results["baseline_vs_ridge_on_pseudo_test"]
    L.append(f"- Train: {b['train_range'][0]} → {b['train_range'][1]}")
    L.append(f"- Pseudo-test: {b['pseudo_test_range'][0]} → {b['pseudo_test_range'][1]}")
    L.append("")
    L.append("| Mô hình | MAE | RMSE | R² | n |")
    L.append("| --- | --- | --- | --- | --- |")
    for k in ("baseline", "ridge"):
        m = b[k]
        L.append(f"| {k} | {m['MAE']} | {m['RMSE']} | {m['R2']} | {m['n']:,} |".replace(",", "."))
    L.append("")
    d_mae = b["baseline"]["MAE"] - b["ridge"]["MAE"]
    d_rmse = b["baseline"]["RMSE"] - b["ridge"]["RMSE"]
    if d_mae > 0 and d_rmse > 0:
        L.append(
            f"- Ridge tốt hơn baseline ở **cả hai** chỉ số: MAE {d_mae:+.2f}, RMSE {d_rmse:+.2f} — "
            "kết luận này đã có căn cứ **trước khi** nhìn 2018."
        )
    else:
        L.append(
            f"- ⚠️ **Hai chỉ số không đồng ý — báo cáo trung thực, không chọn lợi hơn:** so với baseline, "
            f"Ridge có MAE {d_mae:+.2f} ({'TỐT HƠN' if d_mae > 0 else 'KÉM HƠN'}) nhưng "
            f"RMSE {d_rmse:+.2f} ({'TỐT HƠN' if d_rmse > 0 else 'KÉM HƠN'}), và "
            f"R² {b['ridge']['R2']} so với {b['baseline']['R2']} "
            f"({'cao hơn' if b['ridge']['R2'] > b['baseline']['R2'] else 'thấp hơn'})."
        )
        L.append("")
        if d_mae < 0 < d_rmse:
            L.append(
                "  - Cách đọc đúng: Ridge **giảm sai số lớn** (RMSE, R² tốt hơn) nhưng lại **tăng "
                "sai số ở các giờ bình thường** (MAE kém hơn). Baseline là trung bình theo "
                "hour × day_of-week — một đường bằng phẳng rất mượt, nên hiếm khi sai lệch lớn, "
                "nhưng cũng không bám được mức lưu lượng thực tế khi năm đổi mức."
            )
            L.append(
                "  - Nguyên nhân hợp lý: tập train 2012–2015 có dữ liệu rất thưa (2014 kết thúc "
                "08/08, 2015 bắt đầu 11/06), nên Ridge thiếu dữ liệu để học mức lưu lượng ổn định, "
                "trong khi baseline không cần học mức nào."
            )
            L.append(
                "  - **Vì vậy không được dùng một chỉ số đơn lẻ để kết luận.** Báo cáo cuối sẽ "
                "trình bày MAE, RMSE và R² cùng lúc, và nêu rõ chỗ nào mô hình thắng, chỗ nào thua."
            )
    L.append("")

    exp1 = results["experiment_1_random_vs_time_split"]
    L.append("## 1. Thí nghiệm 1 — Random split vs Time split (CHỈ MINH HỌA)")
    L.append("")
    t_, r_ = exp1["time_split"], exp1["random_split"]
    L.append("| Cách chia | MAE | RMSE | R² | n test | Khoảng test |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    L.append(
        f"| Time split | {t_['MAE']} | {t_['RMSE']} | {t_['R2']} | {t_['n']:,} | "
        f"{t_['test_range'][0][:10]} → {t_['test_range'][1][:10]} |".replace(",", ".")
    )
    L.append(
        f"| Random split | {r_['MAE']} | {r_['RMSE']} | {r_['R2']} | {r_['n']:,} | "
        f"mẫu ngẫu nhiên rải rác 2012–2017 |".replace(",", ".")
    )
    L.append("")
    L.append(f"- Chênh lệch MAE (random − time) = **{exp1['delta_mae_random_minus_time']}**")
    L.append(
        "- **Không khẳng định trước** random split sẽ tốt hơn hay xấu hơn. Ở lần chạy này "
        + (
            "random split cho MAE **cao hơn** time split."
            if exp1["delta_mae_random_minus_time"] > 0
            else "random split cho MAE **thấp hơn** time split."
        )
    )
    L.append(f"- ⚠️ {exp1['caveat']}")
    L.append("")

    exp1b = results["experiment_1b_leakage_controlled"]
    L.append("## 1b. Thí nghiệm 1b — Kiểm soát rò rễ trên MỘT tập test cố định")
    L.append("")
    st = exp1b["shared_test"]
    L.append(f"- Tập test dùng CHUNG cho cả 3 arm: **{st['n']:,}** giờ, {st['range'][0][:10]} → {st['range'][1][:10]}".replace(",", "."))
    L.append(f"- {st['note']}")
    L.append("")
    L.append("| Arm | Tập train kết thúc | Cách xa test | n train | MAE | RMSE | R² | n test |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    labels = {
        "A_train_le_2015": "A: chỉ tới 2015",
        "B_train_le_2016": "B: chỉ tới 2016",
        "C_train_le_2016_plus_half_2017": "C: tới 2016 + nửa 2017",
    }
    for key, m in exp1b["arms"].items():
        L.append(
            f"| {labels.get(key, key)} | {m['train_end'][:10]} | {m['gap_to_test_years']} năm | "
            f"{m['n_train']:,} | **{m['MAE']}** | {m['RMSE']} | {m['R2']} | {m['n']:,} |".replace(",", ".")
        )
    L.append("")
    L.append(
        f"- MAE giảm **{exp1b['mae_improvement_closest_vs_farthest']}** điểm khi đưa train sát "
        "test hơn (A → C), trên **cùng một tập test**."
    )
    L.append(f"- {exp1b['interpretation']}")
    L.append("")

    exp3 = results["experiment_3_rolling_origin"]
    L.append("## 3. Thí nghiệm 3 — Rolling-origin evaluation (chỉ cửa sổ OUT-OF-SAMPLE)")
    L.append("")
    L.append(f"- Thiết kế: {exp3['design']}")
    L.append(
        "- ⚠️ **Không** dùng MAE trên tập train (in-sample) để kết luận drift: trộn in-sample "
        "với out-of-sample là so sánh không cùng đối tượng."
    )
    L.append("")
    L.append("| Fold | Năm test | Khoảng train | Khoảng test | Loại đánh giá | MAE | RMSE | R² | n |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for f in exp3["folds"]:
        L.append(
            f"| {f['fold']} | {f['test_year']} | {f['train_range'][0][:10]} → {f['train_range'][1][:10]} | "
            f"{f['test_range'][0][:10]} → {f['test_range'][1][:10]} | {f['evaluation_type']} | "
            f"**{f['MAE']}** | {f['RMSE']} | {f['R2']} | {f['n']:,} |".replace(",", ".")
        )
    L.append("")
    trend = exp3["mae_trend"]
    L.append(f"- Chuỗi MAE theo năm test: {trend['maes']} — {trend['verdict']}")
    L.append("")
    pooled = exp3["pooled_out_of_sample"]
    L.append(
        f"### Gộp các cửa sổ out-of-sample ({', '.join(str(y) for y in pooled['years'])}), n={pooled['n']:,}, MAE={pooled['MAE']}".replace(",", ".")
    )
    L.append("")
    L.append("| Mùa | MAE | n |")
    L.append("| --- | --- | --- |")
    for s, m in pooled["by_season"].items():
        L.append(f"| {s} | {m['MAE']} | {m['n']:,} |".replace(",", "."))
    L.append("")
    L.append("| Tháng | MAE | n |")
    L.append("| --- | --- | --- |")
    for m, v in pooled["by_month"].items():
        L.append(f"| {MONTH_LABELS[m - 1]} | {v['MAE']} | {v['n']:,} |".replace(",", "."))
    L.append("")

    cov = results["experiment_3b_covariate_drift"]
    L.append("## 3b. Drift DỮ LIỆU (phân bố) — khác với drift hiệu năng ở trên")
    L.append("")
    L.append(f"- Năm tham chiếu: **{cov['reference_year']}** (năm TRAIN đầy đủ nhất)")
    L.append(f"- PSI: < {cov['psi_thresholds']['stable']} ổn định, "
             f"{cov['psi_thresholds']['stable']}–{cov['psi_thresholds']['moderate']} dịch chuyển vừa, "
             f">= {cov['psi_thresholds']['moderate']} dịch chuyển lớn")
    L.append("")
    L.append("| Năm | mean traffic | median | std | n |")
    L.append("| --- | --- | --- | --- | --- |")
    for y, m in cov["traffic_volume_by_year"].items():
        L.append(f"| {y} | {m['mean']} | {m['median']} | {m['std']} | {m['n']:,} |".replace(",", "."))
    L.append("")
    years = sorted(cov["psi_vs_reference"]["traffic_volume"], key=int)
    L.append("| Biến | " + " | ".join(f"PSI {y}" for y in years) + " |")
    L.append("| --- |" + " --- |" * len(years))
    for col, per_year in cov["psi_vs_reference"].items():
        cells = " | ".join(str(per_year[y]) for y in years)
        L.append(f"| `{col}` | {cells} |")
    L.append("")
    L.append(
        "- PSI đo **mức dịch chuyển phân bố** của dữ liệu đầu vào so với năm tham chiếu; "
        "nó không nói gì về chất lượng dự báo. Drift hiệu năng nằm ở mục 3, drift phân bố nằm ở đây."
    )
    L.append(
        "- ⚠️ **Cảnh báo về cách đọc PSI:** các năm có độ phủ thời gian khác nhau sẽ cho PSI cao "
        "mà **không phải do drift thật**. 2012 chỉ có từ 10/02, 2014 kết thúc 08/08, 2015 bắt đầu "
        "11/06 — tức thiếu hẳn một số mùa so với năm tham chiếu 2013 (đủ 12 tháng). "
        "PSI cao ở `temp` (2012 và 2015) và `clouds_all` (2015) phần lớn phản ánh "
        "**khác biệt về thành phần mùa trong mẫu**, không phải hệ thống đã đổi hành vi."
    )
    L.append(
        "- Vì vậy chỉ nên so PSI giữa các năm có **độ phủ tương đương** (ví dụ 2013 vs 2017), "
        "hoặc chuẩn hoá theo tháng trước khi kết luận."
    )
    L.append("")

    L.append("## Kết luận giai đoạn phát triển")
    L.append("")
    base_line = (
        f"Ridge vượt baseline {b['baseline']['MAE'] - b['ridge']['MAE']:.2f} MAE"
        if b["ridge"]["MAE"] < b["baseline"]["MAE"]
        else (
            f"⚠️ Ridge **kém baseline về MAE** ({b['ridge']['MAE']} vs {b['baseline']['MAE']}) "
            f"nhưng **tốt hơn về RMSE** ({b['ridge']['RMSE']} vs {b['baseline']['RMSE']})"
        )
    )
    L.append(f"1. Trên pseudo-test 2016–2017: {base_line} — đã biết **trước khi** nhìn vào 2018.")
    L.append(
        "2. Thí nghiệm 1b cho thấy đưa dữ liệu sát test hơn làm MAE giảm "
        f"{exp1b['mae_improvement_closest_vs_farthest']} điểm trên cùng tập test — "
        "cơ sở để giữ 2018 hoàn toàn nguyên vẹn."
    )
    L.append(f"3. Rolling-origin (chỉ out-of-sample): {trend['verdict']}")
    L.append(
        "4. Cấu hình (alpha, feature, quy tắc tiền xử lý) đã được chốt. Bước kế tiếp là "
        "`python src/evaluate.py` — đánh giá FINAL TEST 2018 đúng một lần. "
        "**Sau khi đọc kết quả 2018, không được quay lại sửa mô hình.**"
    )
    L.append("")
    return "\n".join(L)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from src.features import build_features, load_clean

    reports_dir = _ROOT / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((_ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    alpha = metadata["hyperparameters"]["alpha"]

    df = build_features(load_clean())
    results = run_all(df, alpha)
    report = render_report(results)

    (reports_dir / "experiments_report.md").write_text(report, encoding="utf-8")
    (reports_dir / "experiments_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(report)
    print(f"\nĐã lưu: {reports_dir / 'experiments_report.md'}")
    print(f"Đã lưu: {reports_dir / 'experiments_results.json'}")


if __name__ == "__main__":
    main()

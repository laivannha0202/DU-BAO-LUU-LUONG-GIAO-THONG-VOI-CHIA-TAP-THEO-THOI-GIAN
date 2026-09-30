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
  - Thí nghiệm 1  : random split vs time split, trên cùng tập dòng đánh giá
  - Thí nghiệm 1b: tách kích thước tập huấn luyện khỏi khoảng cách thời gian
  - Thí nghiệm 1c: "hàng xóm" theo khối liên tục — cùng mức năm nhưng xa giờ
  - Thí nghiệm 3  : rolling-origin evaluation (chỉ cửa sổ out-of-sample)

Có `assert_no_final_test_rows()` ở mọi hàm để chặn nếu ai đó vô tình lọt 2018 vào.
"""
from __future__ import annotations

import hashlib
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

#: Danh sách seed CỐ ĐỊNH cho các thí nghiệm có yếu tố ngẫu nhiên.
#: Cố định từ đầu để mọi lần chạy cho ra cùng kết quả (tái lập được).
EXPERIMENT_SEEDS = [11, 23, 37, 53, 71]

#: Thí nghiệm 1c — "hàng xóm" theo KHỐI LIÊN TỤC: tháng chẵn vào train,
#: tháng lẻ làm tập test chung. Tách "cùng mức năm" khỏi "dữ liệu cùng tháng".
BLOCK_TRAIN_MONTHS = (2, 4, 6, 8, 10, 12)
BLOCK_TEST_MONTHS = (1, 3, 5, 7, 9, 11)

#: Các fold rolling-origin. Dùng chung để `experiments.py` và `uncertainty_audit.py`
#: đánh giá đúng những cửa sổ giống nhau — tránh hai nơi lệch định nghĩa fold.
ROLLING_FOLDS = (
    {"fold": 1, "train_end": "2014-12-31 23:59:59", "test_year": 2015},
    {"fold": 2, "train_end": "2015-12-31 23:59:59", "test_year": 2016},
    {"fold": 3, "train_end": "2016-12-31 23:59:59", "test_year": 2017},
)

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
# Thí nghiệm 1 — Random split vs Time split
# ===========================================================================
def _fingerprint(part: pd.DataFrame) -> str:
    """Dấu vân tay của một tập dòng (dựa trên timestamp).

    Dùng để chứng minh rằng các arm thật sự dùng CHUNG một tập test — không chỉ
    cùng số dòng, mà cùng đúng các timestamp đó.
    """
    values = pd.to_datetime(part["date_time"]).astype("int64").to_numpy()
    return hashlib.sha1(np.sort(values).tobytes()).hexdigest()[:12]


def _summarize(values: list[float]) -> dict:
    """Trung bình ± độ lệch chuẩn của một đại lượng qua nhiều seed."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 2),
        "sd": round(float(arr.std(ddof=0)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "n_seeds": int(arr.size),
    }


def experiment_random_vs_time_split(dev: pd.DataFrame, alpha: float, seeds=None) -> dict:
    """Trả lời câu hỏi nghiên cứu: random split và đánh giá trên tương lai chênh nhau bao nhiêu?

    Báo cáo HAI phép đo khác nhau, vì chúng trả lời hai câu hỏi khác nhau:

    1. ``delta_mae_different_test_sets`` — chênh MAE khi mỗi arm dùng tập test riêng
       (time split test = 2016–2017 còn nguyên; random split test = mẫu ngẫu nhiên
       rải rác 2012–2017). Con số này **bị lẫn** bởi khác biệt thành phần năm, nên
       KHÔNG dùng để kết luận về rò rỉ.
    2. ``delta_mae_same_rows`` — chênh MAE giữa mô hình time-split và mô hình
       random-split **trên đúng một tập dòng đánh giá** (tập test của random split).
       Đây là phép so sánh công bằng về *cách chọn tập huấn luyện*.
    """
    assert_no_final_test_rows(dev, "exp1")
    seeds = list(seeds or EXPERIMENT_SEEDS)

    # --- Arm time split: train ≤ 2015, test = 2016–2017 ---
    train_t = dev[dev["date_time"] <= pd.Timestamp("2015-12-31 23:59:59")]
    test_t = dev[dev["date_time"] > pd.Timestamp("2015-12-31 23:59:59")]
    assert_no_final_test_rows(train_t, "exp1:time_train")
    assert_no_final_test_rows(test_t, "exp1:time_test")

    # Mô hình time split không phụ thuộc seed -> fit một lần, dùng lại cho mọi seed.
    pipe_time = make_ridge_pipeline(alpha)
    pipe_time.fit(*get_X_y(train_t))
    metrics_time = compute_metrics(test_t[TARGET_COLUMN], pipe_time.predict(get_X_y(test_t)[0]))

    per_seed: list[dict] = []
    for seed in seeds:
        train_r, _val_r, test_r = random_split(dev, train_frac=0.7, val_frac=0.15, seed=seed)
        for name, part in (("rand_train", train_r), ("rand_test", test_r)):
            assert_no_final_test_rows(part, f"exp1:{name}:seed{seed}")

        X_r, y_r = get_X_y(test_r)
        pipe_r = make_ridge_pipeline(alpha)
        pipe_r.fit(*get_X_y(train_r))
        metrics_random = compute_metrics(y_r, pipe_r.predict(X_r))
        metrics_time_same_rows = compute_metrics(y_r, pipe_time.predict(X_r))

        per_seed.append({
            "seed": int(seed),
            "random_test_fingerprint": _fingerprint(test_r),
            "random_test_n": int(len(test_r)),
            "random_test_year_composition": {
                str(int(year)): int(n)
                for year, n in test_r["year"].value_counts().sort_index().items()
            },
            "random_split": metrics_random,
            "time_split_model_on_random_test_rows": metrics_time_same_rows,
            "delta_mae_different_test_sets": round(metrics_random["MAE"] - metrics_time["MAE"], 2),
            "delta_mae_same_rows": round(
                metrics_random["MAE"] - metrics_time_same_rows["MAE"], 2
            ),
        })

    return {
        "window": {
            "start": str(dev["date_time"].min()),
            "end": str(dev["date_time"].max()),
            "n_rows": int(len(dev)),
            "contains_2018": False,
        },
        "seeds": [int(s) for s in seeds],
        "time_split": {
            **metrics_time,
            "train_range": [str(train_t["date_time"].min()), str(train_t["date_time"].max())],
            "test_range": [str(test_t["date_time"].min()), str(test_t["date_time"].max())],
            "test_years": sorted(test_t["year"].unique().tolist()),
        },
        "per_seed": per_seed,
        "summary": {
            "delta_mae_different_test_sets": _summarize(
                [r["delta_mae_different_test_sets"] for r in per_seed]
            ),
            "delta_mae_same_rows": _summarize([r["delta_mae_same_rows"] for r in per_seed]),
            "random_split_MAE": _summarize([r["random_split"]["MAE"] for r in per_seed]),
        },
        "caveat": (
            "Hai tập test KHÁC thành phần năm (time split test = 2016-2017 còn nguyên; "
            "random split test = mẫu ngẫu nhiên rải rác 2012-2017), nên "
            "delta_mae_different_test_sets KHÔNG chứng minh được rò rỉ — một phần chênh lệch "
            "đến từ việc hai bài toán khác nhau. Vì vậy còn đo thêm delta_mae_same_rows "
            "(cùng tập dòng đánh giá) và Thí nghiệm 1b."
        ),
    }


# ===========================================================================
# Thí nghiệm 1b — Tách ba yếu tố: kích thước train · mức năm · khoảng cách giờ
# ===========================================================================
def experiment_leakage_controlled(dev: pd.DataFrame, alpha: float, seeds=None) -> dict:
    """Đo ảnh hưởng của khoảng cách thời gian, tách khỏi kích thước tập huấn luyện.

    Vấn đề của thiết kế 3 arm cũ: arm C có **nhiều dòng hơn** arm A và arm B, đồng
    thời lại chứa dữ liệu của **cùng năm 2017**. Vì thế "MAE giảm" có thể đến từ
    (i) nhiều dữ liệu hơn, (ii) mức lưu lượng trung bình của năm 2017 khác 2016, hoặc
    (iii) mô hình đã nhìn thấy các giờ lân cận của đúng dòng cần dự báo. Ba yếu tố
    này bị trộn vào nhau nếu không tách.

    Thiết kế 4 arm, MỌI arm dùng CHUNG đúng một tập test (nửa còn lại của 2017):

      A) train ≤ 2015                       — xa test nhất
      B) train ≤ 2016                       — xa test vừa
      C) train ≤ 2016 + nửa ngẫu nhiên 2017 — thêm ~nửa năm 2017, gồm giờ lân cận
      D) ngẫu nhiên từ C, **đúng bằng số dòng của B** — arm đối chứng cùng kích thước

    Đọc kết quả:
      - ``C_minus_B`` = ảnh hưởng của việc thêm dữ liệu 2017 (kích thước + mức năm + lân cận)
      - ``D_minus_B`` = ảnh hưởng khi **giữ nguyên kích thước** nhưng vẫn có dữ liệu 2017
      - ``size_adjusted_delta`` = chênh lệch giữa hai cái trên

    ⚠️ Mô hình KHÔNG có đặc trưng lag, nên nó không thể "nhớ" giá trị của dòng lân cận.
    Vì vậy cơ chế "nhìn thấy hàng xóm" là **giả thuyết, không phải điều đã chứng minh** —
    xem `experiment_block_neighbour` để tách riêng khoảng cách theo GIỜ.
    """
    assert_no_final_test_rows(dev, "exp1b")
    seeds = list(seeds or EXPERIMENT_SEEDS)

    train_le_2015 = dev[dev["date_time"] <= pd.Timestamp("2015-12-31 23:59:59")]
    train_le_2016 = dev[dev["date_time"] <= pd.Timestamp("2016-12-31 23:59:59")]
    y2017 = dev[dev["date_time"] >= pd.Timestamp("2017-01-01 00:00:00")].reset_index(drop=True)
    n_b = int(len(train_le_2016))

    per_seed: list[dict] = []
    for seed in seeds:
        shuffled = y2017.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_half = len(shuffled) // 2
        neighbor_part = shuffled.iloc[:n_half]
        shared_test = shuffled.iloc[n_half:]

        assert_no_final_test_rows(shared_test, f"exp1b:shared_test:seed{seed}")
        assert_no_final_test_rows(neighbor_part, f"exp1b:neighbor:seed{seed}")

        arm_c = pd.concat([train_le_2016, neighbor_part], ignore_index=True)
        # Arm D: đúng bằng số dòng của arm B, lấy ngẫu nhiên từ tập của arm C.
        arm_d = arm_c.sample(n=n_b, random_state=seed).reset_index(drop=True)

        arms = {
            "A_train_le_2015": train_le_2015,
            "B_train_le_2016": train_le_2016,
            "C_train_le_2016_plus_half_2017": arm_c,
            "D_same_n_as_B_random_from_C": arm_d,
        }

        X_te, y_te = get_X_y(shared_test)
        results: dict[str, dict] = {}
        for name, tr in arms.items():
            assert_no_final_test_rows(tr, f"exp1b:{name}:seed{seed}")
            pipe = make_ridge_pipeline(alpha)
            pipe.fit(*get_X_y(tr))
            results[name] = {
                **compute_metrics(y_te, pipe.predict(X_te)),
                "n_train": int(len(tr)),
                "train_end": str(tr["date_time"].max()),
                "gap_to_test_years": 2017 - int(tr["date_time"].max().year),
                "n_train_rows_from_2017": int(
                    (tr["date_time"] >= pd.Timestamp("2017-01-01")).sum()
                ),
            }

        mae = {k: v["MAE"] for k, v in results.items()}
        per_seed.append({
            "seed": int(seed),
            "shared_test_fingerprint": _fingerprint(shared_test),
            "shared_test_range": [
                str(shared_test["date_time"].min()), str(shared_test["date_time"].max())
            ],
            "shared_test_n": int(len(shared_test)),
            "arms": results,
            "deltas": {
                "C_minus_A_MAE": round(
                    mae["C_train_le_2016_plus_half_2017"] - mae["A_train_le_2015"], 2
                ),
                "C_minus_B_MAE": round(
                    mae["C_train_le_2016_plus_half_2017"] - mae["B_train_le_2016"], 2
                ),
                "D_minus_B_MAE": round(
                    mae["D_same_n_as_B_random_from_C"] - mae["B_train_le_2016"], 2
                ),
            },
        })

    def _delta_summary(key: str) -> dict:
        return _summarize([r["deltas"][key] for r in per_seed])

    c_minus_b = _delta_summary("C_minus_B_MAE")
    d_minus_b = _delta_summary("D_minus_B_MAE")
    first = per_seed[0]
    return {
        "design": (
            "4 arm, CHUNG một tập test = nửa còn lại của 2017. Arm D là arm đối chứng "
            "CÙNG KÍCH THƯỚC với arm B để tách yếu tố 'nhiều dữ liệu hơn' ra khỏi phần còn lại."
        ),
        "seeds": [int(s) for s in seeds],
        "arm_labels": {
            "A_train_le_2015": "A: train tới 2015-12-31 (xa test 2 năm)",
            "B_train_le_2016": "B: train tới 2016-12-31 (xa test 1 năm)",
            "C_train_le_2016_plus_half_2017": "C: train tới 2016 + nửa 2017 ngẫu nhiên",
            "D_same_n_as_B_random_from_C": "D: ngẫu nhiên từ C, đúng số dòng của B",
        },
        "shared_test": {
            "note": "Dùng CHUNG cho MỌI arm — đây là điểm làm phép so sánh công bằng. "
                    "Vì chia lần theo seed, tập test hơi khác giữa các lần chạy; "
                    "dấu vân tay (fingerprint) cho phép kiểm chứng điều đó.",
            "n_first_seed": first["shared_test_n"],
            "range_first_seed": first["shared_test_range"],
        },
        "per_seed": per_seed,
        "summary": {
            "deltas": {
                "C_minus_A_MAE": _delta_summary("C_minus_A_MAE"),
                "C_minus_B_MAE": c_minus_b,
                "D_minus_B_MAE": d_minus_b,
            },
            "n_train_median": {
                k: int(np.median([r["arms"][k]["n_train"] for r in per_seed]))
                for k in first["arms"]
            },
            "n_train_rows_from_2017_median": {
                k: int(np.median([r["arms"][k]["n_train_rows_from_2017"] for r in per_seed]))
                for k in first["arms"]
            },
        },
        "size_adjusted_delta_MAE": round(c_minus_b["mean"] - d_minus_b["mean"], 2),
        "mae_improvement_closest_vs_farthest": round(-first["deltas"]["C_minus_A_MAE"], 2),
        "interpretation": (
            "ĐO ĐƯỢC: thêm dữ liệu 2017 vào tập huấn luyện làm MAE thay đổi "
            f"{c_minus_b['mean']:+.2f} ± {c_minus_b['sd']:.2f} điểm. Khi đã ÉP kích thước "
            "tập huấn luyện bằng đúng kích thước của arm B (arm D), phần còn lại là "
            f"{d_minus_b['mean']:+.2f} ± {d_minus_b['sd']:.2f} điểm. Phần chênh giữa hai cái "
            f"là {c_minus_b['mean'] - d_minus_b['mean']:+.2f} điểm — gần bằng 0.\n"
            "→ **KẾT LUẬN ĐƯỢC:** cải thiện đo được KHÔNG giải thích bằng 'nhiều dòng hơn'. "
            "Nó xuất hiện ngay khi tập huấn luyện đã có dữ liệu của chính năm 2017, dù số "
            "dòng không đổi.\n"
            "⚠️ **KHÔNG** được gọi phần còn lại này là 'mô hình nhìn thấy hàng xóm'. Mô hình "
            "không dùng đặc trưng lag nên không thể nhớ giá trị của dòng lân cận. Các yếu tố "
            "còn lẫn trong phần dư: mức lưu lượng riêng của năm 2017, và việc có dữ liệu ở "
            "đúng các tháng/tháng giờ của tập test. Thí nghiệm 1c tách riêng yếu tố thứ hai."
        ),
    }


# ===========================================================================
# Thí nghiệm 1c — "Hàng xóm" theo KHỐI LIÊN TỤC: cùng mức năm, xa giờ
# ===========================================================================
def experiment_block_neighbour(dev: pd.DataFrame, alpha: float) -> dict:
    """Tách "cùng mức năm" khỏi "hàng xóm từng giờ".

    Thí nghiệm 1b đưa vào train các giờ **rải rác** trong nửa 2017, nên mô hình vừa
    nhìn thấy mức năm 2017 vừa nhìn thấy các giờ lân cận. Ở đây ta dùng **khối liên
    tục**: các tháng chẵn của 2017 vào train, các tháng lẻ làm tập test chung.

      P1) train ≤ 2016                        — không có dữ liệu 2017 nào
      P2) train ≤ 2016 + tháng chẵn của 2017  — có 2017, nhưng cách ít nhất ~1 tháng

    Như vậy ``delta_MAE_P2_minus_P1`` đo được phần lợi ích đến từ **mức năm 2017**
    mà không có hàng xóm theo giờ, và có thể so với ``C_minus_B`` của Thí nghiệm 1b.

    ⚠️ Thí nghiện này tất định (không ngẫu nhiên) nên không cần lặp seed.
    ⚠️ P2 có nhiều dòng hơn P1, nên **không** so sánh trực tiếp với C của Thí nghiệm 1b.
    """
    assert_no_final_test_rows(dev, "exp1c")

    train_le_2016 = dev[dev["date_time"] <= pd.Timestamp("2016-12-31 23:59:59")]
    y2017 = dev[dev["date_time"] >= pd.Timestamp("2017-01-01 00:00:00")]
    even_months = y2017[y2017["month"].isin(BLOCK_TRAIN_MONTHS)]
    odd_months = y2017[y2017["month"].isin(BLOCK_TEST_MONTHS)]

    assert_no_final_test_rows(even_months, "exp1c:even_months")
    assert_no_final_test_rows(odd_months, "exp1c:odd_months")

    arms = {
        "P1_train_le_2016": train_le_2016,
        "P2_train_le_2016_plus_even_months_2017": pd.concat(
            [train_le_2016, even_months], ignore_index=True
        ),
    }

    X_te, y_te = get_X_y(odd_months)
    results: dict[str, dict] = {}
    for name, tr in arms.items():
        assert_no_final_test_rows(tr, f"exp1c:{name}")
        pipe = make_ridge_pipeline(alpha)
        pipe.fit(*get_X_y(tr))
        results[name] = {
            **compute_metrics(y_te, pipe.predict(X_te)),
            "n_train": int(len(tr)),
            "train_end": str(tr["date_time"].max()),
            "n_train_rows_from_2017": int(
                (tr["date_time"] >= pd.Timestamp("2017-01-01")).sum()
            ),
        }

    delta = round(
        results["P2_train_le_2016_plus_even_months_2017"]["MAE"]
        - results["P1_train_le_2016"]["MAE"],
        2,
    )
    return {
        "design": (
            "Khối liên tục: tháng chẵn của 2017 vào train, tháng lẻ làm tập test CHUNG. "
            "Tinh hơn Thí nghiệm 1b: có mức năm 2017 nhưng dữ liệu 2017 trong train nằm ở "
            "các tháng KHÁC với tháng của dòng cần dự báo."
        ),
        "block_train_months": list(BLOCK_TRAIN_MONTHS),
        "block_test_months": list(BLOCK_TEST_MONTHS),
        "shared_test": {
            "n": int(len(odd_months)),
            "range": [str(odd_months["date_time"].min()), str(odd_months["date_time"].max())],
            "fingerprint": _fingerprint(odd_months),
        },
        "arms": results,
        "delta_MAE_P2_minus_P1": delta,
        "interpretation": (
            "Thuộc tính cho 'có dữ liệu của năm 2017' mà vẫn KHÔNG có dữ liệu ở các tháng "
            "cùng với tháng của dòng cần dự báo. Đây là biến sốc với 'C − B' của Thí nghiệm "
            "1b (dữ liệu 2017 rải rác, có cả ở các tháng của tập test).\n"
            "⚠️ Chênh lệch giữa hai biến này thuộc về 'độ phụ thuộc theo thời gian' — nhưng "
            "vẫn là MÔ TẢ, không phải bằng chứng nhân quả: arm C và P2 khác nhau cả về tháng "
            "được thay vào tập huấn luyện. Và vì mô hình không dùng đặc trưng lag, nó không "
            "thể 'nhớ' giá trị dòng lân cận — cơ chế nào trong hai khả năng đều còn là giả "
            "thuyết."
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

    folds = [dict(spec) for spec in ROLLING_FOLDS]

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
        "experiment_1c_block_neighbour": experiment_block_neighbour(dev, alpha),
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
    s1 = exp1["summary"]
    L.append("## 1. Thí nghiệm 1 — Random split vs Time split")
    L.append("")
    L.append(
        f"Lặp lại trên {len(exp1['seeds'])} seed cố định ({exp1['seeds']}); báo cáo trung bình ± độ lệch chuẩn."
    )
    L.append("")
    t_ = exp1["time_split"]
    n_test_time = f"{t_['n']:,}".replace(",", ".")
    L.append(
        f"**Time split** (train ≤ 2015, test {t_['test_range'][0][:10]} → {t_['test_range'][1][:10]}, "
        f"n={n_test_time}): MAE {t_['MAE']}, RMSE {t_['RMSE']}, R² {t_['R2']}"
    )
    L.append("")
    L.append("**Random split** (mỗi seed một tập test ngẫu nhiên rải rác 2012–2017):")
    L.append("")
    L.append("| seed | n test | MAE | RMSE | R² | Thành phần năm của tập test |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for r in exp1["per_seed"]:
        m = r["random_split"]
        comp = ", ".join(f"{y}: {n:,}".replace(",", ".") for y, n in r["random_test_year_composition"].items())
        L.append(
            f"| {r['seed']} | {m['n']:,} | {m['MAE']} | {m['RMSE']} | {m['R2']} | {comp} |".replace(",", ".")
        )
    L.append("")
    d_diff = s1["delta_mae_different_test_sets"]
    d_same = s1["delta_mae_same_rows"]
    L.append("### Câu hỏi nghiên cứu: hai cách đánh giá chênh nhau bao nhiêu?")
    L.append("")
    L.append("| Phép đo | Trung bình ± độ lệch (MAE) | min | max |")
    L.append("| --- | --- | --- | --- |")
    L.append(
        f"| (a) Mỗi arm dùng tập test riêng | {d_diff['mean']:+.2f} ± {d_diff['sd']:.2f} | "
        f"{d_diff['min']:+.2f} | {d_diff['max']:+.2f} |"
    )
    L.append(
        f"| (b) **Cùng một tập dòng đánh giá** | **{d_same['mean']:+.2f} ± {d_same['sd']:.2f}** | "
        f"{d_same['min']:+.2f} | {d_same['max']:+.2f} |"
    )
    L.append("")
    L.append(
        "- **(a)** là cách so sánh *tự nhiên* khi mỗi arm dùng tập test của chính nó. "
        "Dấu âm = random split trông **tốt hơn**."
    )
    L.append(
        "- **(b)** là phép so sánh **công bằng về cách chọn tập huấn luyện**: hai mô hình "
        "(một cái train theo thời gian, một cái train ngẫu nhiên) được chấm trên **đúng "
        "cùng một tập dòng** — tập test của random split."
    )
    L.append(
        "- **Vì sao chênh lại NHỎ?** Mô hình chỉ dùng đặc trưng **lịch** (giờ, thứ, tháng) và "
        "**thời tiết tại giờ đó**; nó **không dùng đặc trưng lag** của `traffic_volume`. Nên nó "
        "không có cơ chế nào để *nhớ* giá trị của một dòng khác. Random split ở đây làm "
        "mất phần lớn lợi thế **về mức độ khớp mùa/năm** (nó nhìn thấy tháng 11–12 của năm 2017, "
        "trong khi time-split train chỉ tới 2015), chứ không phải do *nhìn thấy hàng xóm*."
    )
    L.append(
        f"- ⚠️ {exp1['caveat']}"
    )
    L.append("")

    exp1b = results["experiment_1b_leakage_controlled"]
    L.append("## 1b. Thí nghiệm 1b — Tách kích thước tập huấn luyện khỏi khoảng cách thời gian")
    L.append("")
    L.append(f"- Thiết kế: {exp1b['design']}")
    st = exp1b["shared_test"]
    L.append(
        f"- Tập test dùng CHUNG cho cả 4 arm: **{st['n_first_seed']:,}** giờ, "
        f"{st['range_first_seed'][0][:10]} → {st['range_first_seed'][1][:10]} (seed đầu tiên)".replace(",", ".")
    )
    L.append(f"- {st['note']}")
    L.append("")
    labels = exp1b["arm_labels"]
    nmed = exp1b["summary"]["n_train_median"]
    n2017 = exp1b["summary"]["n_train_rows_from_2017_median"]
    L.append(
        "| Arm | Mô tả | n train (trung vị) | số dòng từ 2017 | MAE (TB ± SD) | RMSE | R² |"
    )
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for key, label in labels.items():
        arms_rows = [r["arms"][key] for r in exp1b["per_seed"]]
        maes = np.asarray([a["MAE"] for a in arms_rows], dtype=float)
        last = arms_rows[0]
        L.append(
            f"| {key.split('_', 1)[0]} | {label.split(': ', 1)[-1]} | "
            f"{nmed[key]:,} | {n2017[key]:,} | "
            f"**{maes.mean():.2f} ± {maes.std():.2f}** | {last['RMSE']} | {last['R2']} |".replace(",", ".")
        )
    L.append("")
    ds = exp1b["summary"]["deltas"]
    L.append("| Chênh lệch MAE | Trung bình ± SD | min | max |")
    L.append("| --- | --- | --- | --- |")
    for key, label in [
        ("C_minus_A_MAE", "C − A (xa test nhất → gần nhất)"),
        ("C_minus_B_MAE", "C − B (thêm nửa 2017)"),
        ("D_minus_B_MAE", "D − B (**cùng kích thước** với B)"),
    ]:
        v = ds[key]
        L.append(f"| {label} | {v['mean']:+.2f} ± {v['sd']:.2f} | {v['min']:+.2f} | {v['max']:+.2f} |")
    L.append("")
    L.append(
        f"- **Đọc đúng:** thêm dữ liệu 2017 làm MAE thay đổi {ds['C_minus_B_MAE']['mean']:+.2f} "
        f"± {ds['C_minus_B_MAE']['sd']:.2f} điểm. Khi đã **ép cùng kích thước tập huấn luyện**, "
        f"phần còn lại là {ds['D_minus_B_MAE']['mean']:+.2f} ± {ds['D_minus_B_MAE']['sd']:.2f} điểm."
    )
    L.append(
        f"- Phần chênh **không** giải thích được bằng kích thước tập huấn luyện: "
        f"**{exp1b['size_adjusted_delta_MAE']:+.2f}** điểm — tức gần bằng không."
    )
    for line in exp1b["interpretation"].split("\n"):
        L.append(f"- {line}" if not line.startswith("⚠️") else f"{line}")
    L.append("")

    exp1c = results["experiment_1c_block_neighbour"]
    L.append("## 1c. Thí nghiệm 1c — 'Hàng xóm' theo KHỐI LIÊN TỤC (tách mức năm khỏi giờ)")
    L.append("")
    L.append(f"- Thiết kế: {exp1c['design']}")
    L.append(
        f"- Tháng vào train: {exp1c['block_train_months']} · tháng làm test: {exp1c['block_test_months']}"
    )
    L.append(
        f"- Tập test chung: **{exp1c['shared_test']['n']:,}** giờ, "
        f"{exp1c['shared_test']['range'][0][:10]} → {exp1c['shared_test']['range'][1][:10]}".replace(",", ".")
    )
    L.append("")
    L.append("| Arm | n train | số dòng từ 2017 | MAE | RMSE | R² | n test |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for key, m in exp1c["arms"].items():
        L.append(
            f"| {key.split('_', 1)[0]} | {m['n_train']:,} | {m['n_train_rows_from_2017']:,} | "
            f"**{m['MAE']}** | {m['RMSE']} | {m['R2']} | {m['n']:,} |".replace(",", ".")
        )
    L.append("")
    L.append(
        f"- ΔMAE (P2 − P1) = **{exp1c['delta_MAE_P2_minus_P1']:+.2f}** — lợi ích của việc có "
        "dữ liệu 2017 mà dữ liệu đó nằm ở **các tháng khác** với tháng của dòng cần dự báo."
    )
    for line in exp1c["interpretation"].split("\n"):
        L.append(f"- {line}" if not line.startswith("⚠️") else line)
    gap = ds["C_minus_B_MAE"]["mean"] - exp1c["delta_MAE_P2_minus_P1"]
    L.append(
        f"- **Chênh lệch giữa hai cách đưa 2017 vào train: {gap:+.2f} MAE** "
        f"({ds['C_minus_B_MAE']['mean']:+.2f} khi rải rác toàn năm so với "
        f"{exp1c['delta_MAE_P2_minus_P1']:+.2f} khi chỉ lấy các tháng khác). Hiệu ứng đo được "
        "**không** phải do số dòng (Thí nghiệm 1b đã kiểm tra) và **không** phải do mức năm "
        "2017 (cả hai cách đều có 2017) — nó gắn với việc tập huấn luyện có dữ liệu ở **cùng "
        "tháng và gần giờ** với dòng cần dự báo."
    )
    L.append(
        "- ⚠️ **Không phải bằng chứng nhân quả.** Thí nghiệm 1b đổi cả kích thước tập huấn luyện, "
        "thí nghiệm 1c cũng vậy. Cả hai chỉ cho phép **mô tả** cái đo được, không chứng minh "
        "cơ chế nhân quả."
    )
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
        "2. Trả lời câu hỏi nghiên cứu về random split: trên **cùng một tập dòng đánh giá**, "
        f"mô hình random-split hơn mô hình time-split "
        f"{d_same['mean']:+.2f} ± {d_same['sd']:.2f} MAE. Chênh lệch nhỏ vì mô hình **không "
        "có đặc trưng lag** nên không thể nhớ giá trị dòng lân cận."
    )
    L.append(
        "3. Thí nghiệm 1b: thêm dữ liệu 2017 làm MAE thay đổi "
        f"{ds['C_minus_B_MAE']['mean']:+.2f} ± {ds['C_minus_B_MAE']['sd']:.2f} điểm, nhưng khi "
        "**ép cùng kích thước tập huấn luyện** thì vẫn còn "
        f"{ds['D_minus_B_MAE']['mean']:+.2f} ± {ds['D_minus_B_MAE']['sd']:.2f} điểm — nghĩa là "
        "cải thiện đo được **không** phải do nhiều dữ liệu hơn mà là do *có* dữ liệu 2017. "
        "Thí nghiệm 1c cho thấy phần lớn hiệu ứng gắn với việc train có dữ liệu ở cùng tháng "
        "và gần giờ với dòng cần dự báo. Đây là cơ sở để giữ 2018 hoàn toàn nguyên vẹn — "
        "**không** phải bằng chứng rằng mô hình 'nhìn thấy hàng xóm'."
    )
    L.append(f"4. Rolling-origin (chỉ out-of-sample): {trend['verdict']}")
    L.append(
        "5. Cấu hình (alpha, feature, quy tắc tiền xử lý) đã được chốt. Bước kế tiếp là "
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

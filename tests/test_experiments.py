"""
tests/test_experiments.py

BẢO VỆ FINAL TEST 2018 — các bài kiểm tra hồi quy cho ràng buộc của checkpoint 3.1.

Ba điều được bảo vệ:
  1. `src/experiments.py` KHÔNG BAO GIỜ dùng dữ liệu 2018.
  2. Guard `assert_no_final_test_rows()` thật sự chặn được khi 2018 lọt vào.
  3. Drift chỉ được kết luận trên cửa sổ OUT-OF-SAMPLE (rolling origin), không trộn
     in-sample với out-of-sample.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import experiments as exp  # noqa: E402
from src.experiments import (  # noqa: E402
    DEV_WINDOW_END,
    FINAL_TEST_START,
    FinalTestLeakError,
    assert_no_final_test_rows,
    dev_window,
)

CLEAN_PATH = ROOT / "data" / "processed" / "traffic_clean.csv"
needs_data = pytest.mark.skipif(not CLEAN_PATH.exists(), reason="Chạy `python src/data.py` trước")


# ===========================================================================
# 1. Source-level guard: experiments.py không được nhắc tới 2018 như dữ liệu
# ===========================================================================
def test_experiments_source_never_references_2018():
    """Chặn hồi quy ở mức source.

    Được phép duy nhất: định nghĩa hằng số `FINAL_TEST_START` — đó chính là hàng rào
    bảo vệ 2018. Bất kỳ phép CHIA/LỌC/CHỌN nào cắm hằng số 2018 vào dữ liệu đều là lỗi,
    vì khi đó 2018 đã tham gia vào quá trình thiết kế.
    """
    source = (ROOT / "src" / "experiments.py").read_text(encoding="utf-8")
    code_lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith("#")]
    body = "\n".join(code_lines).split('"""', 2)[-1]  # bỏ docstring đầu file

    offenders = [
        ln.strip()
        for ln in body.splitlines()
        if re.search(r'["\']\s*20(1[89]|2\d)\s*-', ln)
        and not ln.strip().startswith("FINAL_TEST_START")
    ]
    assert not offenders, f"experiments.py có tham chiếu năm 2018+ trong code: {offenders}"


def test_final_test_guard_constant_is_actually_used():
    """Hàng rào phải được DÙNG, không chỉ khai báo cho có."""
    source = (ROOT / "src" / "experiments.py").read_text(encoding="utf-8")
    # guard được gọi trong chính nó và trong mọi hàm thí nghiệm
    assert source.count("assert_no_final_test_rows(") >= 8, (
        "Phải gọi guard ở mọi hàm thí nghiệm"
    )
    assert "DEV_WINDOW_END" in source


def test_dev_window_end_is_before_2018():
    assert DEV_WINDOW_END < FINAL_TEST_START
    assert DEV_WINDOW_END == pd.Timestamp("2017-12-31 23:59:59")


def test_final_test_start_constant():
    assert FINAL_TEST_START == pd.Timestamp("2018-01-01 00:00:00")


def test_evaluate_module_is_the_only_one_touching_2018():
    """Chỉ `src/evaluate.py` được phép đánh giá 2018; `src/train.py` chỉ tune trên validation."""
    train_src = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
    # train.py không được lấy test để fit hay tune
    assert "test_df, y_test = get_X_y(test" not in train_src
    assert "test_df[TARGET_COLUMN]" not in train_src


# ===========================================================================
# 2. Guard thực thi
# ===========================================================================
def _frame_with_2018():
    ts = pd.to_datetime(["2017-12-31 23:00:00", "2018-01-01 00:00:00", "2018-06-15 12:00:00"])
    return pd.DataFrame({"date_time": ts, "value": [1, 2, 3]})


def test_assert_no_final_test_rows_accepts_clean_frame():
    df = _frame_with_2018()
    clean = df[df["date_time"] <= DEV_WINDOW_END]
    assert_no_final_test_rows(clean, "test")  # không được ném


def test_assert_no_final_test_rows_rejects_2018():
    with pytest.raises(FinalTestLeakError) as exc:
        assert_no_final_test_rows(_frame_with_2018(), "test")
    assert "2018" in str(exc.value)


def test_assert_no_final_test_rows_handles_empty():
    assert_no_final_test_rows(pd.DataFrame({"date_time": []}), "test")


def test_assert_no_final_test_rows_rejects_exactly_2018_01_01():
    """Ranh giới phải chính xác: 2018-01-01 00:00:00 là dòng 2018 đầu tiên."""
    df = pd.DataFrame({"date_time": pd.to_datetime(["2017-12-31 23:59:59"])})
    assert_no_final_test_rows(df, "boundary")
    with pytest.raises(FinalTestLeakError):
        assert_no_final_test_rows(
            pd.DataFrame({"date_time": pd.to_datetime(["2018-01-01 00:00:00"])}), "boundary"
        )


# ===========================================================================
# 3. Trên dữ liệu thật
# ===========================================================================
@needs_data
def test_dev_window_excludes_all_2018_rows():
    from src.features import build_features, load_clean

    df = build_features(load_clean())
    dev = dev_window(df)
    assert dev["date_time"].max() <= DEV_WINDOW_END
    assert (dev["date_time"] >= FINAL_TEST_START).sum() == 0
    n_2018_total = int((df["date_time"] >= FINAL_TEST_START).sum())
    assert n_2018_total > 0, "dữ liệu thật phải có 2018 để test này có ý nghĩa"
    assert len(dev) == len(df) - n_2018_total


@needs_data
def test_experiments_run_never_touches_2018():
    """Chạy thật toàn bộ thí nghiệm và kiểm tra không kết quả nào chứa dòng 2018."""
    from src.features import build_features, load_clean

    df = build_features(load_clean())
    alpha = 0.001
    dev = dev_window(df)

    e1 = exp.experiment_random_vs_time_split(dev, alpha)
    e1b = exp.experiment_leakage_controlled(dev, alpha)
    e1c = exp.experiment_block_neighbour(dev, alpha)
    e3 = exp.rolling_origin_evaluation(dev, alpha)
    e3b = exp.covariate_drift(dev)

    for name, blob in [("exp1", e1), ("exp1b", e1b), ("exp1c", e1c),
                       ("exp3", e3), ("exp3b", e3b)]:
        text = json.dumps(blob, default=str)
        assert "2018-" not in text, f"{name} có dữ liệo 2018: {text[:400]}"

    # exp3 chỉ được dự báo các năm trong cửa sổ phát triển
    assert all(f["test_year"] <= 2017 for f in e3["folds"])
    assert e3["pooled_out_of_sample"]["years"] == [2015, 2016, 2017]


@needs_data
def test_leakage_effect_is_stable_across_seeds():
    """Hiệu ứng đo được phải ổn định qua các seed, và phần 'do kích thước' phải nhỏ.

    Không assert dấu của hiệu ứng như một 'kết luận đúng' — chỉ assert nó **ổn định**
    (độ lệch nhỏ hơn trung bình), tức là kết luận không phụ thuộc seed cụ thể.
    """
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e1b = exp.experiment_leakage_controlled(dev, 0.001)
    deltas = e1b["summary"]["deltas"]

    for key, stat in deltas.items():
        assert stat["sd"] < abs(stat["mean"]), (
            f"{key} không ổn định qua các seed: {stat}"
        )

    # Arm đối chứng cùng kích thước phải cho kết quả gần arm C — tức phần giảm
    # không giải thích được bằng 'nhiều dòng hơn'.
    size_effect = e1b["size_adjusted_delta_MAE"]
    assert abs(size_effect) < 2.0, (
        f"Phần chênh do kích thước tập huấn luyện lớn bất thường: {size_effect}"
    )


@needs_data
def test_leakage_experiment_arms_share_one_test_set_and_are_seed_reproducible():
    """Mọi arm phải dùng CHUNG đúng một tập test, và kết quả phải tái lập theo seed.

    Đây là bảo vệ cho thiết kế "phép so sánh công bằng": nếu các arm lệch tập test thì
    chênh lệch MAE có thể do khác biệt bài toán chứ không phải do cách chia.
    """
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e1b = exp.experiment_leakage_controlled(dev, 0.001, seeds=[11, 23])

    assert len(e1b["per_seed"]) == 2
    for run in e1b["per_seed"]:
        for key, arm in run["arms"].items():
            assert arm["n"] == run["shared_test_n"], (
                f"{key} dùng {arm['n']} dòng test, tập chung là {run['shared_test_n']} — "
                "phép so sánh không còn công bằng"
            )

    # Chạy lại cùng seed => kết quả y hệt
    again = exp.experiment_leakage_controlled(dev, 0.001, seeds=[11, 23])
    assert again["per_seed"] == e1b["per_seed"], "Thí nghiệm 1b không tái lập được theo seed"


@needs_data
def test_leakage_arm_D_is_size_matched_to_arm_B():
    """Arm D là arm đối chứng CÙNG KÍCH THƯỚC — điều làm cho phép so sánh có ý nghĩa."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e1b = exp.experiment_leakage_controlled(dev, 0.001, seeds=[11, 23, 37])
    for run in e1b["per_seed"]:
        arms = run["arms"]
        assert arms["D_same_n_as_B_random_from_C"]["n_train"] == arms["B_train_le_2016"]["n_train"]
        # Arm C thì NHIỀU hơn B — đó chính là yếu tố gây nhiễu ở thiết kế 3 arm cũ
        assert arms["C_train_le_2016_plus_half_2017"]["n_train"] > arms["B_train_le_2016"]["n_train"]


@needs_data
def test_new_experiment_arms_contain_no_2018_rows():
    """Các arm và tập test mới không được chứa bất kỳ dòng 2018 nào."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))

    # Dựng lại đúng các tập dữ liệu mà Thí nghiệm 1c dùng, rồi kiểm tra trực tiếp
    # trên dữ liệu thật (kết quả trả về chỉ có số liệu, không có DataFrame).
    train_le_2016 = dev[dev["date_time"] <= pd.Timestamp("2016-12-31 23:59:59")]
    y2017 = dev[dev["date_time"] >= pd.Timestamp("2017-01-01 00:00:00")]
    even = y2017[y2017["month"].isin(exp.BLOCK_TRAIN_MONTHS)]
    odd = y2017[y2017["month"].isin(exp.BLOCK_TEST_MONTHS)]

    assert_no_final_test_rows(train_le_2016, "exp1c:P1")
    assert_no_final_test_rows(even, "exp1c:even_months")
    assert_no_final_test_rows(odd, "exp1c:odd_months")
    assert_no_final_test_rows(
        pd.concat([train_le_2016, even], ignore_index=True), "exp1c:P2"
    )

    e1c = exp.experiment_block_neighbour(dev, 0.001)
    # Thí nghiệm 1c chỉ dùng dữ liệu năm 2017
    assert e1c["shared_test"]["range"][0][:4] == "2017"
    assert e1c["shared_test"]["range"][1][:4] == "2017"
    assert set(e1c["block_train_months"]) | set(e1c["block_test_months"]) == set(range(1, 13))
    assert not set(e1c["block_train_months"]) & set(e1c["block_test_months"]), (
        "Tháng train và tháng test của Thí nghiệm 1c không được chồng nhau"
    )
    # n_train phải khớp với số dòng thật
    assert e1c["arms"]["P1_train_le_2016"]["n_train"] == len(train_le_2016)
    assert e1c["arms"]["P2_train_le_2016_plus_even_months_2017"]["n_train"] == (
        len(train_le_2016) + len(even)
    )


@needs_data
def test_block_neighbour_is_deterministic():
    """Thí nghiệm 1c tất định: chạy lại phải cho ra cùng kết quả."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    assert exp.experiment_block_neighbour(dev, 0.001) == exp.experiment_block_neighbour(dev, 0.001)


@needs_data
def test_experiment_1_reports_both_deltas_with_dispersion():
    """Thí nghiệm 1 phải trả lời câu hỏi nghiên cứu bằng CẢ HAI phép đo, kèm độ lệch."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e1 = exp.experiment_random_vs_time_split(dev, 0.001)

    for key in ("delta_mae_different_test_sets", "delta_mae_same_rows"):
        stat = e1["summary"][key]
        assert stat["n_seeds"] == len(e1["seeds"])
        assert {"mean", "sd", "min", "max"} <= set(stat)
        assert stat["min"] <= stat["mean"] <= stat["max"]

    # Tái lập được theo seed
    assert exp.experiment_random_vs_time_split(dev, 0.001)["per_seed"] == e1["per_seed"]


@needs_data
def test_experiments_report_does_not_claim_proven_neighbour_leakage():
    """Báo cáo thí nghiệm không được khẳng định cơ chế chưa được chứng minh."""
    path = ROOT / "reports" / "figures" / "experiments_report.md"
    if not path.exists():
        pytest.skip("Chưa chạy src/experiments.py")
    text = path.read_text(encoding="utf-8")
    for bad in [
        "không đến từ năng lực mô hình mà từ việc mô hình đã",
        'mô hình đã **nhìn thấy "hàng xóm"',
    ]:
        assert bad not in text, f"Báo cáo khẳng định nhân quả chưa được chứng minh: {bad!r}"
    # Phải nói rõ đây là mô tả, không phải bằng chứng nhân quả
    assert "Không phải bằng chứng nhân quả" in text


# ===========================================================================
# 3c. Thí nghiệm 4/5 — kiểm tra giờ đêm & log-target (CHỈ dữ liệu dev)
# ===========================================================================
@needs_data
def test_night_hour_analysis_covers_only_dev_windows():
    """Phân tích giờ đêm phải chạy trên các cửa sổ dev, không dùng 2018."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e4 = exp.experiment_night_hour_failure(dev, 0.001)

    assert e4["n_windows"] == len(exp.ROLLING_FOLDS)
    assert "2018-" not in json.dumps(e4, default=str)
    for w in e4["windows"]:
        assert "2018" not in w["name"]
        assert w["n_night_hours_observed"] == len(exp.NIGHT_HOURS)
        assert w["n_hours"] == 24
        # 24 dòng, đúng 24 giờ, số mẫu khớp tổng
        assert len(w["by_hour"]) == 24
        assert sum(r["n_samples"] for r in w["by_hour"]) == w["n_rows"]


@needs_data
def test_night_hour_relative_mae_is_well_formed():
    """MAE tương đối = MAE / lưu lượng thực TB, và phải khớp phép tính."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e4 = exp.experiment_night_hour_failure(dev, 0.001)
    for w in e4["windows"]:
        for r in w["by_hour"]:
            assert r["mean_actual"] > 0
            expected = r["MAE_ridge"] / r["mean_actual"]
            assert r["rel_MAE_ridge"] == pytest.approx(expected, rel=0.01)
            assert r["ridge_worse"] == (r["MAE_ridge"] > r["MAE_baseline"])
        n = w["night"]
        assert n["rel_MAE_ridge"] > 0 and n["rel_MAE_baseline"] > 0
        assert n["n_samples"] > 0
        d = w["daytime"]
        assert d["n_samples"] + n["n_samples"] == w["n_rows"]


@needs_data
def test_night_hour_finding_is_reproducible_in_dev():
    """Hiện tượng 'Ridge kém ở giờ đêm' phải tái lập được qua các lần chạy.

    Đây là bảo vệ cho tính trung thực của kết luận: nếu kết luận phụ thuộc ngẫu nhiên thì
    phải đổi. Ở đây mọi thứ tất định (seed cố định), nên hai lần chạy phải y hệt.
    """
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    a = exp.experiment_night_hour_failure(dev, 0.001)
    b = exp.experiment_night_hour_failure(dev, 0.001)
    assert a == b, "Phân tích giờ đêm phải tất định và tái lập được"
    assert a["pattern_reproducible_in_dev"] == (
        all(
            w["n_hours_ridge_worse_night"] == w["n_night_hours_observed"]
            for w in a["windows"]
        )
    )


@needs_data
def test_log_target_probe_is_labelled_post_hoc_and_not_applied():
    """Log-target là khám phá hậu nghiệm — phải được ghi rõ và không đụng mô hình chính."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e5 = exp.experiment_log_target_probe(dev, 0.001)

    assert "KHÁM PHÁ HẬU NGHIỆM" in e5["status"]
    assert "2018" in e5["why_cannot_be_shipped"]
    assert "2018-" not in json.dumps(e5, default=str), "Log-target probe dùng 2018"

    # Mỗi cửa sổ phải so sánh được 2 biến đích với nhau
    for w in e5["windows"]:
        transforms = {r["target_transform"] for r in w["results"]}
        assert transforms == {"linear_target", "log1p_target"}
        for r in w["results"]:
            assert r["night_MAE"] > 0
            assert r["n"] == w["n_rows"]


@needs_data
def test_experiments_module_does_not_fit_on_2018():
    """Không hàm nào của experiments.py được phép gọi .fit trên dữ liệu 2018."""
    source = (ROOT / "src" / "experiments.py").read_text(encoding="utf-8")
    # final_test_window chỉ tồn tại trong uncertainty_audit.py, không ở đây
    assert "def final_test_window" not in source
    # Không được nạp artifact của FINAL TEST trong module thí nghiệm phát triển
    assert "RIDGE_PIPELINE_PATH" not in source


# ===========================================================================
# 3d. Thí nghiệm 6 — độ nhạy alpha (CHỈ validation, KHÔNG đổi alpha đóng băng)
# ===========================================================================
@needs_data
def test_alpha_sensitivity_uses_validation_only_and_keeps_frozen_alpha():
    from src.features import build_features, load_clean, time_split

    df = build_features(load_clean())
    train_df, val_df, _test_df = time_split(df)
    e6 = exp.experiment_alpha_sensitivity(train_df, val_df, 0.001)

    assert e6["frozen_alpha"] == 0.001
    assert e6["frozen_alpha_kept"] is True
    assert "2018-" not in json.dumps(e6, default=str), "Độ nhạy alpha không được dùng 2018"
    # Lưới mở rộng phải nhỏ hơn giá trị nhỏ nhất của lưới gốc
    assert min(e6["grid"]) < 0.001
    assert 0.0 in e6["grid"], "Phải có alpha = 0 (OLS) để so sánh với hồi quy tuyến tính"
    # Có đánh dấu OLS
    ols = [r for r in e6["results"] if r["is_ols"]]
    assert len(ols) == 1
    # Số dòng kết quả bằng số phần tử lưới
    assert len(e6["results"]) == len(e6["grid"])


@needs_data
def test_alpha_sensitivity_reports_small_alpha_region_and_threshold():
    from src.features import build_features, load_clean, time_split

    df = build_features(load_clean())
    train_df, val_df, _test_df = time_split(df)
    e6 = exp.experiment_alpha_sensitivity(train_df, val_df, 0.001)

    small = e6["small_alpha_region"]
    assert small["MAE_spread"] >= 0
    assert small["MAE_max"] >= small["MAE_min"]
    # Ngưỡng "cải thiện có ý nghĩa" phải được nêu rõ
    assert e6["meaningful_delta_threshold"] > 0
    for b in e6["alphas_meaningfully_better_than_frozen"]:
        assert b["delta_vs_frozen"] <= -e6["meaningful_delta_threshold"]
    # Không được tuyên bố đã đổi alpha
    assert "test-informed" in e6["why_not_changed_now"]


def test_alpha_sensitivity_json_artifact_is_written():
    path = ROOT / "reports" / "figures" / "alpha_sensitivity.json"
    if not path.exists():
        pytest.skip("Chưa chạy src/experiments.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["frozen_alpha_kept"] is True
    assert "grid" in data and "results" in data
    assert len(data["results"]) == len(data["grid"])


# ===========================================================================
# 3e. Thí nghiệm 7 — hệ quả của việc FINAL TEST thiếu quý IV
# ===========================================================================
@needs_data
def test_test_window_bias_quantifies_missing_q4():
    """Phải định lượng được việc thiếu tháng 10–12, không chỉ nói 'không có dữ liệu'."""
    from src.features import build_features, load_clean, time_split

    df = build_features(load_clean())
    train_df, val_df, _test_df = time_split(df)
    e7 = exp.experiment_test_window_bias(train_df, val_df, 0.001)

    assert "2018-" not in json.dumps(e7, default=str)
    full = e7["validation_full_year"]
    jan_sep = e7["validation_jan_sep"]
    q4 = e7["validation_oct_dec"]
    # Tổng các phần phải khớp cả năm
    assert full["n"] == jan_sep["n"] + q4["n"]
    # Jan–Sep và Oct–Dec là hai khoảng rời nhau
    assert jan_sep["n"] > 0 and q4["n"] > 0
    # Chênh lệch phải khớp phép trừ
    assert e7["delta_mae_jan_sep_minus_full_year"] == pytest.approx(
        round(jan_sep["MAE"] - full["MAE"], 2), abs=0.01
    )
    assert e7["test_window_last_month"] == 9
    # Có báo cáo 3 tháng tệ nhất, và phải chứa tháng 11 và 12 (quý IV bị thiếu)
    worst_months = [m for m, _ in e7["worst_months"]]
    assert 11 in worst_months and 12 in worst_months
    assert set(e7["by_month"]) == set(range(1, 13))
    assert e7["direction"] and e7["interpretation"]


@needs_data
def test_test_window_bias_uses_only_validation_rows():
    """Cửa sổ đo phải nằm trong 2017, không tràn sang 2018."""
    from src.features import build_features, load_clean, time_split

    df = build_features(load_clean())
    train_df, val_df, _test_df = time_split(df)
    assert val_df["date_time"].max() < exp.FINAL_TEST_START
    e7 = exp.experiment_test_window_bias(train_df, val_df, 0.001)
    assert e7["validation_full_year"]["n"] == len(val_df)


# ===========================================================================
# 3f. Thí nghiệm 8 — lag (độc lập, KHÔNG vào serving)
# ===========================================================================
def test_lag_is_built_by_timestamp_not_by_row_shift():
    """Lag phải ghép theo THỜI ĐIỂM. Đây là bảo vệ quan trọng nhất của Thí nghiệm 8."""
    # Dữ liệu thiếu giờ -> shift theo dòng và ghép theo thời điểm phải cho KẾT QUẢ KHÁC NHAU
    ts = pd.to_datetime([
        "2017-01-01 00:00", "2017-01-01 01:00", "2017-01-01 05:00", "2017-01-01 06:00",
    ])
    df = pd.DataFrame({"date_time": ts, "traffic_volume": [100.0, 200.0, 300.0, 400.0]})

    by_time = exp.add_lag_features(df, lags=(1,))
    naive = exp.add_lag_features_row_shift(df, lags=(1,))

    # Dòng cuối: "1 giờ trước" = 05:00 (giá trị 300). Dòng trước đó (05:00) cách 4 giờ,
    # nên bản đúng theo thời điểm phải cho NaN.
    assert by_time.loc[3, "traffic_lag_1h"] == 300.0
    assert np.isnan(by_time.loc[2, "traffic_lag_1h"]), (
        "Không có quan sát lúc 04:00 -> phải NaN, KHÔNG được lấy giá trị của dòng trước"
    )
    # shift theo dòng gán giá trị sai (và gán cả cho dòng không có dữ liệu trước đó)
    assert naive.loc[3, "traffic_lag_1h"] == 300.0
    assert naive.loc[2, "traffic_lag_1h"] == 200.0, (
        "shift theo dòng lấy giá trị dòng trước dù cách nhau 4 giờ — đây chính là lỗi"
    )


def test_lag_features_never_interpolate_missing_hours():
    df = pd.DataFrame({
        "date_time": pd.to_datetime(["2017-01-01 00:00", "2017-01-01 03:00"]),
        "traffic_volume": [10.0, 30.0],
    })
    out = exp.add_lag_features(df, lags=(1, 24))
    assert out["traffic_lag_1h"].isna().all(), "Không được nội suy giờ thiếu"
    assert out["traffic_lag_24h"].isna().all()


@needs_data
def test_lag_experiment_is_independent_and_never_used_for_serving():
    """Thí nghiệm lag KHÔNG được chạm vào bộ feature của mô hình đã đóng băng."""
    from src.features import FEATURE_COLUMNS_ALL

    # Không cột lag nào được thêm vào bộ feature chính
    assert not any("lag" in c.lower() for c in FEATURE_COLUMNS_ALL)
    # Không chạm vào models/
    source = (ROOT / "src" / "experiments.py").read_text(encoding="utf-8")
    assert "RIDGE_PIPELINE_PATH" not in source
    assert "joblib" not in source


@needs_data
def test_lag_experiment_only_uses_dev_rows_and_reports_all_three_arms():
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e8 = exp.experiment_lag_features(dev, 0.001, lags=(1, 24, 168))

    assert "2018-" not in json.dumps(e8, default=str), "Thí nghiệm lag dùng 2018"
    assert e8["lags_hours"] == [1, 24, 168]
    for run in e8["per_seed"]:
        assert {"with_lag", "without_lag", "naive_row_shift_lag"} <= set(run)
    s = e8["summary"]
    assert {"with_lag", "without_lag", "naive_row_shift_lag"} <= set(s)
    for key in ("with_lag", "without_lag", "naive_row_shift_lag"):
        assert s[key]["n_seeds"] == len(e8["seeds"])
        assert s[key]["min"] <= s[key]["mean"] <= s[key]["max"]
    # Số dòng bị lọc phải được báo minh bạch
    rows = e8["rows"]
    assert rows["n_dropped"] == rows["n_before_drop"] - rows["n_after_drop"]
    assert rows["share_dropped_pct"] > 0
    assert "KHÔNG đưa vào serving" in e8["not_for_serving"]


@needs_data
def test_lag_report_does_not_overclaim_leakage():
    """Báo cáo phải trung thực: kết quả ĐO ĐƯỢC không ủng hộ giả thuyết thì phải nói thẳng."""
    path = ROOT / "reports" / "figures" / "experiments_report.md"
    if not path.exists():
        pytest.skip("Chưa chạy src/experiments.py")
    text = path.read_text(encoding="utf-8")
    assert "không ủng hộ" in text, (
        "Phải nói rõ kết quả không ủng hộ giả thuyết 'lag tự tạo rò rỉ'"
    )
    assert "Lag ĐÚNG" in text and "Lag SAI" in text
    assert "KHÔNG" in text and "serving" in text



@needs_data
def test_rolling_origin_folds_are_all_out_of_sample():
    """Mọi fold phải out-of-sample và không overlap với train."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e3 = exp.rolling_origin_evaluation(dev, 0.001)
    assert len(e3["folds"]) == 3
    for fold in e3["folds"]:
        assert fold["evaluation_type"] == "out_of_sample"
        assert pd.Timestamp(fold["train_range"][1]) < pd.Timestamp(fold["test_range"][0]), (
            f"Fold {fold['fold']}: train và test chồng nhau"
        )
        assert fold["test_year"] <= 2017


@needs_data
def test_rolling_origin_never_reports_in_sample_metric():
    """Không được gọi nhãn 'drift' cho số liệu in-sample."""
    e3_report = (ROOT / "reports" / "figures" / "experiments_report.md")
    if not e3_report.exists():
        pytest.skip("Chưa chạy src/experiments.py")
    text = e3_report.read_text(encoding="utf-8")
    low = text.lower()
    assert "in-sample" in low, "Báo cáo phải nêu rõ không dùng in-sample cho drift"
    assert "rolling-origin" in low
    # bảng drift phải không chứa dòng train in-sample
    assert "train_2012_2016_in_sample" not in low
    # KHÔNG được xuất hiện bất kỳ KHOẢNG DỮ LIỆU nào của 2018 (việc nhắc tên 2018 trong
    # câu "FINAL TEST 2018 chưa bị chạm" là hợp lệ và cần thiết)
    assert "2018-01" not in text and "2018-09" not in text, (
        "Báo cáo thí nghiệm phát triển không được chứa khoảng dữ liệu của 2018"
    )


@needs_data
def test_covariate_drift_is_separate_from_performance_drift():
    dev = dev_window(__import__("src.features", fromlist=["x"]).build_features(
        __import__("src.features", fromlist=["x"]).load_clean()
    ))
    cov = exp.covariate_drift(dev)
    assert cov["reference_year"] == 2013
    assert set(cov["psi_vs_reference"].keys()) >= {"traffic_volume", "temp", "clouds_all"}
    # drift phân bố không được chứa metric dự báo
    assert "MAE" not in json.dumps(cov)

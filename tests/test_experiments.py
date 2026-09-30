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
    e3 = exp.rolling_origin_evaluation(dev, alpha)
    e3b = exp.covariate_drift(dev)

    for name, blob in [("exp1", e1), ("exp1b", e1b), ("exp3", e3), ("exp3b", e3b)]:
        text = json.dumps(blob, default=str)
        assert "2018-" not in text, f"{name} có dữ liệo 2018: {text[:400]}"

    # exp3 chỉ được dự báo các năm trong cửa sổ phát triển
    assert all(f["test_year"] <= 2017 for f in e3["folds"])
    assert e3["pooled_out_of_sample"]["years"] == [2015, 2016, 2017]


@needs_data
def test_leakage_experiment_shows_monotonic_improvement():
    """Leakage demo phải thật sự cho thấy càng gần test càng tốt — nếu không, thí nghiệm vô nghĩa."""
    from src.features import build_features, load_clean

    dev = dev_window(build_features(load_clean()))
    e1b = exp.experiment_leakage_controlled(dev, 0.001)
    arms = e1b["arms"]
    assert "A_train_le_2015" in arms and "C_train_le_2016_plus_half_2017" in arms
    # cùng một tập test cho mọi arm -> phép so sánh mới công bằng
    assert len({a["n"] for a in arms.values()}) == 1
    assert e1b["mae_improvement_closest_vs_farthest"] > 0, (
        "Arm gần test hơn phải có MAE thấp hơn; nếu không thì thiết kế thí nghiệm hỏng"
    )


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

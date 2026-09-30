"""
tests/test_uncertainty_audit.py

BẢO VỆ `src/uncertainty_audit.py` — script CHỈ ĐO, chạy SAU `src/evaluate.py`.

Bốn điều được bảo vệ:
  1. Script không bao giờ quyết định gì: không train lại mô hình 2018, không ghi vào
     `models/`, không ghi đè `evaluation_results.json`.
  2. Bootstrap là **cặp** và lấy mẫu lại theo **khối ngày lịch**, với seed cố định
     nên tái lập được.
  3. Các cửa sổ dev (pseudo-test 2016–2017 và các fold rolling-origin) **không** được
     chứa dòng 2018, và dùng đúng định nghĩa fold như `src/experiments.py`.
  4. Kết luận phải **khớp với số liệu** — không được viết trước rồi bịa số sau.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import uncertainty_audit as ua  # noqa: E402
from src.experiments import ROLLING_FOLDS, dev_window  # noqa: E402
from src.features import build_features, load_clean  # noqa: E402

CLEAN_PATH = ROOT / "data" / "processed" / "traffic_clean.csv"
AUDIT_JSON = ROOT / "reports" / "figures" / "uncertainty_audit.json"
AUDIT_MD = ROOT / "reports" / "figures" / "uncertainty_audit.md"
needs_data = pytest.mark.skipif(not CLEAN_PATH.exists(), reason="Chạy `python src/data.py` trước")
needs_audit = pytest.mark.skipif(
    not AUDIT_JSON.exists(), reason="Chạy `python src/uncertainty_audit.py` trước"
)


def _load_audit() -> dict:
    if not AUDIT_JSON.exists():
        pytest.skip("Chưa chạy src/uncertainty_audit.py")
    return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))


# ===========================================================================
# 1. Script chỉ đo, không quyết định
# ===========================================================================
def test_audit_declares_it_does_not_train_or_decide():
    source = (ROOT / "src" / "uncertainty_audit.py").read_text(encoding="utf-8")
    # Không được ghi vào thư mục models/ (artifact đã đóng băng)
    assert "MODELS_DIR" not in source, (
        "uncertainty_audit.py không được chạm vào thư mục models/"
    )
    # Không được ghi đè artifact đánh giá
    assert "EVAL_JSON_PATH.write_text" not in source
    assert "EVAL_JSON_PATH.write" not in source
    # Từ chối chạy nếu chưa có kết quả FINAL TEST -> buộc chạy SAU evaluate.py
    assert "if not EVAL_JSON_PATH.exists()" in source, (
        "Phải từ chối chạy khi chưa có kết quả FINAL TEST"
    )


@needs_audit
def test_audit_artifact_declares_measurement_only():
    a = _load_audit()
    assert a["trains_or_tunes"] is False
    assert a["overwrites_evaluation_results"] is False
    assert a["report_type"].startswith("uncertainty audit")
    assert a["settings"]["n_bootstrap"] >= 2000, (
        f"Yêu cầu >= 2000 lần lấy mẫu lại, đang là {a['settings']['n_bootstrap']}"
    )


@needs_audit
def test_audit_uses_frozen_artifact_for_2018():
    """Trên 2018 phải dùng pipeline ĐÃ ĐÓNG BĂNG, không fit lại."""
    source = (ROOT / "src" / "uncertainty_audit.py").read_text(encoding="utf-8")
    assert "joblib.load(RIDGE_PIPELINE_PATH)" in source, (
        "Phần 2018 phải nạp artifact đã đóng băng, không được train lại"
    )
    # Không được gọi .fit trên dữ liệu 2018: các lệnh fit chỉ nằm trong hàm dev
    dev_fns = re.findall(r"def (dev_\w+)\(.*?\n(?=\ndef |\Z)", source, flags=re.S)
    for fn in dev_fns:
        block = re.search(rf"def {fn}\(.*?\n(?=\ndef |\Z)", source, flags=re.S).group(0)
        assert ".fit(" in block, f"{fn} phải fit (chỉ trên dev)"
    final_block = re.search(
        r"def final_test_window\(.*?\n(?=\ndef |\Z)", source, flags=re.S
    ).group(0)
    assert ".fit(" not in final_block, (
        "final_test_window() KHÔNG được fit — 2018 chỉ được dự báo, không huấn luyện"
    )


# ===========================================================================
# 2. Bootstrap cặp theo khối ngày lịch, seed cố định
# ===========================================================================
def test_block_bootstrap_is_deterministic_and_uses_calendar_days():
    rng = np.random.default_rng(0)
    n_hours = 24 * 6
    ts = pd.date_range("2017-03-01", periods=n_hours, freq="h")
    part = pd.DataFrame({"date_time": ts})
    err_b = rng.normal(0, 60, n_hours)
    err_r = rng.normal(0, 55, n_hours)

    a = ua.paired_block_bootstrap(part, err_b, err_r, n_boot=500, seed=7)
    b = ua.paired_block_bootstrap(part, err_b, err_r, n_boot=500, seed=7)
    c = ua.paired_block_bootstrap(part, err_b, err_r, n_boot=500, seed=8)

    def comparable(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    assert comparable(a) == comparable(b), "Cùng seed phải cho kết quả y hệt (tái lập được)"
    assert np.array_equal(a["_draws_mae"], b["_draws_mae"]), (
        "Cùng seed phải cho đúng tập số lấy mẫu giống nhau"
    )
    assert a["block_unit"] == "calendar_day"
    assert a["n_blocks_days"] == 6, "6 ngày lịch -> 6 khối"
    assert a["n_rows"] == n_hours
    assert a["_draws_mae"].shape == (500,)
    assert a["_draws_mae"].std() > 0, "Phải có phân tán"
    assert not np.array_equal(c["_draws_mae"], a["_draws_mae"]), (
        "Seed khác phải cho kết quả khác (nếu không, seed không có tác dụng)"
    )


def test_block_bootstrap_point_estimate_matches_direct_computation():
    rng = np.random.default_rng(1)
    ts = pd.date_range("2017-05-01", periods=24 * 5, freq="h")
    part = pd.DataFrame({"date_time": ts})
    err_b = rng.normal(0, 50, len(ts))
    err_r = rng.normal(0, 45, len(ts))
    out = ua.paired_block_bootstrap(part, err_b, err_r, n_boot=200, seed=3)
    expected = float(np.abs(err_b).mean() - np.abs(err_r).mean())
    assert out["MAE"]["point_estimate"] == pytest.approx(round(expected, 2), abs=0.01)


def test_block_bootstrap_ci_contains_point_estimate_usually():
    """Khoảng phần trăm 95 % phải bao quanh ước lượng gần như luôn."""
    rng = np.random.default_rng(2)
    ts = pd.date_range("2017-06-01", periods=24 * 40, freq="h")
    part = pd.DataFrame({"date_time": ts})
    err_b = rng.normal(0, 50, len(ts))
    err_r = err_b + rng.normal(2.0, 5.0, len(ts))
    out = ua.paired_block_bootstrap(part, err_b, err_r, n_boot=1000, seed=11)
    s = out["MAE"]
    assert s["ci_low"] <= s["point_estimate"] <= s["ci_high"]
    assert 0.0 <= s["share_ridge_better_pct"] <= 100.0
    assert out["RMSE"]["ci_low"] <= out["RMSE"]["point_estimate"] <= out["RMSE"]["ci_high"]


# ===========================================================================
# 3. Cửa sổ dev không được chứa 2018
# ===========================================================================
@needs_data
def test_dev_windows_contain_no_2018_rows():
    dev = dev_window(build_features(load_clean()))
    folds = ua.dev_rolling_folds(dev, 0.001)
    assert len(folds) == len(ROLLING_FOLDS)
    for w in [ua.dev_pseudo_test_window(dev, 0.001)] + folds:
        assert w["range"][0][:4] != "2018" and w["range"][1][:4] != "2018", (
            f"Cửa sổ dev chứa 2018: {w['range']}"
        )
        assert w["range"][1][:4] <= "2017"
        for row in w["months"]["by_month"]:
            assert 1 <= row["month"] <= 12
    assert all(f["test_year"] <= 2017 for f in folds)


@needs_data
def test_rolling_folds_match_experiments_definition():
    """Hai script phải đánh giá đúng những cửa sổ giống nhau (không lệch định nghĩa)."""
    from src.experiments import rolling_origin_evaluation

    dev = dev_window(build_features(load_clean()))
    exp = rolling_origin_evaluation(dev, 0.001)
    aud = ua.dev_rolling_folds(dev, 0.001)
    assert [f["test_year"] for f in aud] == [f["test_year"] for f in exp["folds"]]
    assert [f["n_rows"] for f in aud] == [f["n"] for f in exp["folds"]]
    for a, e in zip(aud, exp["folds"]):
        assert a["range"] == [e["test_range"][0], e["test_range"][1]]


# ===========================================================================
# 4. Kết luận phải khớp số liệu
# ===========================================================================
@needs_audit
def test_verdict_matches_the_numbers_it_is_derived_from():
    a = _load_audit()
    v, f = a["verdict"], a["final_test"]
    m = f["bootstrap"]["MAE"]

    # Số trong kết luận phải bằng số trong bảng
    assert v["final_test_MAE"]["point_estimate"] == m["point_estimate"]
    assert v["final_test_MAE"]["ci95"] == [m["ci_low"], m["ci_high"]]
    assert v["final_test_MAE"]["ci_excludes_zero"] == m["ci_excludes_zero"]
    assert v["months_ridge_better"] == (
        f"{f['months']['n_months_ridge_better']}/{f['months']['n_months']}"
    )

    # Nếu CI chứa 0 thì kết luận KHÔNG được nói "chắc chắn hơn"
    if not m["ci_excludes_zero"]:
        assert "chứa 0" in v["verdict_2018"]
    else:
        assert "không chứa 0" in v["verdict_2018"]

    # Tỉ lệ cửa sổ dev ủng hộ Ridge phải khớp dữ liệu
    pts = [w["bootstrap"]["MAE"]["point_estimate"] for w in a["dev_windows"]]
    expected = f"{sum(1 for x in pts if x > 0)}/{len(pts)}"
    assert v["dev_windows_with_ridge_better"] == expected
    assert v["dev_direction_consistent"] == all(x > 0 for x in pts)


@needs_audit
def test_monthly_counts_are_internally_consistent():
    a = _load_audit()
    for w in [a["final_test"]] + a["dev_windows"]:
        months = w["months"]
        n_win = sum(1 for r in months["by_month"] if r["ridge_better"])
        assert n_win == months["n_months_ridge_better"]
        assert (
            months["n_months_ridge_better"] + months["n_months_baseline_better"]
            == months["n_months"]
        )
        for r in months["by_month"]:
            # delta được tính từ MAE chưa làm tròn rồi mới round, nên khi tính lại
            # từ hai số đã round có thể lệch tối đa 0,01.
            expected = r["MAE_baseline"] - r["MAE_ridge"]
            assert r["delta_MAE"] == pytest.approx(expected, abs=0.02)
            assert r["ridge_better"] == (r["MAE_ridge"] < r["MAE_baseline"])


@needs_audit
def test_audit_markdown_reports_the_ci_and_month_count():
    text = AUDIT_MD.read_text(encoding="utf-8")
    a = _load_audit()
    m = a["final_test"]["bootstrap"]["MAE"]
    assert str(m["point_estimate"]) in text
    assert str(m["ci_low"]) in text and str(m["ci_high"]) in text
    assert a["verdict"]["months_ridge_better"] in text
    # Phải nói rõ điều kiện đi kèm
    assert "Điều kiện đi kèm" in text
    assert "RAW MODEL" in text


# ===========================================================================
# 5. Phân tích giờ đêm trên 2018 (MÔ TẢ — nằm đúng ở script chạy sau evaluate.py)
# ===========================================================================
@needs_audit
def test_night_hour_block_exists_and_is_consistent():
    a = _load_audit()
    nh = a["final_test"]["night_hour"]
    assert nh["n_hours"] == 24
    assert len(nh["by_hour"]) == 24
    assert nh["n_hours_ridge_worse"] + nh["n_hours_ridge_better"] == 24
    assert (
        nh["n_hours_ridge_worse"] == sum(1 for r in nh["by_hour"] if r["ridge_worse"])
    )
    for r in nh["by_hour"]:
        assert r["mean_actual"] > 0
        assert r["rel_MAE_ridge"] == pytest.approx(
            r["MAE_ridge"] / r["mean_actual"], rel=0.01
        )
        assert r["ridge_worse"] == (r["MAE_ridge"] > r["MAE_baseline"])
    assert nh["night"]["n_samples"] + nh["daytime"]["n_samples"] == a["final_test"]["n_rows"]
    assert nh["night_hours"] == [0, 1, 2, 3, 4]


@needs_audit
def test_audit_night_hour_section_appears_in_markdown():
    text = AUDIT_MD.read_text(encoding="utf-8")
    assert "giờ nào trên 2018" in text
    a = _load_audit()
    nh = a["final_test"]["night_hour"]
    assert f"{nh['n_hours_ridge_worse']}/{nh['n_hours']} giờ" in text
    # Phải trỏ về nơi kiểm chứng trên dev
    assert "experiments_report.md" in text


# ===========================================================================
# 6. Ngày lễ trong FINAL TEST: số ngày, MAE từng ngày, cảnh báo mẫu nhỏ
# ===========================================================================
@needs_audit
def test_holiday_detail_lists_every_holiday_date_with_warning():
    a = _load_audit()
    h = a["final_test"]["holidays"]

    assert h["n_holiday_dates"] == len(h["by_holiday_date"]), (
        "Số ngày lễ phải khớp số dòng bảng chi tiết"
    )
    assert h["n_holiday_dates"] > 0
    assert h["n_holiday_hours"] + h["n_non_holiday_hours"] == a["final_test"]["n_rows"]
    assert sum(r["n_hours"] for r in h["by_holiday_date"]) == h["n_holiday_hours"]
    for r in h["by_holiday_date"]:
        assert r["MAE_ridge"] > 0 and r["MAE_baseline"] > 0
        assert r["n_hours"] > 0
    # Ngày lẻ 7 ngày rất khác nhau -> phải có cảnh báo mẫu nhỏ
    assert "độ bất định lớn" in h["warning"]
    assert str(h["n_holiday_dates"]) in h["warning"]


@needs_audit
def test_holiday_detail_records_zero_sample_weather_without_faking_zero():
    """Phân khúc không có mẫu phải được nêu tên, không được báo số 0."""
    a = _load_audit()
    h = a["final_test"]["holidays"]
    assert "Squall" in h["weather_categories_with_zero_samples"]
    assert "không có mẫu" in h["zero_sample_note"].lower()
    text = AUDIT_MD.read_text(encoding="utf-8")
    assert "Squall" in text
    assert "không có mẫu" in text.lower()


@needs_audit
def test_audit_markdown_shows_holiday_table():
    text = AUDIT_MD.read_text(encoding="utf-8")
    a = _load_audit()
    h = a["final_test"]["holidays"]
    assert f"{h['n_holiday_dates']} ngày lịch" in text
    for row in h["by_holiday_date"]:
        assert row["date"] in text, f"Thiếu ngày lễ {row['date']} trong báo cáo"

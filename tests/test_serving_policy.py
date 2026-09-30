"""
tests/test_serving_policy.py

KIỂM TOÁN CHÍNH SÁCH PHỤC VỤ (postprocess `max(0, ·)`).

Câu hỏi cần trả lời bằng bằng chứng, không bằng khẳng định:
  1. Policy `max(0, ·)` có phải **test-informed postprocessing** không?
     -> Phải KHÔNG. Nó phải được chốt trên TRAIN + VALIDATION, và FINAL TEST
        không được tham gia quyết định.
  2. Có chứng minh được policy bằng (a) miền giá trị và (b) validation không?
  3. Nếu không chứng minh được thì có bỏ clamp không?
     -> Script đóng băng sẽ tự quyết `policy = "none"` nếu điều kiện không đạt,
        và test ở đây kiểm tra đúng hành vi đó.

Test cũng khóa lập luận toán học: cắt về sàn KHÔNG BAO GIỜ làm tăng sai số tuyệt đối
ở dòng nào mà sự thật nằm trên sàn — đây là phép chiếu, không phải siêu tham số.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.registry import get_artifacts  # noqa: E402
from app.serving import predict  # noqa: E402

from conftest import requires_artifacts  # noqa: E402

POLICY_PATH = config.models_dir() / "serving_policy.json"
AUDIT_PATH = config.reports_dir() / "postprocess_audit.json"
FREEZE_SCRIPT = ROOT / "src" / "freeze_serving_policy.py"
AUDIT_SCRIPT = ROOT / "src" / "postprocess_audit.py"

REASON_NO_POLICY = "Chưa chốt serving policy. Chạy: py src\\freeze_serving_policy.py"
REASON_NO_AUDIT = "Chưa có báo cáo tác động. Chạy: py src\\postprocess_audit.py"

requires_policy = pytest.mark.skipif(not POLICY_PATH.exists(), reason=REASON_NO_POLICY)
requires_audit = pytest.mark.skipif(not AUDIT_PATH.exists(), reason=REASON_NO_AUDIT)


@pytest.fixture(scope="module")
def policy() -> dict:
    if not POLICY_PATH.exists():
        pytest.skip(REASON_NO_POLICY)
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def audit() -> dict:
    if not AUDIT_PATH.exists():
        pytest.skip(REASON_NO_AUDIT)
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Policy KHÔNG được test-informed
# ---------------------------------------------------------------------------
@requires_policy
def test_policy_was_not_chosen_using_final_test(policy):
    assert policy["final_test_used_for_selection"] is False, (
        "Policy mang tính test-informed postprocessing — không được chấp nhận."
    )


@requires_policy
def test_policy_selected_only_on_allowed_splits(policy):
    selected = policy["selected_on"]
    assert "VALIDATION 2017" in selected
    assert "TRAIN 2012-2016" in selected
    for entry in selected:
        assert "2018" not in str(entry), f"2018 không được tham gia chọn policy: {entry}"


@requires_policy
def test_freeze_script_source_never_reads_final_test():
    """Bằng chứng ở mức MÃ NGUỒN: quyết định chỉ dựa trên train + validation.

    Kiểm tra hai tầng:
      (1) artifact: bằng chứng chỉ chứa `train` và `validation`, không có `test`.
      (2) mã nguồn: `build_policy()` chỉ gọi `_evidence()` cho train và validation.
    """
    policy_json = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert set(policy_json["evidence"]) == {"train", "validation"}, (
        "bằng chứng trong artifact chứa tập ngoài train/validation"
    )
    assert "test" not in json.dumps(policy_json["evidence"]).lower()

    source = FREEZE_SCRIPT.read_text(encoding="utf-8")
    body = source[source.index("def build_policy"):source.index("def render_markdown")]
    # _evidence chỉ được gọi cho train và val
    calls = re.findall(r'_evidence\(\s*["\']([^"\']+)["\']', body)
    assert calls, "build_policy() không gọi _evidence() cho tập nào"
    assert all("TEST" not in c.upper() and "2018" not in c for c in calls), (
        f"build_policy() đọc tập không được phép: {calls}"
    )
    # và phải nêu rõ loại trừ 2018 trong docstring
    assert "FINAL TEST 2018 KHÔNG ĐƯỢC ĐỌC" in source


@requires_policy
def test_audit_script_refuses_to_run_without_frozen_policy():
    """Báo cáo tác động phải TỪ CHỐI chạy nếu chưa có policy đóng băng."""
    source = AUDIT_SCRIPT.read_text(encoding="utf-8")
    assert "FROZEN_POLICY_PATH.exists()" in source
    assert "raise SystemExit" in source
    # và phải từ chối nếu artifact tự khai đã dùng 2018
    assert 'final_test_used_for_selection" ) is not False' in source.replace(" ", " ") or \
           "final_test_used_for_selection\") is not False" in source


# ---------------------------------------------------------------------------
# 2. Căn cứ (a) — miền giá trị
# ---------------------------------------------------------------------------
@requires_policy
def test_domain_floor_is_zero_from_train(policy):
    """Sàn miền giá trị lấy từ TRAIN, không phải từ test."""
    assert policy["domain_floor_from_train"] == 0.0
    assert policy["decision_trace"]["D1_domain_floor_non_negative"] is True


@requires_policy
def test_domain_argument_is_stated(policy):
    rule = policy["decision_rule"]
    assert "D1_domain_floor" in rule
    assert "min(traffic_volume)" in rule["D1_domain_floor"]
    assert "tren TRAIN" in rule["D1_domain_floor"]
    # D2/D3 cũng phải nêu rõ tập đo
    assert "TRAIN" in rule["D2_model_extrapolates_below_floor"] or \
           "TRAIN" in rule["D2_model_extrapolates_below_floor"]
    assert "VALIDATION 2017" in rule["D3_validation_not_worse"]
    assert "khong cat" in rule["otherwise"]


# ---------------------------------------------------------------------------
# 3. Căn cứ (b) — bằng chứng VALIDATION
# ---------------------------------------------------------------------------
@requires_policy
def test_validation_evidence_present(policy):
    val = policy["evidence"]["validation"]
    assert val["split"].startswith("VALIDATION")
    assert val["n_raw_negative"] > 0, "D2 cần bằng chứng mô hình thực sự tràn xuống 0"
    assert val["min_raw_prediction"] < 0
    assert val["min_actual_target"] >= 0, "sự thật phải nằm trên sàn thì phép chiếu mới hợp lệ"


@requires_policy
def test_validation_clipping_does_not_worsen_metrics(policy):
    val = policy["evidence"]["validation"]
    assert val["metrics_clipped"]["MAE"] <= val["metrics_raw"]["MAE"], (
        "D3 thất bại: cắt về sàn làm MAE tăng trên VALIDATION"
    )
    assert val["metrics_clipped"]["RMSE"] <= val["metrics_raw"]["RMSE"]
    assert val["metrics_clipped"]["R2"] >= val["metrics_raw"]["R2"]


@requires_policy
def test_pointwise_property_holds_on_validation(policy):
    """Lập luận toán học: cắt về sàn không tăng sai số tuyệt đối ở BẤT KỲ dòng nào."""
    val = policy["evidence"]["validation"]
    assert val["pointwise_abs_error_never_increases"] is True
    assert val["n_rows_where_clip_changes_error"] == val["n_raw_negative"], (
        "chỉ đúng những dòng có dự báo âm mới bị đổi giá trị"
    )


@requires_policy
def test_mathematical_argument_is_recorded(policy):
    arg = policy["mathematical_argument"]
    assert "max(0" in arg
    assert "y_hat" in arg
    assert "khong bao gio lam tang sai so tuyet doi" in arg.replace("không", "khong")


@requires_policy
def test_decision_trace_is_consistent(policy):
    d = policy["decision_trace"]
    assert d["policy_adopted"] == (
        d["D1_domain_floor_non_negative"]
        and d["D2_model_extrapolates_below_floor"]
        and d["D3_validation_clipped_not_worse"]
    )
    if d["policy_adopted"]:
        assert policy["policy_id"] == "non_negative_projection"
    else:
        # nếu không đủ điều kiện thì KHÔNG được cắt, phải ghi thành limitation
        assert policy["policy_id"] == "none"
        assert "KHONG cat" in policy["rule"] or "không" in policy["rule"].lower()


@requires_policy
def test_policy_artifact_has_integrity_hash(policy):
    assert len(policy.get("content_sha256", "")) == 64


# ---------------------------------------------------------------------------
# 4. Đường dẫn từ chối: nếu D3 fail thì phải bỏ clamp
# ---------------------------------------------------------------------------
def test_freeze_script_has_fallback_to_no_policy():
    """Script phải có nhánh BỎ clamp — không được ép cắt bất kể thế nào."""
    source = FREEZE_SCRIPT.read_text(encoding="utf-8")
    assert 'policy_id = "non_negative_projection" if adopted else "none"' in source
    assert "negative prediction" in source
    assert "limitation" in source
    assert "KHONG cat" in source
    rule = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["decision_rule"]["otherwise"] \
        if POLICY_PATH.exists() else ""
    assert "khong cat" in rule


# ---------------------------------------------------------------------------
# 5. App thực sự dùng policy đã đóng băng
# ---------------------------------------------------------------------------
@requires_artifacts
def test_app_reads_the_frozen_policy(artifacts):
    info = artifacts.serving_policy_info()
    assert info["policy_artifact_present"] is True
    assert info["final_test_used_for_selection"] is False
    assert artifacts.apply_non_negative_projection() is True


@requires_artifacts
def test_api_exposes_policy_provenance(client):
    body = client.get("/api/model-info").json()["serving_policy"]
    assert body["final_test_used_for_selection"] is False
    assert "2017" in " ".join(body["selected_on"])
    assert "2018" not in " ".join(body["selected_on"])
    assert body["justification_domain"]
    assert "2018" in body["explicitly_not_justified_by"]


@requires_artifacts
def test_health_reports_policy(client):
    body = client.get("/health").json()
    assert "serving_policy" in body
    assert body["serving_policy"]["final_test_used_for_selection"] is False


@requires_artifacts
def test_forecast_response_separates_raw_from_deployed(client, valid_payload):
    body = client.post("/api/traffic-forecast", json=valid_payload).json()
    pred = body["predictor"]
    assert "Ridge" in pred["raw_model"]
    assert "max(0" in pred["deployed_predictor"]
    assert pred["raw_model_output"] == body["raw_model_output"]
    assert pred["deployed_prediction"] == body["predicted_traffic_volume"]
    assert body["clipped_to_zero"] is (body["raw_model_output"] < 0)
    assert body["serving_policy"]["final_test_used_for_selection"] is False


@requires_artifacts
def test_clipped_case_returns_zero_but_reports_raw(client):
    """Đêm 25/12 lúc 02:00: RAW âm, DEPLOYED = 0, cờ bật, và sự thật là ngày lẅ."""
    body = client.post("/api/traffic-forecast", json={
        "date_time": "2018-12-25T02:00:00",
        "temperature_celsius": 5.0,
        "clouds_all": 10,
        "weather": ["Clear"],
    }).json()
    assert body["raw_model_output"] < 0
    assert body["predicted_traffic_volume"] == 0.0
    assert body["clipped_to_zero"] is True
    assert body["holiday"]["is_holiday"] == 1


@requires_artifacts
def test_deployed_prediction_never_below_floor(artifacts):
    for hour in range(0, 24, 2):
        out = predict(artifacts, {
            "date_time": __import__("datetime").datetime(2018, 12, 25, hour, 0, 0),
            "temperature_celsius": -5.0,
            "clouds_all": 80,
            "weather": ["Snow"],
        })
        assert out["predicted_traffic_volume"] >= 0.0
        assert out["predictor"]["deployed_prediction"] >= 0.0


# ---------------------------------------------------------------------------
# 6. Báo cáo tác động (chạy SAU khi đóng băng)
# ---------------------------------------------------------------------------
@requires_audit
def test_audit_reports_frozen_policy_provenance(audit):
    fp = audit["frozen_policy"]
    assert fp["final_test_used_for_selection"] is False
    assert "VALIDATION 2017" in fp["selected_on"]


@requires_audit
def test_audit_uses_distinct_metric_names(audit):
    conv = audit["metric_convention"]
    assert "RAW MODEL" in conv["raw_model"] or "Ridge" in conv["raw_model"]
    assert "wrapper phục vụ" in conv["deployed_predictor"]
    assert "KHÔNG phải mô hình khác" in conv["deployed_predictor"]
    for split in audit["splits"]:
        assert "metrics_raw" in split
        assert "metrics_deployed" in split


@requires_audit
@requires_artifacts
def test_audit_raw_metrics_equal_official_final_test(audit, artifacts):
    """Metric RAW trong báo cáo tác động phải BẰNG metric chính thức của mô hình."""
    official = artifacts.evaluation["final_test"]["ridge"]
    split = next(s for s in audit["splits"] if s["split"].startswith("FINAL TEST"))
    assert split["n_rows"] == official["n"]
    for key in ("MAE", "RMSE", "R2"):
        assert split["metrics_raw"][key] == official[key], (
            f"{key} RAW trong báo cáo tác động khác FINAL TEST chính thức"
        )


@requires_audit
def test_audit_shows_negative_predictions_exist(artifacts):
    """Chứng minh policy không phải hình thức: mô hình THỰC SỰ trả dự báo âm."""
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    for split in audit["splits"]:
        assert split["n_raw_negative"] > 0
        assert split["min_raw_prediction"] < 0
        assert split["any_nan_or_inf"] is False


@requires_audit
def test_audit_pointwise_property_holds_on_final_test(audit):
    split = next(s for s in audit["splits"] if s["split"].startswith("FINAL TEST"))
    assert split["pointwise_abs_error_never_increases"] is True
    assert split["n_rows_where_clip_changes_error"] == split["n_raw_negative"]


# ---------------------------------------------------------------------------
# 7. Tài liệu không được nói sai
# ---------------------------------------------------------------------------
@requires_policy
def test_report_never_justifies_policy_by_final_test():
    report = (ROOT / "reports" / "final_report.md").read_text(encoding="utf-8")
    # không được viết kiểu "chọn clamp vì thấy N lỗi âm trên FINAL TEST"
    forbidden = [
        "chọn max(0) vì thấy 34",
        "chọn clamp vì thấy 34",
        "vì thấy 34 dòng",
        "thấy 34 dự báo âm trên FINAL TEST nên chọn",
    ]
    for phrase in forbidden:
        assert phrase not in report, f"báo cáo biện minh policy bằng FINAL TEST: {phrase!r}"
    # phải nêu rõ policy chốt trên train+validation
    assert "VALIDATION" in report
    assert "2017" in report


def test_report_uses_raw_and_deployed_labels():
    report = (ROOT / "reports" / "final_report.md").read_text(encoding="utf-8")
    assert "RAW MODEL" in report
    assert "DEPLOYED PREDICTOR" in report
    assert "không phải cùng một" in report or "KHÔNG phải cùng một" in report


@requires_policy
def test_serving_policy_report_exists_and_is_consistent():
    md = config.reports_dir() / "serving_policy.md"
    if not md.exists():
        pytest.skip("Chưa sinh reports/figures/serving_policy.md")
    text = md.read_text(encoding="utf-8")
    assert "KHÔNG đọc FINAL TEST 2018" in text
    assert "RAW MODEL" in text
    assert "DEPLOYED PREDICTOR" in text

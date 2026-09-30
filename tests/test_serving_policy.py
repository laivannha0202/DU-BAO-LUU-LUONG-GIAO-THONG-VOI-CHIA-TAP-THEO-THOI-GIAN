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


# ---------------------------------------------------------------------------
# 8. ĐÓNG BĂNG LẶP LẠI ĐƯỢC (idempotent)
#
# Lỗi gốc: script ghi `frozen_at_utc = datetime.now(...)` ở mỗi lần chạy và tái dùng
# nó trong `serving_policy.md`, nên `models/serving_policy.json` đổi hash dù quyết
# định không đổi. Ba hậu quả: (1) không tái lập được trên máy sạch; (2) một script
# tên "freeze" lại ghi đè chính policy đã đóng băng mà không cảnh báo; (3) không
# chứng minh được rằng chạy lại chỉ khác timestamp.
#
# Test ở đây chạy `main()` trong tiến trình với `POLICY_PATH` trỏ vào thư mục tạm —
# KHÔNG bao giờ chạm vào `models/` thật. `build_policy()` thật chỉ được gọi MỘT lần
# (fixture `real_content`) vì nó phải nạp pipeline + dữ liệu (~6 giây).
# ---------------------------------------------------------------------------
CLEAN_DATA = ROOT / "data" / "processed" / "traffic_clean.csv"
RIDGE = ROOT / "models" / "ridge_pipeline.joblib"

REASON_NO_INPUTS = (
    "Chưa có dữ liệu/artifact để tính lại policy. Chạy:\n"
    "    py src\\download_data.py\n"
    "    py src\\data.py\n"
    "    py src\\train.py"
)
requires_freeze_inputs = pytest.mark.skipif(
    not (CLEAN_DATA.exists() and RIDGE.exists()), reason=REASON_NO_INPUTS
)


@pytest.fixture(scope="session")
def real_content() -> dict:
    """Nội dung policy tính bằng `build_policy()` THẬT — gọi một lần cho cả module."""
    from src.freeze_serving_policy import build_policy

    return build_policy()


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Bản sao policy + báo cáo trong thư mục tạm, và script trỏ vào đó.

    Nhờ vậy các test dưới đây thử ghi đè / tạo mới mà `models/` thật không bị đụng.
    """
    from src import freeze_serving_policy as fsp

    models = tmp_path / "models"
    figures = tmp_path / "figures"
    models.mkdir()
    figures.mkdir()

    real_md = config.reports_dir() / "serving_policy.md"
    policy_file = models / "serving_policy.json"
    md_file = figures / "serving_policy.md"
    if POLICY_PATH.exists():
        policy_file.write_bytes(POLICY_PATH.read_bytes())
    if real_md.exists():
        md_file.write_bytes(real_md.read_bytes())

    monkeypatch.setattr(fsp, "POLICY_PATH", policy_file)
    monkeypatch.setattr(fsp, "POLICY_MD_PATH", md_file)
    return {"json": policy_file, "md": md_file, "module": fsp}


def _stamp_of(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["frozen_at_utc"]


# --- 8.1. Nguyên nhân gốc: nội dung phải tách khỏi đồng hồ ---------------------
def test_build_policy_content_carries_no_wallclock_field(real_content):
    """`build_policy()` không được nhúng thời điểm chạy vào nội dung quyết định.

    Đây là điều kiện tiên quyết để chạy lại cho ra cùng một file: nếu `frozen_at_utc`
    nằm trong kết quả của `build_policy()` thì mọi lần ghi đều khác nhau.
    """
    assert "frozen_at_utc" not in real_content
    assert "content_sha256" not in real_content


def _code_of(source: str, header: str, next_header: str) -> str:
    """Cắt phần MÃ (bỏ docstring) giữa hai mốc nguồn.

    Docstring của `render_markdown()` giải thích rằng nó KHÔNG gọi đồng hồ và có
    chữ `datetime.now()` trong câu đó — nên phải bỏ docstring thì mới kiểm tra
    được phần thực thi, thay vì kiểm tra nhầm vào lời giải thích.
    """
    body = source[source.index(header):source.index(next_header)]
    first_quote = body.index('"""')
    after_open = body[first_quote + 3:]
    if after_open.lstrip().startswith('"""'):  # docstring một dòng
        return after_open[after_open.index('"""') + 3:]
    return after_open[after_open.index('"""', 1) + 3:]


def test_freeze_script_does_not_stamp_wallclock_into_content():
    """Bằng chứng ở mức mã nguồn: chỗ duy nhất đọc đồng hồ là lần đóng băng MỚI."""
    source = FREEZE_SCRIPT.read_text(encoding="utf-8")
    # `build_policy()` và `render_markdown()` phải là hàm thuần theo nội dung
    assert "datetime.now(" not in _code_of(source, "def build_policy", "def render_markdown"), (
        "build_policy() không được đọc đồng hồ — nội dung phải là hằng số"
    )
    assert "datetime.now(" not in _code_of(
        source, "def render_markdown", "# Đóng băng lặp lại được"
    ), "render_markdown() không được gọi đồng hồ"
    # chỗ duy nhất được phép đọc đồng hồ là trong `main()`, tại lần tạo policy MỚI
    main_code = source[source.index("def main("):source.index('if __name__ == "__main__"')]
    assert main_code.count("datetime.now(") == 2, (
        "trong main() chỉ được đọc đồng hồ ở lần tạo policy MỚI (có fallback khi "
        f"thiếu frozen_at_utc); đang có {main_code.count('datetime.now(')} chỗ"
    )
    assert 'datetime.now(timezone.utc).isoformat(timespec="seconds")' in source


def test_markdown_timestamp_is_read_from_json_not_the_clock():
    """`serving_policy.md` lấy mốc đóng băng từ JSON, không từ `datetime.now()`."""
    source = FREEZE_SCRIPT.read_text(encoding="utf-8")
    body = _code_of(source, "def render_markdown", "# Đóng băng lặp lại được")
    assert "datetime.now(" not in body, "render_markdown() không được gọi đồng hồ"
    assert "p['frozen_at_utc']" in body or 'p["frozen_at_utc"]' in body


# --- 8.2. Chạy hai lần -> bytes giống hệt ------------------------------------
@requires_policy
@requires_freeze_inputs
def test_running_freeze_twice_gives_byte_identical_policy(sandbox, real_content):
    """Chạy lại lần nữa phải cho ra ĐÚNG file đã có — kể cả byte-for-byte."""
    fsp = sandbox["module"]
    policy_file = sandbox["json"]

    first_exit = fsp.main([])
    first_bytes = policy_file.read_bytes()
    first_stamp = _stamp_of(policy_file)

    second_exit = fsp.main([])
    second_bytes = policy_file.read_bytes()

    assert first_exit == 0
    assert second_exit == 0
    assert second_bytes == first_bytes, (
        "chạy lại lần 2 phải cho ra đúng bytes của lần 1 "
        f"({len(first_bytes)} vs {len(second_bytes)} bytes)"
    )
    assert _stamp_of(policy_file) == first_stamp, "frozen_at_utc bị đặt lại"


@requires_policy
@requires_freeze_inputs
def test_rerun_on_matching_policy_does_not_touch_the_file(sandbox, capsys):
    """Nội dung khớp -> KHÔNG ghi lại file, in đúng thông điệp yêu cầu."""
    fsp = sandbox["module"]
    policy_file = sandbox["json"]
    before_bytes = policy_file.read_bytes()
    before_mtime = policy_file.stat().st_mtime_ns

    assert fsp.main([]) == 0

    out = capsys.readouterr().out
    assert "policy đã đóng băng, nội dung khớp" in out
    assert policy_file.read_bytes() == before_bytes
    assert policy_file.stat().st_mtime_ns == before_mtime, (
        "file đã bị ghi lại dù nội dung không đổi"
    )


@requires_policy
@requires_freeze_inputs
def test_rerun_keeps_the_original_freeze_timestamp(sandbox):
    """Mốc đóng băng thật phải được giữ nguyên, không đặt lại thành 'bây giờ'."""
    fsp = sandbox["module"]
    policy_file = sandbox["json"]
    original = _stamp_of(policy_file)

    fsp.main([])
    fsp.main([])

    assert _stamp_of(policy_file) == original


# --- 8.3. Nội dung lệch -> dừng, không ghi đè --------------------------------
def _tampered_content(real_content: dict) -> dict:
    """Nội dung KHÁC quyết định thật — dùng để ép nhánh 'lệch'."""
    bad = json.loads(json.dumps(real_content))
    bad["evidence"]["validation"]["metrics_raw"]["MAE"] += 5.0
    return bad


@requires_policy
@requires_freeze_inputs
def test_changed_content_exits_nonzero_and_never_overwrites(sandbox, real_content, monkeypatch):
    fsp = sandbox["module"]
    monkeypatch.setattr(fsp, "build_policy", lambda: _tampered_content(real_content))
    policy_file = sandbox["json"]
    before = policy_file.read_bytes()

    assert fsp.main([]) != 0, "nội dung lệch mà vẫn exit 0"
    assert policy_file.read_bytes() == before, "file đã đóng băng bị ghi đè khi không có --force"


@requires_policy
@requires_freeze_inputs
def test_changed_content_prints_a_readable_diff(sandbox, real_content, monkeypatch, capsys):
    fsp = sandbox["module"]
    monkeypatch.setattr(fsp, "build_policy", lambda: _tampered_content(real_content))

    assert fsp.main([]) != 0

    out = capsys.readouterr().out
    assert "KHÔNG KHỚP" in out
    assert "evidence.validation.metrics_raw.MAE" in out, "diff phải chỉ rõ đường dẫn trường"
    assert "file hiện có" in out and "tính lại được" in out, "diff phải hiện cả hai giá trị"
    assert "--force" in out, "phải chỉ ra lối thoát: cần --force"


@requires_policy
@requires_freeze_inputs
def test_check_exits_1_when_content_differs(sandbox, real_content, monkeypatch):
    fsp = sandbox["module"]
    monkeypatch.setattr(fsp, "build_policy", lambda: _tampered_content(real_content))

    assert fsp.main(["--check"]) == 1


# --- 8.4. --force: ghi đè được, nhưng phải cảnh báo --------------------------
@requires_policy
@requires_freeze_inputs
def test_force_overwrites_and_warns_that_it_is_a_refreeze(sandbox, real_content, monkeypatch, capsys):
    fsp = sandbox["module"]
    monkeypatch.setattr(fsp, "build_policy", lambda: _tampered_content(real_content))
    policy_file = sandbox["json"]
    original_stamp = _stamp_of(policy_file)

    assert fsp.main(["--force"]) == 0

    out = capsys.readouterr().out
    assert "CẢNH BÁO" in out, "--force phải in cảnh báo"
    assert "docs/project-log.md" in out, (
        "cảnh báo phải yêu cầu ghi lý do vào docs/project-log.md"
    )

    frozen = json.loads(policy_file.read_text(encoding="utf-8"))
    assert frozen["evidence"]["validation"]["metrics_raw"]["MAE"] == pytest.approx(
        _tampered_content(real_content)["evidence"]["validation"]["metrics_raw"]["MAE"]
    ), "--force phải ghi nội dung mới"
    assert frozen["frozen_at_utc"] == original_stamp, (
        "đóng băng lại không được tự ý đổi mốc đóng băng đã có"
    )


# --- 8.5. --check: không ghi bất kỳ file nào --------------------------------
@requires_policy
@requires_freeze_inputs
def test_check_writes_no_file_at_all(sandbox):
    """`--check` là chế độ CHỈ ĐỌC: cả hash lẫn mtime của mọi file đều không đổi."""
    fsp = sandbox["module"]
    watched = [p for p in (sandbox["json"], sandbox["md"]) if p.exists()]
    assert watched, "sandbox phải có file để theo dõi"
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in watched}

    assert fsp.main(["--check"]) == 0

    for path, (data, mtime) in before.items():
        assert path.exists(), f"--check đã xoá {path.name}"
        assert path.read_bytes() == data, f"--check đã ghi lại {path.name}"
        assert path.stat().st_mtime_ns == mtime, f"--check đã chạm vào mtime của {path.name}"


@requires_policy
@requires_freeze_inputs
def test_check_does_not_create_a_policy_that_does_not_exist_yet(sandbox):
    """Chưa đóng băng + `--check` -> exit 1, KHÔNG tạo file."""
    fsp = sandbox["module"]
    policy_file = sandbox["json"]
    policy_file.unlink()

    assert fsp.main(["--check"]) == 1
    assert not policy_file.exists(), "--check không được tạo policy"


@requires_policy
@requires_freeze_inputs
def test_check_reports_the_existing_frozen_timestamp(sandbox, capsys):
    fsp = sandbox["module"]
    original = _stamp_of(sandbox["json"])

    assert fsp.main(["--check"]) == 0
    assert original in capsys.readouterr().out


# --- 8.6. Chữ ký nội dung bắt được việc sửa tay ------------------------------
@requires_policy
@requires_freeze_inputs
def test_hand_edited_policy_is_detected_and_not_overwritten(sandbox, capsys):
    """Sửa tay `serving_policy.json` phải bị phát hiện qua chữ ký nội dung."""
    fsp = sandbox["module"]
    policy_file = sandbox["json"]
    tampered = json.loads(policy_file.read_text(encoding="utf-8"))
    tampered["policy_id"] = "none"
    policy_file.write_text(json.dumps(tampered, indent=2, ensure_ascii=False), encoding="utf-8")
    before = policy_file.read_bytes()

    assert fsp.main([]) != 0

    out = capsys.readouterr().out
    assert "SỬA TAY" in out, "phải báo file đã bị sửa tay sau khi đóng băng"
    assert policy_file.read_bytes() == before


def test_content_hash_detects_a_single_field_edit(real_content):
    """Chữ ký nội dung phải đổi khi chỉ một trường duy nhất bị sửa."""
    from src.freeze_serving_policy import check_integrity, stamp

    good = stamp(real_content, "2020-01-01T00:00:00+00:00")
    assert check_integrity(good)[0] is True

    edited = dict(good)
    edited["policy_id"] = "none"
    assert check_integrity(edited)[0] is False


# --- 8.7. So sánh số thực theo dung sai -------------------------------------
def test_float_comparison_uses_tolerance():
    """Float có thể khác nhẹ giữa nền tảng — chênh nhỏ KHÔNG được coi là lệch."""
    from src.freeze_serving_policy import compare_content

    assert compare_content({"v": 272.12}, {"v": 272.12 + 1e-9}) == []
    assert compare_content({"v": 272.12}, {"v": 272.1200004}) == []
    # nhưng lệch thật thì phải bị bắt
    diffs = compare_content({"v": 272.12}, {"v": 272.90})
    assert len(diffs) == 1
    assert diffs[0]["path"] == "v"


def test_booleans_are_not_treated_as_numbers():
    """`True` KHÔNG được coi là bằng `1` — nếu không, quyết định D1-D3 có thể lọt."""
    from src.freeze_serving_policy import compare_content

    assert compare_content({"policy_adopted": True}, {"policy_adopted": 1}) != []


# --- 8.8. serving_policy.md chỉ chứa mốc đóng băng từ JSON ------------------
@requires_policy
def test_serving_policy_md_shows_only_the_timestamp_stored_in_json():
    """Tài liệu được sinh tự động không được chứa thời điểm hiện tại.

    Mọi mốc thời gian dạng ISO-8601 trong `serving_policy.md` phải đúng bằng
    `frozen_at_utc` trong JSON — nếu có mốc nào khác thì tài liệu đã bị ghi bằng
    đồng hồ lúc chạy, tức mất tính tái lập.
    """
    md = config.reports_dir() / "serving_policy.md"
    if not md.exists():
        pytest.skip("Chưa sinh reports/figures/serving_policy.md")
    frozen_at = _stamp_of(POLICY_PATH)
    stamps = set(re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?", md.read_text(encoding="utf-8")))
    assert stamps == {frozen_at}, (
        f"serving_policy.md chứa mốc thời gian {stamps} khác frozen_at_utc {frozen_at!r}"
    )


@requires_policy
@requires_freeze_inputs
def test_generated_md_follows_the_json_timestamp(sandbox, real_content):
    """Sinh lại tài liệu với một mốc đóng băng khác -> tài liệu phải theo mốc đó."""
    fsp = sandbox["module"]
    from src.freeze_serving_policy import render_markdown, stamp

    chosen = "2019-05-04T03:02:01+00:00"
    md_text = render_markdown(stamp(real_content, chosen))
    stamps = set(re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?", md_text))
    assert stamps == {chosen}, (
        f"tài liệu sinh ra phải chỉ chứa mốc đóng bang {chosen}, thực tế {stamps}"
    )

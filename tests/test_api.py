"""
tests/test_api.py

Test cho FastAPI: route, validation, error handling, web routes, dashboard.

Nguyên tắc được kiểm chứng ở đây:
  - Web/API CHỈ load artifact đã đóng băng, KHÔNG train, KHÔNG tune, KHÔNG refit.
  - Input sai -> HTTP 4xx + JSON rõ ràng, KHÔNG crash server.
  - Prediction luôn hữu hạn, không âm.
  - Số liệu dashboard đến từ artifact, KHÔNG hard-code trong HTML/JS.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.schemas import ForecastRequest  # noqa: E402

from conftest import requires_artifacts  # noqa: E402

APP_DIR = ROOT / "app"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

WEB_ROUTES = ["/", "/du-bao", "/dashboard"]
STATIC_ASSETS = [
    "/static/css/style.css",
    "/static/js/predict.js",
    "/static/js/dashboard.js",
]


# ===========================================================================
# Hệ thống
# ===========================================================================
@requires_artifacts
def test_health_returns_200_and_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"]["name"]
    assert body["model"]["alpha"] is not None
    required = [a for a in body["artifacts"] if a["required"]]
    assert required and all(a["present"] for a in required)


def test_health_reports_503_with_clear_reason_when_artifact_missing(client_without_artifacts):
    """Thiếu artifact -> 503 + thông báo rõ, KHÔNG crash, KHÔNG 500 mơ hồ."""
    r = client_without_artifacts.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "ridge_pipeline.joblib" in body["detail"]
    assert "src/train.py" in body["detail"]  # hướng dẫn cách tạo lại


def test_forecast_endpoint_fails_clearly_when_artifact_missing(client_without_artifacts, valid_payload):
    r = client_without_artifacts.post("/api/traffic-forecast", json=valid_payload)
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "model_artifact_unavailable"
    assert "ridge_pipeline.joblib" in body["message"]


def test_model_info_fails_clearly_when_artifact_missing(client_without_artifacts):
    r = client_without_artifacts.get("/api/model-info")
    assert r.status_code == 503
    assert r.json()["error"] == "model_artifact_unavailable"


@requires_artifacts
def test_model_info_matches_metadata_artifact(client, artifacts):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model"]["name"] == artifacts.metadata["model_name"]
    assert body["model"]["hyperparameters"] == artifacts.metadata["hyperparameters"]
    assert body["model"]["alpha"] == artifacts.alpha
    assert body["model"]["target"] == config.TARGET_NAME
    assert body["feature_columns"] == artifacts.metadata["feature_columns"]
    assert body["time_split"]["train"]["n_rows"] == artifacts.metadata["time_split"]["train"]["n_rows"]
    assert body["validation_metrics"] == artifacts.metadata["validation_metrics"]


@requires_artifacts
def test_model_info_documents_serving_policy(client):
    body = client.get("/api/model-info").json()
    serving = body["serving"]
    assert serving["trains_at_serving"] is False
    assert serving["tunes_at_serving"] is False
    assert serving["fits_preprocessors_at_serving"] is False
    assert serving["reads_target_at_serving"] is False
    assert serving["uses_training_feature_function"] == "src.features.build_features"
    assert serving["celsius_to_kelvin_offset"] == pytest.approx(273.15)
    assert serving["postprocess_policy"] == "max(0, raw_prediction)"
    assert len(serving["allowed_weather"]) == 11
    assert serving["state_fair_known_years"] == [2012, 2020]
    assert serving["unit"] == "xe/giờ"


@requires_artifacts
def test_model_info_contains_no_personal_machine_path(client):
    """Không được lộ đường dẫn máy của người chạy (C:\\Users\\...)."""
    raw = client.get("/api/model-info").text
    assert "C:/Users" not in raw and "C:\\Users" not in raw
    for art in client.get("/api/model-info").json()["artifacts"]:
        assert not art["path"].startswith("C:")


# ===========================================================================
# Web routes
# ===========================================================================
@pytest.mark.parametrize("route", WEB_ROUTES)
def test_web_routes_return_200(client, route):
    r = client.get(route)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.parametrize("route", WEB_ROUTES)
def test_web_routes_are_vietnamese(client, route):
    body = client.get(route).text
    assert "<html lang=\"vi\">" in body
    assert "charset=\"utf-8\"" in body


@pytest.mark.parametrize("asset", STATIC_ASSETS)
def test_static_assets_return_200(client, asset):
    assert client.get(asset).status_code == 200


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


@requires_artifacts
def test_openapi_declares_required_endpoints(client):
    """Các endpoint API phải xuất hiện trong OpenAPI; các trang HTML thì kiểm tra riêng."""
    spec = client.get("/openapi.json").json()
    for path in ("/health", "/api/model-info", "/api/traffic-forecast", "/api/dashboard-metrics"):
        assert path in spec["paths"], f"thiếu {path} trong OpenAPI"
    assert "post" in spec["paths"]["/api/traffic-forecast"]
    # ba màn hình web là HTML tĩnh, kiểm tra bằng test_web_routes_return_200
    for route in WEB_ROUTES:
        assert client.get(route).status_code == 200


def test_all_three_screens_exist():
    for name in ("index.html", "predict.html", "dashboard.html"):
        assert (TEMPLATES_DIR / name).exists(), f"thiếu màn hình {name}"


def test_forecast_form_calls_the_real_endpoint():
    js = (STATIC_DIR / "js" / "predict.js").read_text(encoding="utf-8")
    assert '"/api/traffic-forecast"' in js
    assert 'method: "POST"' in js


def test_forecast_form_has_required_inputs():
    html = (TEMPLATES_DIR / "predict.html").read_text(encoding="utf-8")
    for field_id in ("date", "time", "temperature", "clouds", "rain", "snow"):
        assert f'id="{field_id}"' in html, f"thiếu ô nhập {field_id}"
    # KHÔNG được bắt người dùng nhập is_holiday
    assert 'id="holiday"' not in html
    assert "DỰ BÁO LƯU LƯỢNG" in html


# ===========================================================================
# Dự báo hợp lệ
# ===========================================================================
@requires_artifacts
def test_valid_prediction_returns_200(client, valid_payload):
    r = client.post("/api/traffic-forecast", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["unit"] == "vehicles_per_hour"
    assert body["unit_vi"] == "xe/giờ"
    assert body["model"]["artifact"] == "models/ridge_pipeline.joblib"
    assert body["model"]["alpha"] == 0.001


@requires_artifacts
def test_valid_prediction_is_numeric_and_finite(client, valid_payload):
    body = client.post("/api/traffic-forecast", json=valid_payload).json()
    value = body["predicted_traffic_volume"]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value)
    assert value >= 0.0
    assert math.isfinite(body["raw_model_output"])


@requires_artifacts
def test_response_echoes_input_and_derived_features(client, valid_payload):
    body = client.post("/api/traffic-forecast", json=valid_payload).json()
    assert body["input_echo"]["date_time"] == "2018-06-15T08:00:00"
    assert body["input_echo"]["temperature_celsius"] == 20.0
    assert body["derived_features"]["temp"] == pytest.approx(293.15, abs=0.01)
    assert body["derived_features"]["clouds_all"] == 20
    assert body["calendar"]["hour"] == 8
    assert body["calendar"]["day_of_week_name"] == "Thứ 6"
    assert body["calendar"]["month"] == 6


@requires_artifacts
def test_celsius_conversion_reported_in_response(client, valid_payload):
    body = client.post(
        "/api/traffic-forecast", json=dict(valid_payload, temperature_celsius=-15.5)
    ).json()
    assert body["input_echo"]["temperature_celsius"] == -15.5
    assert body["derived_features"]["temp"] == pytest.approx(257.65, abs=0.01)


@requires_artifacts
def test_holiday_computed_automatically(client, valid_payload):
    """Không có trường is_holiday trong request, nhưng response phải có cờ đúng."""
    body = client.post(
        "/api/traffic-forecast", json=dict(valid_payload, date_time="2018-07-04T09:00:00")
    ).json()
    assert body["holiday"]["is_holiday"] == 1
    assert body["holiday"]["holiday_name"] == "Independence Day"

    body2 = client.post(
        "/api/traffic-forecast", json=dict(valid_payload, date_time="2018-06-15T09:00:00")
    ).json()
    assert body2["holiday"]["is_holiday"] == 0
    assert body2["holiday"]["holiday_name"] == "None"


@requires_artifacts
def test_multi_weather_produces_correct_features(client, valid_payload):
    body = client.post(
        "/api/traffic-forecast",
        json=dict(valid_payload, weather=["Rain", "Snow"], rain_1h_mm=1.5, snow_1h_mm=2.5),
    ).json()
    hot = body["weather"]["multi_hot"]
    assert hot["wm_rain"] == 1 and hot["wm_snow"] == 1
    assert hot["wm_clear"] == 0
    assert body["weather"]["requested"] == ["Rain", "Snow"]
    assert body["derived_features"]["weather_severity"] == 3
    assert body["calendar"]["is_extreme_weather"] is True


@requires_artifacts
def test_single_weather_accepted(client, valid_payload):
    for category in ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
                     "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]:
        r = client.post(
            "/api/traffic-forecast", json=dict(valid_payload, weather=[category])
        )
        assert r.status_code == 200, f"{category} bị từ chối"
        assert r.json()["weather"]["multi_hot"][f"wm_{category.lower()}"] == 1


@requires_artifacts
def test_peak_hour_predicts_more_than_night(client, valid_payload):
    peak = client.post(
        "/api/traffic-forecast", json=dict(valid_payload, date_time="2018-06-15T08:00:00")
    ).json()["predicted_traffic_volume"]
    night = client.post(
        "/api/traffic-forecast", json=dict(valid_payload, date_time="2018-06-15T03:00:00")
    ).json()["predicted_traffic_volume"]
    assert peak > night


# ===========================================================================
# Validation — input sai
# ===========================================================================
@requires_artifacts
@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"clouds_all": 101}, "clouds_all"),
        ({"clouds_all": 1000}, "clouds_all"),
        ({"clouds_all": -1}, "clouds_all"),
        ({"rain_1h_mm": -0.1}, "rain_1h_mm"),
        ({"snow_1h_mm": -5.0}, "snow_1h_mm"),
        ({"temperature_celsius": -100.0}, "temperature_celsius"),
        ({"temperature_celsius": 100.0}, "temperature_celsius"),
        ({"weather": ["Blizzard"]}, "weather"),
        ({"weather": ["clear"]}, "weather"),  # sai hoa/thường -> không tự sửa
        ({"weather": []}, "weather"),
        ({"date_time": "2018-13-45T99:00:00"}, "date_time"),
    ],
)
def test_invalid_input_returns_422_with_field_name(client, valid_payload, overrides, field):
    r = client.post("/api/traffic-forecast", json=dict(valid_payload, **overrides))
    assert r.status_code == 422, f"{overrides} phải bị 422"
    body = r.json()
    assert body["error"] == "validation_error"
    assert body["message"]
    assert any(d["field"] == field for d in body["details"]), body["details"]


@requires_artifacts
def test_unknown_weather_lists_allowed_categories(client, valid_payload):
    r = client.post("/api/traffic-forecast", json=dict(valid_payload, weather=["Hurricane"]))
    assert r.status_code == 422
    message = r.json()["message"]
    for category in ("Clear", "Thunderstorm", "Squall"):
        assert category in message, "thông báo phải liệt kê danh mục hợp lệ"


@requires_artifacts
@pytest.mark.parametrize("missing", ["date_time", "temperature_celsius", "clouds_all", "weather"])
def test_missing_field_returns_422(client, valid_payload, missing):
    payload = {k: v for k, v in valid_payload.items() if k != missing}
    r = client.post("/api/traffic-forecast", json=payload)
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == missing


@requires_artifacts
@pytest.mark.parametrize("bad_dt", ["15/06/2018", "2018-13-45T99:00:00", "not-a-date", "20180615T080000", ""])
def test_malformed_datetime_returns_422(client, valid_payload, bad_dt):
    r = client.post("/api/traffic-forecast", json=dict(valid_payload, date_time=bad_dt))
    assert r.status_code == 422
    assert any(d["field"] == "date_time" for d in r.json()["details"])


@requires_artifacts
def test_unknown_field_is_rejected_not_silently_ignored(client, valid_payload):
    """Không âm thầm bỏ qua trường lạ — người dùng phải được báo sai chỗ."""
    r = client.post("/api/traffic-forecast", json=dict(valid_payload, traffic_volume=9999))
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == "traffic_volume"


@requires_artifacts
def test_body_not_a_json_object_returns_422(client):
    r = client.post(
        "/api/traffic-forecast",
        content=b"[1, 2, 3]",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


@requires_artifacts
def test_server_survives_a_burst_of_bad_requests(client, valid_payload):
    """Sau nhiều request lỗi, server vẫn phục vụ bình thường."""
    bad_payloads = [
        dict(valid_payload, clouds_all=999),
        dict(valid_payload, weather=["Nope"]),
        {"date_time": "x"},
        dict(valid_payload, temperature_celsius="abc"),
        {},
    ]
    for payload in bad_payloads:
        client.post("/api/traffic-forecast", json=payload)
    ok = client.post("/api/traffic-forecast", json=valid_payload)
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200


@requires_artifacts
def test_optional_precipitation_defaults_to_zero(client, valid_payload):
    payload = {k: v for k, v in valid_payload.items()
               if k not in ("rain_1h_mm", "snow_1h_mm")}
    r = client.post("/api/traffic-forecast", json=payload)
    assert r.status_code == 200
    assert r.json()["input_echo"]["rain_1h_mm"] == 0.0


@requires_artifacts
def test_state_fair_must_be_same_year(client, valid_payload):
    r = client.post(
        "/api/traffic-forecast",
        json=dict(valid_payload, state_fair_start_date="2019-08-22"),
    )
    assert r.status_code == 422
    assert any(d["field"] == "state_fair_start_date" for d in r.json()["details"])


@requires_artifacts
def test_schema_documents_validation_rules():
    """Ràng buộc phải được MÔ TẢ trong OpenAPI, không chỉ nằm trong code."""
    schema = ForecastRequest.model_json_schema()
    props = schema["properties"]
    assert props["clouds_all"]["minimum"] == 0
    assert props["clouds_all"]["maximum"] == 100
    assert props["rain_1h_mm"]["minimum"] == 0
    assert props["snow_1h_mm"]["minimum"] == 0
    assert props["temperature_celsius"]["minimum"] == config.TEMPERATURE_CELSIUS_MIN
    assert props["temperature_celsius"]["maximum"] == config.TEMPERATURE_CELSIUS_MAX
    assert props["weather"]["minItems"] == 1
    assert len(props["weather"]["items"]["enum"]) == 11
    assert "273,15" in props["temperature_celsius"]["description"]


# ===========================================================================
# Dashboard — số liệu phải đến từ artifact, không hard-code
# ===========================================================================
@requires_artifacts
def test_dashboard_metrics_endpoint_returns_200(client):
    r = client.get("/api/dashboard-metrics")
    assert r.status_code == 200
    body = r.json()
    for key in ("final_test", "time_split", "error_analysis", "alpha_tuning",
                "experiments", "dataset", "model", "actual_vs_predicted"):
        assert key in body, f"thiếu khối {key}"


@requires_artifacts
def test_dashboard_metrics_equal_the_artifact_values(client, artifacts):
    """Đối chiếu TỪNG SỐ với file artifact — chứng minh không có số bịa/bị hard-code."""
    body = client.get("/api/dashboard-metrics").json()
    assert artifacts.evaluation is not None, "cần reports/figures/evaluation_results.json"

    assert body["final_test"] == artifacts.evaluation["final_test"]
    assert body["error_analysis"] == artifacts.evaluation["error_analysis"]
    assert body["time_split"] == artifacts.evaluation["time_split"]
    assert body["alpha_tuning"] == artifacts.alpha_tuning
    assert body["experiments"] == artifacts.experiments
    assert body["dataset"]["raw_rows"] == artifacts.data_audit["n_raw_rows"]
    assert body["dataset"]["modeling_rows"] == artifacts.data_audit["n_rows_after_collapse"]
    assert body["model"]["hyperparameters"] == artifacts.metadata["hyperparameters"]


@requires_artifacts
def test_dashboard_distinguishes_final_test_from_development_experiments(client, artifacts):
    body = client.get("/api/dashboard-metrics").json()
    # FINAL TEST nằm trong evaluation_results (nguồn riêng)
    assert body["final_test"]["ridge"]["n"] == artifacts.time_split["test"]["n_rows"]
    # Thí nghiệm phát triển nằm trong experiments_results và được ghi rõ không chứa 2018
    assert body["experiments"]["dev_window"]["excluded"].startswith("2018")
    assert body["experiments"]["experiment_1_random_vs_time_split"]["window"]["contains_2018"] is False
    assert body["config"]["dev_experiments_window"].startswith("2012-2017")


@requires_artifacts
def test_dashboard_error_segments_include_sample_sizes(client):
    body = client.get("/api/dashboard-metrics").json()
    ea = body["error_analysis"]
    for seg in ea["holiday"]["segments"]:
        assert "n_samples" in seg
    for seg in ea["extreme_weather"]["segments"]:
        assert "n_samples" in seg
        if seg["n_samples"] == 0:
            assert seg["MAE"] is None, "phân khúc rỗng không được báo số 0"
    assert "mae_delta" in ea["holiday"]
    assert "n_holiday_dates" in ea["holiday"]


@requires_artifacts
def test_dashboard_includes_experiment_1_and_1b(client):
    exp = client.get("/api/dashboard-metrics").json()["experiments"]
    assert "experiment_1_random_vs_time_split" in exp
    assert "experiment_1b_leakage_controlled" in exp
    assert "experiment_1c_block_neighbour" in exp

    e1b = exp["experiment_1b_leakage_controlled"]
    assert set(e1b["arm_labels"]) == {
        "A_train_le_2015", "B_train_le_2016",
        "C_train_le_2016_plus_half_2017", "D_same_n_as_B_random_from_C",
    }
    for run in e1b["per_seed"]:
        shared = {a["n"] for a in run["arms"].values()}
        assert len(shared) == 1, "mọi arm phải dùng CHUNG một tập test"
        assert shared == {run["shared_test_n"]}
    # arm D là arm đối chứng cùng kích thước với arm B
    for run in e1b["per_seed"]:
        arms = run["arms"]
        assert (
            arms["D_same_n_as_B_random_from_C"]["n_train"]
            == arms["B_train_le_2016"]["n_train"]
        )


@requires_artifacts
def test_dashboard_includes_rolling_origin_drift(client):
    exp = client.get("/api/dashboard-metrics").json()["experiments"]
    rolling = exp["experiment_3_rolling_origin"]
    assert len(rolling["folds"]) >= 3
    assert all(f["evaluation_type"] == "out_of_sample" for f in rolling["folds"])
    assert rolling["pooled_out_of_sample"]["n"] > 0


@requires_artifacts
@pytest.mark.parametrize(
    "name",
    ["templates/index.html", "templates/predict.html", "templates/dashboard.html",
     "static/js/predict.js", "static/js/dashboard.js", "static/css/style.css"],
)
def test_no_final_test_metric_is_hardcoded_in_frontend(name, artifacts):
    """HTML/JS/CSS KHÔNG được chứa bất kỳ metric FINAL TEST nào dạng chữ.

    Nếu con số ở đây khớp artifact thì bằng chứng 'không hard-code' mới có ý nghĩa.
    """
    path = (APP_DIR / name).resolve()
    text = path.read_text(encoding="utf-8")
    forbidden = []
    for arm in ("ridge", "baseline"):
        for key in ("MAE", "RMSE"):
            value = artifacts.evaluation["final_test"][arm][key]
            for text_form in (str(value), str(value).replace(".", ",")):
                if re.search(rf"(?<![\d.]){re.escape(text_form)}(?![\d])", text):
                    forbidden.append((arm, key, text_form))
    assert not forbidden, f"{name} chứa số liệu hard-code: {forbidden}"
    # và số dòng dataset cũng không được hard-code
    for value in (artifacts.data_audit["n_raw_rows"],
                  artifacts.data_audit["n_rows_after_collapse"]):
        assert not re.search(rf"(?<![\d.]){value}(?![\d])", text), (
            f"{name} chứa số dòng dataset hard-code: {value}"
        )
    # n của FINAL TEST cũng phải đến từ artifact
    n_test = artifacts.time_split["test"]["n_rows"]
    assert not re.search(rf"(?<![\d.]){n_test}(?![\d])", text), (
        f"{name} chứa số dòng FINAL TEST hard-code: {n_test}"
    )
    # R² cũng không được hard-code
    for arm in ("ridge", "baseline"):
        r2 = str(artifacts.evaluation["final_test"][arm]["R2"])
        assert not re.search(rf"(?<![\d.]){re.escape(r2)}(?![\d])", text), (
            f"{name} chứa R² hard-code: {r2}"
        )
    # và cả MAE sau khi chặn âm (số nhạy cảm cũng phải lấy từ artifact/tài liệu, không gõ trong UI)
    for value in ("257,54", "257.54"):
        assert not re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", text), (
            f"{name} chứa MAE sau hậu xử lý hard-code: {value}"
        )


@requires_artifacts
def test_frontend_fetches_metrics_from_api(client):
    js = (STATIC_DIR / "js" / "dashboard.js").read_text(encoding="utf-8")
    assert '"/api/dashboard-metrics"' in js
    assert "fetch(" in js


@pytest.mark.parametrize(
    "name", ["predict.js", "dashboard.js"]
)
def test_frontend_js_is_syntactically_valid(name):
    """Chặn lỗi cú pháp JS — nếu sai, trang im lặng và demo hỏng.

    Dùng `node --check` nếu có Node.js; nếu không có thì bỏ qua (không bắt buộc
    cài Node để chạy test Python).
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Không có Node.js để kiểm tra cú pháp JS")
    result = subprocess.run(
        [node, "--check", str(STATIC_DIR / "js" / name)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Lỗi cú pháp trong {name}:\n{result.stderr}"


@pytest.mark.parametrize("name", ["predict.js", "dashboard.js"])
def test_frontend_js_only_calls_functions_it_defines(name):
    """Mọi hàm nội bộ gọi trong JS phải được ĐỊNH NGHĨA trong chính file đó.

    Bắt được loại lỗi runtime kiểu "gọi hàm của file khác" (ví dụ dùng `fmt()`
    trong `dashboard.js` trong khi hàm đó chỉ tồn tại ở `predict.js`).
    Loại lỗi này `node --check` KHÔNG bắt được vì cú pháp vẫn hợp lệ.
    """
    import re

    source = (STATIC_DIR / "js" / name).read_text(encoding="utf-8")

    # Bỏ comment và nội dung chuỗi — chỉ xét mã thực thi
    code = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    code = re.sub(r"//[^\n]*", " ", code)
    code = re.sub(r"'[^'\n]*'", " '' ", code)
    code = re.sub(r'"[^"\n]*"', ' "" ', code)

    # các hàm được định nghĩa trong file
    defined = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)\s*\(", code))
    defined |= set(re.findall(r"(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=\s*function", code))
    # các hàm có sẵn của trình duyệt — không cần định nghĩa
    builtin = {
        "require", "fetch", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
        "console", "parseInt", "parseFloat", "isNaN", "Number", "String", "Math",
        "Object", "Array", "JSON", "Boolean", "Date", "Error", "Promise", "RegExp",
        "document", "window", "encodeURIComponent", "decodeURIComponent",
    }

    # các lời gọi hàm nội bộ: tên(...) không phải thuộc tính (loại bỏ `obj.fn(`)
    called = set(re.findall(r"(?<![\w.$])([a-zA-Z_$][\w$]*)\s*\(", code))
    keywords = {
        "if", "for", "while", "switch", "catch", "function", "return", "typeof",
        "else", "do", "try", "new", "delete", "in", "of", "void", "await",
    }
    unknown = called - defined - builtin - keywords

    assert not unknown, (
        f"{name} gọi hàm không định nghĩa: {sorted(unknown)}. "
        "Kiểm tra xem có vô tình dùng hàm của file kia không."
    )


@pytest.mark.parametrize("name", ["predict.js", "dashboard.js"])
def test_frontend_js_has_no_placeholder_or_debug_leftovers(name):
    source = (STATIC_DIR / "js" / name).read_text(encoding="utf-8")
    for bad in ("TODO", "FIXME", "XXX", "console.log(", "debugger"):
        assert bad not in source, f"{name} còn sót {bad!r}"
    assert "�" not in source, f"{name} chứa ký tự hỏng (mojibake)"


@pytest.mark.parametrize("name", ["index.html", "predict.html", "dashboard.html"])
def test_html_templates_have_balanced_tags_and_vi_lang(name):
    html = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    assert html.count("<html") == 1 and html.count("</html>") == 1
    assert html.count("<body") == 1 and html.count("</body>") == 1
    assert html.count("<main") == 1 and html.count("</main>") == 1
    assert 'lang="vi"' in html


# ===========================================================================
# Không train-serving skew ở tầng API
# ===========================================================================
@requires_artifacts
def test_api_and_batch_feature_paths_agree(client, artifacts):
    """Cùng một input, đi qua API và qua hàm batch phải cho CÙNG feature + CÙNG dự báo."""
    import pandas as pd

    from app.serving import predict
    from src.features import FEATURE_COLUMNS_ALL

    payload = {
        "date_time": "2018-08-23T15:00:00",
        "temperature_celsius": 24.0,
        "rain_1h_mm": 0.0,
        "snow_1h_mm": 0.0,
        "clouds_all": 40,
        "weather": ["Clouds"],
    }
    api_body = client.post("/api/traffic-forecast", json=payload).json()
    direct = predict(artifacts, {**payload, "date_time": __import__("datetime").datetime(2018, 8, 23, 15, 0, 0)})

    assert api_body["derived_features"] == direct["derived_features"]
    assert api_body["predicted_traffic_volume"] == direct["predicted_traffic_volume"]
    assert sorted(api_body["derived_features"]) == sorted(FEATURE_COLUMNS_ALL)


@requires_artifacts
def test_pipeline_is_not_refit_at_serving(client, artifacts):
    """Gọi API nhiều lần không được làm thay đổi thống kê đã học trong pipeline."""
    scaler_before = artifacts.pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"].mean_.copy()
    encoder_before = [
        list(c) for c in
        artifacts.pipeline.named_steps["preprocess"].named_transformers_["cat"].categories_
    ]
    for hour in range(0, 24, 3):
        client.post("/api/traffic-forecast", json={
            "date_time": f"2018-06-15T{hour:02d}:00:00",
            "temperature_celsius": -10.0 + hour,
            "clouds_all": hour * 4,
            "weather": ["Rain", "Snow"],
        })
    scaler_after = artifacts.pipeline.named_steps["preprocess"].named_transformers_["num"].named_steps["scaler"].mean_
    encoder_after = [
        list(c) for c in
        artifacts.pipeline.named_steps["preprocess"].named_transformers_["cat"].categories_
    ]
    assert (scaler_before == scaler_after).all(), "scaler đã bị thay đổi khi phục vụ!"
    assert encoder_before == encoder_after, "encoder đã bị thay đổi khi phục vụ!"


@requires_artifacts
def test_server_reads_no_target_at_serving(client, artifacts, tmp_path, monkeypatch):
    """Nếu không có dataset trên đĩa, API vẫn dự báo được (không đọc target)."""
    monkeypatch.setenv(config.ENV_DATA_DIR, str(tmp_path / "khong-co-du-lieu"))
    from app import registry

    registry.reset_artifacts_cache()
    try:
        r = client.post("/api/traffic-forecast", json={
            "date_time": "2018-06-15T08:00:00",
            "temperature_celsius": 20.0,
            "clouds_all": 20,
            "weather": ["Clear"],
        })
        assert r.status_code == 200
        assert math.isfinite(r.json()["predicted_traffic_volume"])
    finally:
        monkeypatch.undo()
        registry.reset_artifacts_cache()

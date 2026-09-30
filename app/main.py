"""
app/main.py

FastAPI app phục vụ mô hình Ridge đã đóng băng.

Chạy local:
    py -m uvicorn app.main:app --reload
    http://localhost:8000        (màn hình giới thiệu)
    http://localhost:8000/docs   (OpenAPI)

Nguyên tắc: app CHỈ load artifact. Không train, không tune, không refit,
không đọc target từ dataset.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import config
from app.registry import ArtifactError, Artifacts, get_artifacts
from app.schemas import ErrorResponse, ForecastRequest, ForecastResponse
from app.serving import ServingError, predict

_ROOT = config.ROOT_DIR
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.holidays import KNOWN_YEARS  # noqa: E402
from src.weather import WEATHER_MAIN_CATEGORIES  # noqa: E402

logger = logging.getLogger("traffic_forecast.app")

SERVICE_NAME = "traffic-forecast-api"
SERVICE_VERSION = "checkpoint3.1"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Nạp artifact sẵn để lần request đầu tiên nhanh.

    Lỗi artifact KHÔNG làm sập server: server vẫn khởi động, `/health` trả 503
    kèm lý do cụ thể để người dùng biết chạy lệnh nào để tạo lại model.
    """
    try:
        art = get_artifacts()
        logger.info(
            "Đã load %s (alpha=%s) từ %s",
            art.model_name,
            art.alpha,
            (art.models_dir / "ridge_pipeline.joblib").as_posix(),
        )
    except ArtifactError as exc:
        logger.error("KHÔNG load được model: %s", exc)
    yield


app = FastAPI(
    title="Dự báo lưu lượng giao thông I-94 (westbound) — ATR 301",
    version=SERVICE_VERSION,
    description=(
        "API dự báo `traffic_volume` (xe/giờ) từ lịch và thời tiết.\n\n"
        "Mô hình **Ridge (alpha đã đóng băng)** được load từ "
        "`models/ridge_pipeline.joblib`. Ứng dụng KHÔNG train, KHÔNG tune, "
        "KHÔNG refit encoder/imputer/scaler, và KHÔNG đọc target từ dataset.\n\n"
        "Mọi feature kỹ thuật hoá được sinh bằng CHUNG hàm `src.features.build_features` "
        "với lúc huấn luyện => không có train-serving skew."
    ),
    contact={"name": "Nhóm 20 — Bài 7 (Data leakage & honest evaluation)"},
    license_info={"name": "Dữ liệu: UCI Metro Interstate Traffic Volume (CC BY 4.0)"},
    lifespan=lifespan,
)

config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

#: Phục vụ hình ảnh thật sinh từ `src/evaluate.py` (không nhân bản, không hard-code)
if config.reports_dir().exists():
    app.mount(
        "/figures", StaticFiles(directory=str(config.reports_dir())), name="figures"
    )


# ---------------------------------------------------------------------------
# Dependency + xử lý lỗi
# ---------------------------------------------------------------------------
def get_pipeline_artifacts() -> Artifacts:
    """Nạp artifact đã đóng băng; lỗi -> 503 kèm hướng dẫn rõ ràng."""
    try:
        return get_artifacts()
    except ArtifactError as exc:
        logger.error("Artifact không sẵn sàng: %s", exc)
        raise StarletteHTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_artifact_unavailable",
                "message": str(exc),
                "missing": exc.missing,
            },
        ) from exc


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Lỗi input -> 422 với JSON đọc được: field, message, type."""
    details = []
    for err in exc.errors():
        raw_loc = [p for p in err.get("loc", []) if p not in ("body", "query", "path")]
        # gộp chỉ số mảng vào tên trường: "weather.0" -> "weather" (dễ hiểu hơn)
        parts: list[str] = []
        for part in raw_loc:
            if isinstance(part, int) and parts:
                continue
            parts.append(str(part))
        details.append(
            {
                "field": ".".join(parts) if parts else "(body)",
                "message": err.get("msg", "giá trị không hợp lệ"),
                "type": err.get("type", "value_error"),
            }
        )
    first = details[0]["message"] if details else "Dữ liệu gửi lên không hợp lệ."
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            message=f"Input không hợp lệ: {first}",
            details=details,
        ).model_dump(),
    )


@app.exception_handler(ServingError)
async def serving_exception_handler(request: Request, exc: ServingError):
    logger.exception("Lỗi serving: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="serving_error",
            message=str(exc),
            details=[],
        ).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error", message=str(exc.detail), details=[]
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):  # noqa: BLE001
    """Không để server crash vì một request hỏng."""
    logger.exception("Lỗi không lường trước tại %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            message="Lỗi nội bộ của server. Chi tiết đã ghi vào log.",
            details=[],
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Trang HTML
# ---------------------------------------------------------------------------
def _page(filename: str, title: str) -> FileResponse:
    path = config.TEMPLATES_DIR / filename
    if not path.exists():  # pragma: no cover
        return FileResponse(
            config.TEMPLATES_DIR / "index.html",
            media_type="text/html",
            headers={"X-Page-Title": title},
        )
    return FileResponse(path, media_type="text/html; charset=utf-8")


@app.get("/", include_in_schema=False)
def page_home() -> FileResponse:
    """Màn 1 — Giới thiệu."""
    return _page("index.html", "Giới thiệu")


@app.get("/du-bao", include_in_schema=False)
def page_forecast() -> FileResponse:
    """Màn 2 — Dự báo."""
    return _page("predict.html", "Dự báo")


@app.get("/dashboard", include_in_schema=False)
def page_dashboard() -> FileResponse:
    """Màn 3 — Dashboard + Model Card."""
    return _page("dashboard.html", "Dashboard & Model Card")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/health", response_model=None, tags=["Hệ thống"])
def health() -> JSONResponse:
    """Kiểm tra sống. 200 khi model đã load; 503 + lý do khi chưa."""
    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        art = get_artifacts()
    except ArtifactError as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "service": SERVICE_NAME,
                "model_loaded": False,
                "model": None,
                "artifacts": [],
                "detail": str(exc),
                "checked_at": checked_at,
            },
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "service": SERVICE_NAME,
            "model_loaded": True,
            "model": {
                "name": art.model_name,
                "version": art.metadata.get("model_version"),
                "alpha": art.alpha,
                "artifact": "models/ridge_pipeline.joblib",
            },
            "serving_policy": art.serving_policy_info(),
            "artifacts": art.artifact_manifest(),
            "detail": None,
            "checked_at": checked_at,
        },
    )


@app.get("/api/model-info", tags=["Hệ thống"])
def model_info(art: Artifacts = Depends(get_pipeline_artifacts)) -> dict[str, Any]:
    """Metadata + phạm vi sử dụng, đọc thẳng từ artifact (không hard-code)."""
    fit = art.metadata.get("fit_data") or {}
    split = art.time_split
    audit = art.data_audit or {}
    final = (art.evaluation or {}).get("final_test") or {}

    return {
        "service": {"name": SERVICE_NAME, "version": SERVICE_VERSION},
        "model": {
            "name": art.model_name,
            "version": art.metadata.get("model_version"),
            "created_by": art.metadata.get("created_by"),
            "seed": art.metadata.get("seed"),
            "alpha": art.alpha,
            "hyperparameters": art.metadata.get("hyperparameters"),
            "target": art.metadata.get("target"),
            "n_features_out": art.metadata.get("n_features_out"),
            "artifact": "models/ridge_pipeline.joblib",
            "trained_at_scope": fit,
        },
        "preprocessing": art.metadata.get("preprocessing"),
        "feature_columns": art.metadata.get("feature_columns"),
        "time_split": split,
        "validation_metrics": art.metadata.get("validation_metrics"),
        "baseline_validation_metrics": art.metadata.get("baseline_validation_metrics"),
        "alpha_tuning": art.metadata.get("alpha_tuning"),
        "final_test_metrics": final,
        "dataset": {
            "raw_rows": audit.get("n_raw_rows"),
            "modeling_rows": audit.get("n_rows_after_collapse"),
            "rows_removed_by_collapse": audit.get("n_rows_removed_by_collapse"),
            "coverage": art.coverage,
        },
        "serving": {
            "trains_at_serving": False,
            "tunes_at_serving": False,
            "fits_preprocessors_at_serving": False,
            "reads_target_at_serving": False,
            "uses_training_feature_function": "src.features.build_features",
            "celsius_to_kelvin_offset": config.KELVIN_OFFSET,
            "holiday_source": "src/holidays.py (lịch tất định, không đọc dataset)",
            "state_fair_known_years": [min(KNOWN_YEARS), max(KNOWN_YEARS)],
            "weather_encoding": "multi-hot, nhiều hiện tượng cùng lúc",
            "allowed_weather": list(WEATHER_MAIN_CATEGORIES),
            "temperature_celsius_range": [
                config.TEMPERATURE_CELSIUS_MIN,
                config.TEMPERATURE_CELSIUS_MAX,
            ],
            "clouds_all_range": [0, 100],
            "precipitation_range_mm": [0.0, config.PRECIPITATION_MM_MAX],
            "postprocess_policy": config.NON_NEGATIVE_POLICY_ID,
            "postprocess_policy_vi": config.NON_NEGATIVE_POLICY_VI,
            "metric_convention": {
                "raw_model": (
                    "Ridge — metric của mô hình. Đây là số được báo cáo trong "
                    "evaluation_results.json và là kết luận chính thức."
                ),
                "deployed_predictor": (
                    "max(0,·) ∘ Ridge — wrapper phục vụ. Metric báo riêng, "
                    "KHÔNG gọi chung tên với metric của mô hình."
                ),
            },
            "unit": config.TARGET_UNIT_VI,
        },
        "serving_policy": art.serving_policy_info(),
        "artifacts": art.artifact_manifest(),
    }


@app.post(
    "/api/traffic-forecast",
    response_model=ForecastResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Input không hợp lệ"},
        503: {"model": ErrorResponse, "description": "Artifact model chưa sẵn sàng"},
    },
    tags=["Dự báo"],
)
def traffic_forecast(
    payload: ForecastRequest, art: Artifacts = Depends(get_pipeline_artifacts)
) -> dict[str, Any]:
    """Dự báo lưu lượng cho MỘT mốc thời gian."""
    data = payload.model_dump()
    return predict(art, data)


@app.get("/api/dashboard-metrics", tags=["Dashboard"])
def dashboard_metrics(art: Artifacts = Depends(get_pipeline_artifacts)) -> dict[str, Any]:
    """Số liệu dashboard.

    MỌI con số ở đây đến từ artifact thật
    (`reports/figures/evaluation_results.json`, `experiments_results.json`,
    `models/model_metadata.json`, `data/processed/data_audit.json`).
    Front-end KHÔNG hard-code bất kỳ metric nào.
    """
    return {
        "generated_from": {
            "evaluation_results": "reports/figures/evaluation_results.json",
            "experiments_results": "reports/figures/experiments_results.json",
            "alpha_tuning": "reports/figures/alpha_tuning.json",
            "model_metadata": "models/model_metadata.json",
            "data_audit": "data/processed/data_audit.json",
        },
        "config": (art.evaluation or {}).get("config"),
        "serving_policy": art.serving_policy_info(),
        "postprocess_audit": art.postprocess_audit,
        "time_split": (art.evaluation or {}).get("time_split") or art.time_split,
        "final_test": (art.evaluation or {}).get("final_test"),
        "validation_reference": (art.evaluation or {}).get("validation_reference"),
        "error_analysis": (art.evaluation or {}).get("error_analysis"),
        "figures": (art.evaluation or {}).get("figures"),
        "alpha_tuning": art.alpha_tuning,
        "experiments": art.experiments,
        "dataset": {
            "raw_rows": (art.data_audit or {}).get("n_raw_rows"),
            "modeling_rows": (art.data_audit or {}).get("n_rows_after_collapse"),
            "rows_removed_by_collapse": (art.data_audit or {}).get("n_rows_removed_by_collapse"),
            "duplicate_groups": ((art.data_audit or {}).get("duplicates") or {}).get(
                "n_duplicate_groups"
            ),
            "coverage": art.coverage,
        },
        "model": {
            "name": art.model_name,
            "version": art.metadata.get("model_version"),
            "hyperparameters": art.metadata.get("hyperparameters"),
            "preprocessing": art.metadata.get("preprocessing"),
            "feature_columns": art.metadata.get("feature_columns"),
            "n_features_out": art.metadata.get("n_features_out"),
            "fit_data": art.metadata.get("fit_data"),
            "alpha_tuning": art.metadata.get("alpha_tuning"),
        },
        "actual_vs_predicted": _actual_vs_predicted(art),
    }


def _actual_vs_predicted(art: Artifacts) -> dict[str, Any]:
    """Artifact KHÔNG lưu chuỗi actual/predicted từng giờ của FINAL TEST.

    Vì vậy ta hiển thị trung thực `mean_actual` vs `mean_pred` theo từng phân khúc
    (giờ) — đây là thứ CÓ thật trong `evaluation_results.json`.
    """
    ea = ((art.evaluation or {}).get("error_analysis") or {})
    rows = []
    for seg in (ea.get("hour") or {}).get("segments", []):
        rows.append(
            {
                "segment": seg.get("segment"),
                "n_samples": seg.get("n_samples"),
                "mean_actual": seg.get("mean_actual"),
                "mean_pred": seg.get("mean_pred"),
                "MAE": seg.get("MAE"),
                "baseline_MAE": seg.get("baseline_MAE"),
            }
        )
    return {
        "source": "reports/figures/evaluation_results.json → error_analysis.hour.segments",
        "granularity": "trung bình theo giờ trong ngày (không có chuỗi giờ từng dòng)",
        "note": (
            "Artifact lưu metric + mean theo phân khúc, không lưu từng cặp "
            "(actual, predicted) từng giờ. Dashboard vì vậy hiển thị mean_actual vs "
            "mean_pred theo giờ thay vì một đường cong toàn bộ chuỗi."
        ),
        "by_hour": rows,
    }


__all__ = ["app"]

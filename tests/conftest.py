"""
tests/conftest.py

Fixture dùng chung cho toàn bộ test:
- `client`      : TestClient FastAPI (dùng chung app, không reload).
- `artifacts`   : artifact đã đóng băng.
- `skip_if_no_artifacts`: bỏ qua (kèm lý do rõ) nếu chưa chạy `src/train.py`.
- `client_without_artifacts`: client trỏ tới thư mục model RỖNG, để chứng minh
  app báo lỗi rõ ràng thay vì crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, registry  # noqa: E402

REASON_NO_ARTIFACT = (
    "Chưa có artifact mô hình. Chạy:\n"
    "    py src\\download_data.py\n"
    "    py src\\data.py\n"
    "    py src\\train.py"
)

_PIPELINE = config.models_dir() / "ridge_pipeline.joblib"
_METADATA = config.models_dir() / "model_metadata.json"

has_artifacts = _PIPELINE.exists() and _METADATA.exists()
requires_artifacts = pytest.mark.skipif(not has_artifacts, reason=REASON_NO_ARTIFACT)


@pytest.fixture(scope="session")
def artifacts():
    if not has_artifacts:
        pytest.skip(REASON_NO_ARTIFACT)
    return registry.get_artifacts()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client_without_artifacts(tmp_path, monkeypatch):
    """Client khi thư mục model trống -> mọi endpoint dự báo phải fail RÕ, không crash."""
    from fastapi.testclient import TestClient

    from app.main import app

    empty_models = tmp_path / "models"
    empty_models.mkdir()
    empty_reports = tmp_path / "figures"
    empty_reports.mkdir()

    monkeypatch.setenv(config.ENV_MODELS_DIR, str(empty_models))
    monkeypatch.setenv(config.ENV_REPORTS_DIR, str(empty_reports))
    monkeypatch.setenv(config.ENV_DATA_DIR, str(tmp_path / "data"))
    registry.reset_artifacts_cache()
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        monkeypatch.undo()
        registry.reset_artifacts_cache()


@pytest.fixture()
def valid_payload():
    """Payload hợp lệ mẫu (giờ cao điểm, thời tiết đẹp, ngày thường)."""
    return {
        "date_time": "2018-06-15T08:00:00",
        "temperature_celsius": 20.0,
        "rain_1h_mm": 0.0,
        "snow_1h_mm": 0.0,
        "clouds_all": 20,
        "weather": ["Clear"],
    }

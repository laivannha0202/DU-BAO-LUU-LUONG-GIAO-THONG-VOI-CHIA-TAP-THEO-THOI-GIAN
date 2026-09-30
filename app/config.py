"""
app/config.py

Hằng số cấu hình + đường dẫn artifact cho lớp serving.

Mọi đường dẫn đều suy ra từ vị trí file, KHÔNG có đường dẫn cá nhân.
Có thể ghi đè qua biến môi trường (tiện cho test):
    TRAFFIC_MODELS_DIR, TRAFFIC_REPORTS_DIR, TRAFFIC_DATA_DIR
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
APP_DIR: Path = Path(__file__).resolve().parent

TEMPLATES_DIR: Path = APP_DIR / "templates"
STATIC_DIR: Path = APP_DIR / "static"

ENV_MODELS_DIR = "TRAFFIC_MODELS_DIR"
ENV_REPORTS_DIR = "TRAFFIC_REPORTS_DIR"
ENV_DATA_DIR = "TRAFFIC_DATA_DIR"


def models_dir() -> Path:
    return Path(os.environ.get(ENV_MODELS_DIR, ROOT_DIR / "models"))


def reports_dir() -> Path:
    """Thư mục `reports/figures` — nơi chứa evaluation_results.json ..."""
    return Path(os.environ.get(ENV_REPORTS_DIR, ROOT_DIR / "reports" / "figures"))


def data_dir() -> Path:
    return Path(os.environ.get(ENV_DATA_DIR, ROOT_DIR / "data" / "processed"))


#: artifact BẮT BUỘC phải có để phục vụ dự báo
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "ridge_pipeline.joblib",
    "model_metadata.json",
)
#: artifact TUỲ CHỌN — chỉ phục vụ hiển thị dashboard / model card
OPTIONAL_ARTIFACTS: tuple[str, ...] = (
    "run_config.json",
    "baseline_meta.json",
    "baseline_table.csv",
)

# ---------------------------------------------------------------------------
# Hằng số ngữ nghĩa (phải khớp với src/features.py và src/weather.py)
# ---------------------------------------------------------------------------
TARGET_NAME = "traffic_volume"
TARGET_UNIT_VI = "xe/giờ"
TARGET_UNIT_EN = "vehicles_per_hour"

#: Celsius -> Kelvin
KELVIN_OFFSET = 273.15

#: Chính sách hậu xử lý khi Ridge trả về giá trị âm.
#: Lưu ý: đây là quyết định của nhóm, KHÔNG phải kết quả học từ FINAL TEST 2018.
NON_NEGATIVE_POLICY_ID = "max(0, raw_prediction)"
NON_NEGATIVE_POLICY_VI = (
    "Lưu lượng không thể âm. Nếu Ridge trả về giá trị âm, API đặt kết quả về 0 "
    "và báo lại `raw_model_output` cùng cờ `clipped_to_zero`."
)

#: Khoảng hợp lệ của nhiệt độ mà API chấp nhận (°C).
#: Lý do: nhiệt kế trạm đo trong dataset nằm trong khoảng này; ngoài khoảng này
#: gần như chắc chắn là nhập sai đơn vị (ví dụ nhập Kelvin) hoặc nhập nhầm.
TEMPERATURE_CELSIUS_MIN = -70.0
TEMPERATURE_CELSIUS_MAX = 70.0

#: Trần vật lý cho lượng mưa/tuyết 1 giờ (mm). Dataset không có giá trị này lớn;
#: đặt trần để bắt lỗi nhập (ví dụ nhập tổng mưa cả ngày hoặc nhập sai đơn vị).
PRECIPITATION_MM_MAX = 500.0

#: Chỉ kiểm tra "hợp lệ về mặt hình thức" của năm (lỗi nhập rõ ràng).
DATE_YEAR_MIN = 1900
DATE_YEAR_MAX = 2200

__all__ = [
    "APP_DIR",
    "DATE_YEAR_MAX",
    "DATE_YEAR_MIN",
    "ENV_DATA_DIR",
    "ENV_MODELS_DIR",
    "ENV_REPORTS_DIR",
    "KELVIN_OFFSET",
    "NON_NEGATIVE_POLICY_ID",
    "NON_NEGATIVE_POLICY_VI",
    "OPTIONAL_ARTIFACTS",
    "PRECIPITATION_MM_MAX",
    "REQUIRED_ARTIFACTS",
    "ROOT_DIR",
    "STATIC_DIR",
    "TARGET_NAME",
    "TARGET_UNIT_EN",
    "TARGET_UNIT_VI",
    "TEMPLATES_DIR",
    "TEMPERATURE_CELSIUS_MAX",
    "TEMPERATURE_CELSIUS_MIN",
    "data_dir",
    "models_dir",
    "reports_dir",
]

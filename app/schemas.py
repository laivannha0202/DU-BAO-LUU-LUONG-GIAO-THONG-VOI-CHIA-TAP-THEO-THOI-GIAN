"""
app/schemas.py

Schema Pydantic cho API. Tất cả validation nằm ở đây để:
- input sai -> HTTP 4xx + JSON mô tả rõ trường nào sai, sai ở đâu;
- KHÔNG âm thầm sửa dữ liệu sai (vd tự đặt clouds về 100);
- KHÔNG làm crash server.

Lưu ý: `weather` dùng `Literal` trên đúng 11 danh mục của `src/weather.py`
=> gõ sai chính tả sẽ bị 422 kèm danh sách hợp lệ.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import config

_ROOT = config.ROOT_DIR
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.weather import WEATHER_MAIN_CATEGORIES  # noqa: E402

#: Kiểu literal dựng động từ danh mục chính thức -> không có hai nguồn sự thật.
WeatherCategory = Literal[
    "Clear",  # noqa: E501
    "Clouds",
    "Drizzle",
    "Fog",
    "Haze",
    "Mist",
    "Rain",
    "Smoke",
    "Snow",
    "Squall",
    "Thunderstorm",
]

#: Bảo đảm Literal khớp đúng WEATHER_MAIN_CATEGORIES
assert list(WeatherCategory.__args__) == list(WEATHER_MAIN_CATEGORIES), (
    "WeatherCategory phải khớp WEATHER_MAIN_CATEGORIES của src/weather.py"
)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    """Thông tin thực sự CÓ ĐƯỢC tại thời điểm dự báo.

    Người dùng KHÔNG cần nhập `is_holiday`, `hour`, `day_of_week`, `month`,
    `weather_main_mode`, `weather_family`, `weather_severity` — backend tự suy ra
    bằng đúng các hàm đã dùng lúc huấn luyện.
    """

    model_config = ConfigDict(
        extra="forbid",  # trường lạ -> 422, không bỏ qua âm thầm
        json_schema_extra={
            "example": {
                "date_time": "2018-06-15T08:00:00",
                "temperature_celsius": 20.0,
                "rain_1h_mm": 0.0,
                "snow_1h_mm": 0.0,
                "clouds_all": 20,
                "weather": ["Clear"],
            },
            "examples_multi_weather": [
                {
                    "date_time": "2018-01-22T15:00:00",
                    "temperature_celsius": -8.0,
                    "rain_1h_mm": 1.2,
                    "snow_1h_mm": 4.5,
                    "clouds_all": 90,
                    "weather": ["Rain", "Snow", "Thunderstorm"],
                }
            ],
        },
    )

    date_time: datetime = Field(
        ...,
        description=(
            "Thời điểm dự báo, định dạng ISO-8601 (vd '2018-06-15T08:00:00'). "
            "Chỉ dùng thông tin biết trước hoặc quan sát được tại đúng giờ đó."
        ),
    )
    temperature_celsius: float = Field(
        ...,
        ge=config.TEMPERATURE_CELSIUS_MIN,
        le=config.TEMPERATURE_CELSIUS_MAX,
        description=(
            f"Nhiệt độ không khí (°C), khoảng hợp lệ "
            f"{config.TEMPERATURE_CELSIUS_MIN:g} .. {config.TEMPERATURE_CELSIUS_MAX:g} °C. "
            "Backend tự đổi sang Kelvin bằng phép cộng 273,15 "
            "(K = °C + 273,15) vì dataset lưu nhiệt độ tuyệt đối. "
            "Ngoài khoảng này gần như chắc chắn là nhập nhầm đơn vị (ví dụ nhập Kelvin)."
        ),
    )
    rain_1h_mm: float = Field(
        default=0.0,
        ge=0.0,
        le=config.PRECIPITATION_MM_MAX,
        description="Lượng mưa trong 1 giờ (mm). Phải >= 0. Mặc định 0 = không mưa.",
    )
    snow_1h_mm: float = Field(
        default=0.0,
        ge=0.0,
        le=config.PRECIPITATION_MM_MAX,
        description="Lượng tuyết trong 1 giờ (mm). Phải >= 0. Mặc định 0 = không có tuyết.",
    )
    clouds_all: int = Field(
        ...,
        ge=0,
        le=100,
        description="Độ phủ mây (%). Phải là số nguyên trong khoảng 0..100.",
    )
    weather: list[WeatherCategory] = Field(
        ...,
        min_length=1,
        description=(
            "Các hiện tượng thời tiết xảy ra CÙNG LÚC trong giờ đó (multi-label/multi-hot). "
            f"Chỉ chấp nhận: {', '.join(WEATHER_MAIN_CATEGORIES)}."
        ),
        json_schema_extra={
            "example": ["Rain", "Snow"],
        },
    )
    state_fair_start_date: date | None = Field(
        default=None,
        description=(
            "TUỲ CHỌN. Ngày khai mạc Minnesota State Fair — chỉ cần nhập khi dự báo cho "
            "một năm nằm ngoài bảng lịch 2012–2020 của src/holidays.py. "
            "Nếu bỏ trống và năm nằm ngoài bảng lịch, API KHÔNG đoán: nó trả về cảnh báo "
            "`state_fair_calendar_unknown`."
        ),
    )

    @field_validator("date_time")
    @classmethod
    def _check_year(cls, value: datetime) -> datetime:
        if not (config.DATE_YEAR_MIN <= value.year <= config.DATE_YEAR_MAX):
            raise ValueError(
                f"Năm {value.year} nằm ngoài phạm vi hợp lệ "
                f"({config.DATE_YEAR_MIN}..{config.DATE_YEAR_MAX}). "
                "Kiểm tra lại định dạng date_time (ISO-8601, vd '2018-06-15T08:00:00')."
            )
        return value

    @field_validator("state_fair_start_date")
    @classmethod
    def _check_fair_same_year(cls, value: date | None, info: Any) -> date | None:
        date_time = info.data.get("date_time")
        if value is not None and isinstance(date_time, datetime) and value.year != date_time.year:
            raise ValueError(
                f"state_fair_start_date ({value.isoformat()}) phải cùng năm với "
                f"date_time ({date_time.year})."
            )
        return value

    @property
    def weather_unique(self) -> list[str]:
        """Danh sách hiện tượng đã khử trùng lặp, giữ thứ tự alphabet (tất định)."""
        return sorted(set(self.weather))


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class ModelRef(BaseModel):
    name: str
    version: str | None = None
    alpha: float | None = None
    solver: str | None = None
    target: str
    unit: str = config.TARGET_UNIT_EN
    unit_vi: str = config.TARGET_UNIT_VI
    n_features_out: int | None = None
    artifact: str = "models/ridge_pipeline.joblib"


class ForecastResponse(BaseModel):
    """Kết quả dự báo + thông tin trung thực về việc backend đã làm gì.

    Phân biệt bắt buộc:
      - RAW MODEL          : Ridge trả về trực tiếp -> `predictor.raw_model_output`
      - DEPLOYED PREDICTOR : `max(0, ·)` ∘ Ridge    -> `predicted_traffic_volume`
    Hai giá trị này **không phải cùng một metric của cùng một thứ** và không được
    gọi chung tên.
    """

    predicted_traffic_volume: float = Field(
        ..., description="Giá trị của DEPLOYED PREDICTOR, đơn vị xe/giờ. Luôn hữu hạn và >= 0."
    )
    unit: str = config.TARGET_UNIT_EN
    unit_vi: str = config.TARGET_UNIT_VI
    predictor: dict[str, Any] = Field(
        default_factory=dict,
        description="Tách rõ RAW MODEL (Ridge) và DEPLOYED PREDICTOR (max(0,·) ∘ Ridge).",
    )
    raw_model_output: float = Field(
        ..., description="RAW MODEL: giá trị thô trả về từ Ridge, TRƯỚC hậu xử lý."
    )
    clipped_to_zero: bool = Field(
        ..., description="True nếu giá trị thô < 0 và đã được chiếu về 0."
    )
    postprocess_policy: str = config.NON_NEGATIVE_POLICY_ID
    serving_policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Chính sách đã đóng băng: căn cứ, tập dùng để chọn, và việc KHÔNG dùng 2018.",
    )
    model: ModelRef
    input_echo: dict[str, Any]
    derived_features: dict[str, Any] = Field(
        ..., description="Feature đã kỹ thuật hoá — dùng CHUNG hàm với lúc huấn luyện."
    )
    calendar: dict[str, Any] = Field(
        ..., description="Đặc trưng lịch backend tự suy ra từ `date_time` (không cần người dùng nhập)."
    )
    holiday: dict[str, Any]
    weather: dict[str, Any]
    scope: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    model_loaded: bool
    model: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "ErrorResponse",
    "ForecastRequest",
    "ForecastResponse",
    "HealthResponse",
    "ModelRef",
    "WeatherCategory",
]

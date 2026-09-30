"""
src/weather.py

Định nghĩa DETERMINISTIC về thời tiết, dùng nhất quán cho TRAIN / VALIDATION / TEST /
API / WEB. Ở đây KHÔNG có bất kỳ thống kê nào được "học" từ dữ liệu — mọi ánh xạ là
hằng số do analyst định nghĩa, nên không thể gây train-serving skew.

Bối cảnh: 5445 timestamp có 2–6 bản ghi thời tiết khác nhau (một giờ có thể có
nhiều hiện tượng / nhiều mức mô tả). Vì vậy KHÔNG được chọn "đại dòng đầu tiên";
phải bảo toàn toàn bộ thông tin dưới dạng multi-label / multi-hot + một nhãn đại
diện có quy tắc rõ ràng.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 1. Danh mục weather_main (11 giá trị, thứ tự alphabet để ổn định)
# ---------------------------------------------------------------------------
WEATHER_MAIN_CATEGORIES: list[str] = [
    "Clear",
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

# ---------------------------------------------------------------------------
# 2. Thang mức độ (severity) cho weather_main
#
# Đây là thang thứ tự DO ANALYST ĐỊNH NGHĨA, không học từ dữ liệu và không dùng
# kỷ lục khí hậu thế giới làm rule. Mục đích: gộp nhiều mô tả thời tiết của cùng
# một giờ thành một số duy nhất theo quy tắc "lấy hiện tượng nghiêm trọng nhất".
#   0 = trời quang
#   1 = nhiều mây / giảm tầm nhìn nhẹ
#   2 = mưa phùn, sương mù
#   3 = mưa / tuyết rõ rệt
#   4 = giông / bão (nặng nhất)
# ---------------------------------------------------------------------------
WEATHER_MAIN_SEVERITY: dict[str, int] = {
    "Clear": 0,
    "Clouds": 1,
    "Haze": 1,
    "Drizzle": 2,
    "Fog": 2,
    "Mist": 2,
    "Rain": 3,
    "Snow": 3,
    "Smoke": 3,
    "Squall": 4,
    "Thunderstorm": 4,
}

# ---------------------------------------------------------------------------
# 3. Nén weather_description (38 chuỗi, có lẫn hoa/thường: "sky is clear" vs
#    "Sky is Clear") thành 11 nhóm gia đình thời tiết.
#
# Quy tắc là keyword, thứ tự ưu tiên từ trên xuống, khớp đầu tiên thắng
# (deterministic, không phụ thuộc thứ tự dòng trong file).
# ---------------------------------------------------------------------------
_WEATHER_FAMILY_RULES: list[tuple[str, str]] = [
    ("thunder", "thunder"),            # thunderstorm, bão tố
    ("squall", "thunder"),             # SQUALLS (OpenWeather dùng từ này cho bão tố)
    # -- hỗn hợp mưa/tuyết: phải kiểm TRƯỚC "snow" và "rain" --
    ("rain and snow", "mixed_precip"),
    ("shower snow", "mixed_precip"),
    ("sleet", "mixed_precip"),
    ("freezing", "mixed_precip"),
    # --
    ("snow", "snow"),
    ("thunderstorm", "thunder"),
    ("drizzle", "rain"),
    ("shower", "rain"),
    ("rain", "rain"),                  # rain / mọi mức mưa
    ("fog", "fog"),
    ("mist", "mist"),
    ("haze", "haze_smoke"),
    ("smoke", "haze_smoke"),
    ("overcast", "overcast"),
    ("cloud", "partly_cloudy"),        # broken / scattered / few clouds
    ("clear", "clear"),
]

WEATHER_FAMILIES: list[str] = [
    "clear",
    "partly_cloudy",
    "overcast",
    "haze_smoke",
    "mist",
    "fog",
    "rain",
    "snow",
    "mixed_precip",
    "thunder",
    "other",
]

# Thứ tự nghiêm trọng của gia đình thời tiết (dùng để gộp nhiều mô tả/giờ).
WEATHER_FAMILY_SEVERITY: dict[str, int] = {
    "clear": 0,
    "partly_cloudy": 1,
    "overcast": 2,
    "haze_smoke": 3,
    "mist": 4,
    "fog": 5,
    "rain": 6,
    "snow": 7,
    "mixed_precip": 8,
    "thunder": 9,
    "other": 2,
}


def _slug(text: str) -> str:
    """'Thunderstorm' -> 'thunderstorm' (dùng cho tên cột nhị phân)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def weather_main_columns() -> list[str]:
    """Tên cột multi-hot (multi-label) cho từng weather_main."""
    return [f"wm_{_slug(c)}" for c in WEATHER_MAIN_CATEGORIES]


def classify_weather_description(description: str) -> str:
    """Ánh xạ 1 chuỗi weather_description -> 1 nhóm gia đình thời tiết.

    Không phải học từ dữ liệu: chỉ là quy tắc keyword cố định.
    """
    if description is None:
        return "other"
    text = str(description).strip().lower()
    if not text:
        return "other"
    for keyword, family in _WEATHER_FAMILY_RULES:
        if keyword in text:
            return family
    return "other"


def severity_of_weather_main(value: str) -> int:
    return WEATHER_MAIN_SEVERITY.get(value, WEATHER_FAMILY_SEVERITY["other"])


def severity_of_weather_family(value: str) -> int:
    return WEATHER_FAMILY_SEVERITY.get(value, WEATHER_FAMILY_SEVERITY["other"])


def deterministic_mode(values: list[str]) -> str:
    """Nhãn đại diện deterministic cho một nhóm nhiều giá trị.

    Quy tắc: giá trị xuất hiện NHIỀU NHẤT; nếu hoàn tất thì chọn theo thứ tự
    alphabet (A→Z) để kết quả không phụ thuộc thứ tự dòng trong file.
    KHÔNG bao giờ dùng "dòng đầu tiên".
    """
    uniq = sorted({str(v) for v in values})
    if not uniq:
        return ""
    counts = {v: sum(1 for x in values if str(x) == v) for v in uniq}
    best = max(counts.values())
    return min(v for v, c in counts.items() if c == best)


def aggregate_weather_group(values: list[str]) -> dict:
    """Gộp nhiều bản ghi weather_main của CÙNG một timestamp thành biểu diễn
    không mất thông tin:
      - mode        : nhãn đại diện deterministic (dùng làm categorical feature)
      - set         : chuỗi "; ".join(sorted(set)) — phục vụ báo cáo / API / Web
      - severity    : max theo thang severity đã định nghĩa
      - wm_*        : multi-hot, phát hiện bất kỳ hiện tượng nào trong giờ đó
    """
    values = [str(v) for v in values]
    uniq = sorted(set(values))
    out = {
        "weather_main_mode": deterministic_mode(values),
        "weather_main_set": "; ".join(uniq),
        "weather_main_n": len(uniq),
        "weather_severity": max(severity_of_weather_main(v) for v in values),
    }
    for cat, col in zip(WEATHER_MAIN_CATEGORIES, weather_main_columns()):
        out[col] = int(cat in uniq)
    return out


def aggregate_weather_description_group(values: list[str]) -> dict:
    """Gộp weather_description theo cùng nguyên tắc deterministic."""
    values = [str(v) for v in values]
    uniq = sorted(set(values))
    families = sorted({classify_weather_description(v) for v in values})
    return {
        "weather_desc_mode": deterministic_mode(values),
        "weather_desc_set": "; ".join(uniq),
        "weather_desc_n": len(uniq),
        # Nhãn gia đình = gia đình nghiêm trọng nhất trong giờ đó
        "weather_family": max(families, key=lambda f: (severity_of_weather_family(f), f)),
    }

"""
src/holidays.py

LỊCH NGÀY LỄ TẤT ĐỊNH — tính được từ NGÀY THÁNG, KHÔNG cần đọc bất kỳ dòng dữ liệu nào.

Mục đích
--------
Feature `is_holiday` phải tái tạo được ở thời điểm dự báo (serving time) mà KHÔNG được
phụ thuộc vào việc quét các dòng khác trong dataset. Cách cũ (nhóm theo ngày rồi xem
cột `holiday` có khác "None" hay không) chỉ chạy được khi có sẵn toàn bộ bảng dữ liệu —
đó là train-serving skew, và còn tệ hơn là có thể rò rỉ thông tin của các năm về sau.

Lịch này là **thông tin lịch công cộng, biết trước tại thời điểm dự báo**: ai cũng biết
ngày 4/7 là lễ trước khi nó tới. Vì vậy dùng nó làm feature là hợp lệ.

Cách tính
---------
- 10/11 ngày lễ liên bang tính bằng quy tắc lịch (ngày trong tuần thứ n, hoặc ngày cố
  định có dời sang cuối tuần theo quy ước "observed day").
- `State Fair` (Hội chợ bang Minnesota) không có quy tắc lịch — ngày khai mạc do bang công
  bố từng năm — nên dùng bảng ngày công bố ghi sẵn.
- Với năm ngoài phạm vi bảng, hàm trả về `None` (không phải lễ) và người gọi được cảnh
  báo qua `is_year_known()`.

KHÔNG có thống kê nào được học từ dữ liệu ở module này.
"""
from __future__ import annotations

import calendar as _calendar
from datetime import date, timedelta

HOLIDAY_NONE = "None"

#: Ngày khai mạc Minnesota State Fair (nguồn: lịch công bố của bang Minnesota).
#: Đây là lịch công bố, KHÔNG phải con số bóc từ dữ liệu mục tiêu.
MINNESOTA_STATE_FAIR_START: dict[int, date] = {
    2012: date(2012, 8, 23),
    2013: date(2013, 8, 22),
    2014: date(2014, 8, 21),
    2015: date(2015, 8, 27),
    2016: date(2016, 8, 25),
    2017: date(2017, 8, 24),
    2018: date(2018, 8, 23),
    2019: date(2019, 8, 22),
    2020: date(2020, 8, 21),  # hội chợ bị huỷ do COVID-19 — vẫn ghi ngày lịch
}

KNOWN_YEARS: frozenset[int] = frozenset(MINNESOTA_STATE_FAIR_START)

# MONDAY=0 ... SUNDAY=6
_MON, _TUE, _THU, _FRI, _SAT, _SUN = 0, 1, 3, 4, 5, 6


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Ngày thứ `n` của `weekday` trong tháng (vd: Monday thứ 3 của tháng 1)."""
    if n < 1:
        raise ValueError("n phải >= 1")
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = first + timedelta(days=offset + 7 * (n - 1))
    if day.month != month:
        raise ValueError(f"Không có {weekday} thứ {n} trong tháng {month}/{year}")
    return day


def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Ngày `weekday` cuối cùng của tháng (vd: Monday cuối tháng 5 = Memorial Day)."""
    last_day = date(year, month, _calendar.monthrange(year, month)[1])
    return last_day - timedelta(days=(last_day.weekday() - weekday) % 7)


def observed_date(d: date) -> date:
    """Quy ước "ngày nghỉ được quan sát" cho lễ liên bang.

    Rơi vào thứ Bảy -> dời lên thứ Sáu trước; rơi vào Chủ nhật -> dời sang thứ Hai sau.
    (Không áp dụng cho Lễ Tạ Ơn và Columbus Day — hai ngày này luôn rơi vào đúng thứ.)
    """
    if d.weekday() == _SAT:
        return d - timedelta(days=1)
    if d.weekday() == _SUN:
        return d + timedelta(days=1)
    return d


def holiday_calendar_for_year(year: int) -> dict[date, str]:
    """Trả về {ngày: tên lễ} cho một năm. Hàm thuần của `year`."""
    cal: dict[date, str] = {
        observed_date(date(year, 1, 1)): "New Years Day",
        nth_weekday_of_month(year, 1, _MON, 3): "Martin Luther King Jr Day",
        nth_weekday_of_month(year, 2, _MON, 3): "Washingtons Birthday",
        last_weekday_of_month(year, 5, _MON): "Memorial Day",
        observed_date(date(year, 7, 4)): "Independence Day",
        nth_weekday_of_month(year, 9, _MON, 1): "Labor Day",
        nth_weekday_of_month(year, 10, _MON, 2): "Columbus Day",
        observed_date(date(year, 11, 11)): "Veterans Day",
        nth_weekday_of_month(year, 11, _THU, 4): "Thanksgiving Day",
        observed_date(date(year, 12, 25)): "Christmas Day",
    }
    fair = MINNESOTA_STATE_FAIR_START.get(year)
    if fair is not None:
        cal[fair] = "State Fair"
    return cal


def is_year_known(year: int) -> bool:
    """Năm có nằm trong bảng State Fair hay không (10 lễ còn lại luôn tính được)."""
    return year in KNOWN_YEARS


def holiday_name(day: date) -> str:
    """Tên ngày lễ của một ngày, hoặc "None" nếu không phải lễ.

    ⚠️ KHÔNG đụng vào dataset. Đây là toàn bộ điều Web/API cần để tái tạo `is_holiday`.
    """
    return holiday_calendar_for_year(day.year).get(day, HOLIDAY_NONE)


def is_holiday(day: date) -> int:
    """1 nếu `day` là ngày lễ, 0 nếu không. Tính thuần từ ngày tháng."""
    return int(holiday_name(day) != HOLIDAY_NONE)


def is_holiday_series(dates) -> list[int]:
    """Vector hoá cho một chuỗi `datetime.date`."""
    return [is_holiday(d) for d in dates]


def known_holiday_names() -> list[str]:
    return sorted(
        {name for year in KNOWN_YEARS for name in holiday_calendar_for_year(year).values()}
    )


__all__ = [
    "HOLIDAY_NONE",
    "KNOWN_YEARS",
    "MINNESOTA_STATE_FAIR_START",
    "holiday_calendar_for_year",
    "holiday_name",
    "is_holiday",
    "is_holiday_series",
    "is_year_known",
    "known_holiday_names",
    "last_weekday_of_month",
    "nth_weekday_of_month",
    "observed_date",
]

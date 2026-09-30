# Báo cáo chất lượng dữ liệu — Metro Interstate Traffic Volume

> Sinh tự động bởi `python src/data.py`. Mọi con số dưới đây được đo trên file thật.

## 1. Tổng quan
- Raw rows đọc được: **48.204**
- Rows sau khi loại `date_time` không hợp lệ: **48.204**
- Rows sau khi collapse trùng `date_time`: **40.575**
- Số dòng bị loại bởi collapse: **7.629**

## 2. Trùng lặp theo `date_time` — nguyên nhân và cách xử lý
- Số timestamp duy nhất: **40.575**
- Số nhóm trùng (>= 2 bản ghi): **5.445**
- Số dòng nằm trong các nhóm trùng: **13.074** (27.12%)
- Kích thước nhóm trùng: từ 2 đến 6 bản ghi/timestamp

### Kiểm tra tính bất biến trong từng nhóm trùng

| Cột | Nhóm bất biến | Nhóm khác nhau | Xử lý |
| --- | --- | --- | --- |
| `holiday` | 5445/5445 | 0 | bất biến -> assert rồi lấy giá trị |
| `temp` | 5367/5445 | 78 | nhiều phép đo -> **median** |
| `rain_1h` | 5436/5445 | 9 | nhiều phép đo -> **median** |
| `snow_1h` | 5445/5445 | 0 | bất biến -> assert rồi lấy giá trị |
| `clouds_all` | 5413/5445 | 32 | nhiều phép đo -> **median** |
| `weather_main` | 96/5445 | 5349 | nhiều hiện tượng -> **multi-hot** + nhãn đại diện mode→alphabet |
| `weather_description` | 59/5445 | 5386 | nhiều mô tả -> **multi-label** + nhãn gia đình thời tiết |
| `traffic_volume` | 5445/5445 | 0 | bất biến -> assert rồi lấy giá trị |

- Nguyên nhân trùng lặp: `weather_main` khác nhau trong **5.349**/5445 nhóm trùng. `weather_description` khác nhau trong **5.386** nhóm — tức một giờ được ghi nhận nhiều hiện tượng/mô tả thời tiết.
- Vì `traffic_volume`, `holiday`, `snow_1h` bất biến trong **100%** nhóm trùng, việc gộp KHÔNG làm mất hay bóp méo thông tin lưu lượng hay ngày lễ.
- ❌ KHÔNG dùng `drop_duplicates(subset=['date_time'], keep='first')`: cách đó vứt bỏ mọi hiện tượng thời tiết ngoài dòng đầu và phụ thuộc thứ tự dòng trong file (không deterministic về mặt nội dung).
- ✅ Dùng: median cho phép đo số, multi-hot/multi-label cho thời tiết, nhãn đại diện theo quy tắc (giá trị phổ biến nhất, hoà thì alphabet) — không phụ thuộc thứ tự dòng.

## 3. Ngữ nghĩa cột `holiday`
- Đọc CSV bằng `keep_default_na=False` — bắt buộc, vì pandas mặc định biến chuỗi `"None"` thành NaN.
- Rows với `holiday == "None"` (KHÔNG phải ngày lễ): **40.522**
- Rows có tên ngày lễ thật: **53**
- Số ngày lễ (lịch) duy nhất: **53**
- Số tên ngày lễ khác nhau: **11**

| Tên ngày lễ | Số dòng (giờ 00:00 của ngày đó) |
| --- | --- |
| Christmas Day | 5 |
| Columbus Day | 5 |
| Independence Day | 5 |
| Labor Day | 5 |
| Martin Luther King Jr Day | 3 |
| Memorial Day | 5 |
| New Years Day | 5 |
| State Fair | 5 |
| Thanksgiving Day | 5 |
| Veterans Day | 5 |
| Washingtons Birthday | 5 |

- Trong file gốc, tên ngày lễ chỉ xuất hiện ở **giờ 00:00** của ngày lễ. Vì vậy cột này KHÔNG dùng được trực tiếp làm feature mà cũng không đủ tin cậy để tra cứu ở thời điểm dự báo.

### Đối chiếu với lịch tất định (`src/holidays.py`)

`src/holidays.py` tính ngày lễ **từ ngày tháng bằng quy tắc lịch**, không đọc dữ liệu: MLK = thứ Hai thứ 3 tháng 1, Memorial = thứ Hai cuối tháng 5, Labor = thứ Hai thứ nhất tháng 9, Columbus = thứ Hai thứ 2 tháng 10, Thanksgiving = thứ Năm thứ 4 tháng 11, các ngày cố định có dời cuối tuần theo quy ước *observed day*, và `State Fair` lấy từ bảng ngày khai mạc do bang Minnesota công bố.

| Kiểm tra | Kết quả |
| --- | --- |
| Số ngày lễ dataset ghi nhận | 53 |
| Lịch khớp đúng tên + đúng ngày | 53 |
| Ngày lễ dataset có mà lịch BỎ SÓT | **0** |
| Ngày lịch có nhưng dataset không ghi (thiếu dòng 00:00) | 3 |

- Kết luận: **Lịch tất định khớp hoàn toàn và bổ sung cho dataset**
- Các ngày "lịch có nhưng dataset không ghi" là do dataset **thiếu hẳn dòng giờ 00:00** trong ngày đó, nên cột `holiday` không kịp ghi tên. Lịch tất định vẫn đánh dấu đúng — tức là lịch chính xác HƠN chính cột dữ liệu.
- Vì vậy `is_holiday` được tính từ lịch này, **không** nhóm theo ngày trên bảng dữ liệu. Đây là thông tin lịch công cộng, biết trước tại thời điểm dự báo, nên dùng làm feature là hợp lệ; đồng thời Web/API chỉ cần nhận một ngày là tái tạo được, không cần quét bất kỳ dòng nào khác.
- ❌ Cấm dùng `s.notna().any()` sau khi `"None"` đã là chuỗi — như vậy mọi ngày đều bị coi là ngày lễ.

## 4. Giá trị đo vô lý (quy tắc tất định -> NaN, KHÔNG nội suy)

### 4a. `temp` — quy tắc DỰA TRÊN VẬT LÝ
- Quy tắc: `temp <= 0.0 K -> NaN (quy tắc vật lý: 0 K là nhiệt độ tuyệt đối)`
- Số dòng bị loại: **10**
- 0 K là nhiệt độ tuyệt đối — không tồn tại ngoài trời và không thể đo được. Đây là ngưỡng có căn cứ vật lý, độc lập với bất kỳ bộ dữ liệu cụ thể nào, nên áp dụng được một cách nguyên tắc cho mọi tập dữ liệu khác.

### 4b. `rain_1h` — khớp CHÍNH XÁC giá trị lỗi, KHÔNG dùng ngưỡng
- Quy tắc: `rain_1h ∈ [9831.3] -> NaN (khớp chính xác giá trị lỗi đã audit trong bản phát hành này; KHÔNG dùng ngưỡng)`
- Số dòng bị loại: **1**
- ❌ **Không dùng ngưỡng kiểu `rain_1h > 100`.** Một ngưỡng rút ra từ "giá trị lớn nhất còn lại sau khi lọc" (55,63 mm) là ngưỡng **hậu nghiệm**: nó được chọn bằng cách nhìn toàn bộ tập dữ liệu, nên không khái quát và sẽ xoá nhầm các phép đo hợp lệ trên bất kỳ tập dữ liệu hoặc miền nào khác. Tương tự, cũng không dùng kỷ lục mưa thế giới làm rule cho mô hình.
- ✅ Dùng **khớp chính xác** các giá trị đã audit là lỗi nhập liệu trong đúng bản phát hành UCI này (9831,3 mm ≈ 386 inch — dấu hiệu nhập nhầm đơn vị). Khớp chính xác không bao giờ xoá một phép đo hợp lệ, bất kể phân bố dữ liệu thế nào.
- Hệ quả trung thực: nếu sau này gặp một giá trị mưa lớn bất thường **khác** 9831,3, quy tắc này sẽ không bắt được. Đó là đánh đổi có ý thức giữa "không xoá nhầm dữ liệu" và "bắt được mọi ngoại lệ" — và trong bài toán này, xoá nhầm còn tệ hơn bỏ sót.
- Sau khi đánh dấu: `temp` còn 10 NaN, `rain_1h` còn 1 NaN
- ❌ KHÔNG chạy `interpolate(method='time', limit_direction='both')` trên toàn bộ dữ liệu: nội suy trước khi tách tập là rò rỉ thống kê từ validation/test vào train. NaN được xử lý bởi `SimpleImputer` nằm trong sklearn pipeline, **fit chỉ trên TRAIN**.

## 5. Khoảng trống theo giờ & độ phủ thời gian (chỉ mang tính thông tin)
- Khoảng thời gian: 2012-10-02 09:00:00 → 2018-09-30 23:00:00
- Số giờ lẽ ra phải có nếu liên tục tuyệt đối: 52.551
- Số giờ có thật trong dữ liệu: 40.575
- Số giờ bị THIẾU hẳn dòng: 11.976 (22.79%)
- 1 timestamp = 1 quan sát sau collapse. Nếu sau này tạo lag feature, phải reindex theo lưới giờ đầy đủ trước khi shift, nếu không `shift(1)` sẽ không còn là 'giờ trước'.

## 6. Phân bố sau xử lý (toàn bộ, mô tả dữ liệu)

| Cột | min | max | mean | n NaN |
| --- | --- | --- | --- | --- |
| `traffic_volume` | 0.00 | 7280.00 | 3290.65 | 0 |
| `temp` | 243.39 | 310.07 | 281.39 | 10 |
| `rain_1h` | 0.00 | 55.63 | 0.08 | 1 |
| `snow_1h` | 0.00 | 0.51 | 0.00 | 0 |
| `clouds_all` | 0.00 | 100.00 | 44.20 | 0 |
| `weather_severity` | 0.00 | 4.00 | 1.19 | 0 |

### Phân bố `holiday` sau collapse

| Giá trị | Số timestamp |
| --- | --- |
| None | 40.522 |
| Columbus Day | 5 |
| Veterans Day | 5 |
| Thanksgiving Day | 5 |
| Christmas Day | 5 |
| New Years Day | 5 |
| Washingtons Birthday | 5 |
| Memorial Day | 5 |
| Independence Day | 5 |
| State Fair | 5 |
| Labor Day | 5 |
| Martin Luther King Jr Day | 3 |

### `weather_main` sau collapse (multi-hot nên có thể >0 mã/cột)

| weather_main | Số timestamp có hiện tượng này |
| --- | --- |
| Clear | 13.371 |
| Clouds | 15.127 |
| Drizzle | 1.792 |
| Fog | 912 |
| Haze | 1.359 |
| Mist | 5.940 |
| Rain | 5.563 |
| Smoke | 20 |
| Snow | 2.795 |
| Squall | 4 |
| Thunderstorm | 1.010 |

### Nhãn đại diện `weather_main_mode` (dùng làm categorical feature)

| weather_main_mode | Số timestamp |
| --- | --- |
| Clouds | 15.120 |
| Clear | 13.371 |
| Mist | 3.618 |
| Rain | 3.106 |
| Drizzle | 1.765 |
| Snow | 1.317 |
| Haze | 1.210 |
| Fog | 816 |
| Thunderstorm | 249 |
| Smoke | 2 |
| Squall | 1 |

### Số bản ghi thời tiết gộp lại mỗi giờ

| n_weather_obs | Số timestamp |
| --- | --- |
| 1 | 35.130 |
| 2 | 3.617 |
| 3 | 1.513 |
| 4 | 276 |
| 5 | 37 |
| 6 | 2 |

- Kiểm tra chéo: rows trong TRAIN (≤2016) = 25.329

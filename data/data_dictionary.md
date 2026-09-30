# Data Dictionary — Metro Interstate Traffic Volume

Nguồn: bảng "Variables Table" chính thức trên trang UCI (đã đối chiếu thêm với nhiều notebook công khai dùng cùng dataset để xác nhận kiểu dữ liệu thực tế khi đọc bằng pandas).

| # | Tên biến | Vai trò | Kiểu (UCI) | Kiểu thực tế khi đọc CSV (pandas) | Đơn vị | Mô tả | Thời điểm có sẵn |
| - | --- | --- | --- | --- | --- | --- | --- |
| 1 | `holiday` | Feature | Categorical | object (string), phần lớn giá trị rỗng/None | — | Tên ngày lễ liên bang Mỹ + lễ hội tiểu bang Minnesota (Minnesota State Fair). Rỗng nếu không phải ngày lễ. | Biết trước (lịch) |
| 2 | `temp` | Feature | Continuous | float64 | Kelvin | Nhiệt độ trung bình trong giờ đó | Tại thời điểm dự báo (đầu vào thời tiết) |
| 3 | `rain_1h` | Feature | Continuous | float64 | mm | Lượng mưa trong 1 giờ | Tại thời điểm dự báo |
| 4 | `snow_1h` | Feature | Continuous | float64 | mm | Lượng tuyết trong 1 giờ | Tại thời điểm dự báo |
| 5 | `clouds_all` | Feature | Integer | int64 | % | Phần trăm mây che phủ | Tại thời điểm dự báo |
| 6 | `weather_main` | Feature | Categorical | object | — | Mô tả thời tiết ngắn (VD: Clouds, Rain, Clear...) | Tại thời điểm dự báo |
| 7 | `weather_description` | Feature | Categorical | object | — | Mô tả thời tiết chi tiết hơn (VD: "scattered clouds") | Tại thời điểm dự báo |
| 8 | `date_time` | Feature (dùng tạo đặc trưng lịch) | Date | object → cần `pd.to_datetime` | — | Giờ thu thập dữ liệu, theo local CST | Biết trước (lịch) |
| 9 | `traffic_volume` | **Target** | Integer | int64 | xe/giờ | Lưu lượng xe westbound tại trạm ATR 301 trong giờ đó | Đây là giá trị cần dự đoán — **không phải input** |

## Ghi chú quan trọng khi dùng dữ liệu này (đã được kiểm chứng bằng script, xem `src/data.py`)

1. **`holiday` chứa chuỗi `"None"`, KHÔNG phải giá trị rỗng/NaN.** Bắt buộc đọc bằng `pd.read_csv(..., keep_default_na=False, na_values=[""])`. Nếu đọc mặc định, pandas biến `"None"` thành NaN và ta mất phân biệt "không phải ngày lễ". Tệ hơn nữa, **không được** dùng `s.notna().any()` để suy ra ngày lễ, vì khi đó `"None"` là chuỗi hợp lệ và **mọi ngày** sẽ bị coi là ngày lễ. Cách đúng: so sánh với hằng số `!= "None"`.
2. **Tên ngày lễ chỉ xuất hiện ở giờ 00:00** của ngày đó, và dataset còn thiếu hẳn dòng 00:00 ở 3 ngày lễ nên cột này **không đủ tin cậy** để dùng làm feature hay tra cứu lúc dự báo. `src/holidays.py` tính ngày lễ từ ngày tháng bằng quy tắc lịch (không đọc dữ liệu): khớp 53/53 ngày lễ dataset, bỏ sót 0, và thêm 3 ngày mà dataset bỏ lỡ. Vì vậy `is_holiday` được tính từ lịch — đây là thông tin lịch công cộng, biết trước tại thời điểm dự báo, nên không phải rò rỉ.
3. **`date_time` trùng lặp là có thật, KHÔNG được `drop_duplicates(keep="first")`.** Đã audit: 5.445 timestamp có 2–6 bản ghi. Nguyên nhân là một giờ có nhiều hiện tượng/mô tả thời tiết. Trong mọi nhóm trùng, `traffic_volume`, `holiday`, `snow_1h` bất biến 100%; `temp`/`rain_1h`/`clouds_all` có thể khác; `weather_main`/`weather_description` khác nhau ở ~98% nhóm. Vì vậy: median cho phép đo số, multi-hot/multi-label cho thời tiết, nhãn đại diện theo quy tắc mode→alphabet. Chi tiết: `src/data.py::collapse_duplicates`.
4. **Giá trị đo vô lý**: 10 dòng `temp == 0` K (0 K là nhiệt độ tuyệt đối — quy tắc vật lý, độc lập dữ liệu) và 1 dòng `rain_1h == 9831.3` mm (khớp **chính xác** sentinel; **không** dùng ngưỡng rút ra từ max của tập dữ liệu vì đó là rule hậu nghiệm). Đánh dấu thành NaN, **không nội suy** (nội suy trước khi tách tập là rò rỉ thống kê từ validation/test vào train). NaN được `SimpleImputer` trong pipeline điền, và imputer chỉ fit trên TRAIN.
5. **Chuỗi có khoảng trống giờ** (11.976 giờ, 22,79%) — 2015 đặc biệt thưa. Nếu sau này tạo lag feature, phải reindex theo lưới giờ đầy đủ trước khi shift, nếu không `shift(1)` sẽ không còn là 'giờ trước'.
6. **`traffic_volume` không được dùng làm feature dưới bất kỳ hình thức nào** chưa shift về quá khứ — đây là điểm rò rỉ dễ nhầm nhất của đề.
7. **TEST 2018 chỉ phủ 9/12 tháng** (dữ liệu kết thúc 2018-09-30), nên kết luận về drift theo mùa không được suy rộng ra cả năm.
8. **2018 là FINAL TEST, không tham gia tuning hay model selection.** Mọi thí nghiệm phát triển (random vs time split, kiểm soát rò rỉ, rolling-origin drift) nằm trong `src/experiments.py` và chỉ dùng 2012–2017; module này có `assert_no_final_test_rows()` chặn ở mọi hàm. Chính sách hậu xử lý `max(0,·)` cũng được chốt ở `src/freeze_serving_policy.py` **trước** khi chạy `src/evaluate.py`, và chỉ dựa trên TRAIN + VALIDATION.

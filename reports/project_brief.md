# Project Brief — Dự báo lưu lượng giao thông I-94 với chia tập theo thời gian

## 1. Bối cảnh & câu hỏi nghiên cứu
Một dashboard giao thông muốn ước lượng lưu lượng I-94 (chiều westbound, trạm ATR 301) theo giờ từ lịch và thời tiết để hỗ trợ lập kế hoạch.

**Câu hỏi nghiên cứu:** Đánh giá ngẫu nhiên (random split) và đánh giá trên tương lai (time split) chênh lệch bao nhiêu, và mô hình có vượt baseline theo lịch (hour × day-of-week) không?

## 2. Đơn vị quan sát, đầu vào, đầu ra
- Đơn vị quan sát: 1 giờ tại trạm đo ATR 301.
- Đầu vào: `holiday`, các biến thời tiết (`temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`), và đặc trưng lịch tạo từ `date_time` (giờ, thứ, tháng, cuối tuần...).
- Đầu ra: `traffic_volume` (xe/giờ).
- Loại bài toán: hồi quy chuỗi thời gian.

## 3. Nguồn & giấy phép dữ liệu
Metro Interstate Traffic Volume, UCI ML Repository, CC BY 4.0. 48.204 giờ, 2012-10-02 → 2018-09-30. Chi tiết: `data/README.md`.

## 4. Thời điểm dự đoán & rủi ro rò rỉ dữ liệu
Tại thời điểm dự đoán, mọi input phải là thông tin **biết trước hoặc quan sát được tại đúng giờ đó** — không dùng thông tin từ tương lai. Rủi ro rò rỉ chính đã xác định trước khi động vào dữ liệu:
- Không random split khi báo cáo kết quả chính thức (chỉ dùng minh họa).
- Mọi scaler/encoder/imputer chỉ fit trên train.
- `holiday` gốc chỉ có giá trị ở giờ đầu ngày lễ — forward-fill theo ngày phải làm cẩn thận, không để lộ thông tin từ dòng sau ra dòng trước theo hướng ngược thời gian.
- Nếu làm lag feature (mở rộng): chỉ lấy từ quá khứ, shift trước khi drop missing.

## 5. Tiêu chí thành công tối thiểu
- Toàn bộ pipeline chạy lại được trên máy khác (README + requirements + script tải dữ liệu).
- Vượt hoặc giải thích rõ khi không vượt baseline lịch.
- Chứng minh việc chia tập/tiền xử lý không rò rỉ (bằng test + code review, không chỉ khẳng định suông).
- Ứng dụng web có API chạy được, có validation input.

## 6. Kế hoạch đánh giá
Chỉ số chính: MAE, RMSE, R² trên tập test cuối thời gian (không dùng test để chọn mô hình). Thêm: MAE theo giờ trong ngày, MAE theo ngày trong tuần. 4 thí nghiệm bắt buộc: (1) random vs time split, (2) baseline vs model, (3) drift theo năm/mùa, (4) lỗi vào ngày lễ/thời tiết cực đoan.

## 7. Timeline rút gọn
| Tuần | Việc chính |
| --- | --- |
| 1 | Brief, data README, schema, baseline plan *(tài liệu này)* |
| 2 | Tải/làm sạch, EDA trên train, split, baseline |
| 3 | Pipeline model chính, validation/CV, thí nghiệm vòng 1 |
| 4 | Hoàn tất thí nghiệm, test cuối, phân tích lỗi |
| 5 | API + web UI, tích hợp model, test input |
| 6 | Báo cáo, slide, rehearsal, tái lập trên máy sạch |

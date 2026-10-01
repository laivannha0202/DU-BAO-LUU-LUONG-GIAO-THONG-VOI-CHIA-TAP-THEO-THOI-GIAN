# Data README — Metro Interstate Traffic Volume

## Nguồn chính thức
- Trang dữ liệu: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
- Link tải trực tiếp (file nén .zip chứa .csv.gz, 395.9 KB): https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip
- DOI: https://doi.org/10.24432/C5X60B
- Người đóng góp: John Hogue (donated 5/6/2019)

## Giấy phép
CC BY 4.0 (Creative Commons Attribution 4.0 International). Được phép chia sẻ và chỉnh sửa cho bất kỳ mục đích nào, miễn ghi công nguồn.
https://creativecommons.org/licenses/by/4.0/legalcode

## Trích dẫn
Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X60B.

Đây là trích dẫn **chuẩn và duy nhất** của project — `reports/final_report.md` (§3.1 và §14.2)
dùng đúng tên dataset, tác giả, năm, DOI, nguồn UCI và giấy phép CC BY 4.0 như ở đây.

## Ngày tải
2026-09-30 (qua `src/download_data.py`, đã xác nhận SHA256 khớp giữa 2 lần tải độc lập)

## Tệp sử dụng & checksum
- Tệp gốc: `Metro_Interstate_Traffic_Volume.csv.gz` → giải nén thành `Metro_Interstate_Traffic_Volume.csv`
- SHA256: `749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e`

## Phạm vi dữ liệu (đúng như tài liệu gốc UCI)
| Thuộc tính | Giá trị |
| --- | --- |
| Trạm đo | MN DoT ATR station 301 |
| Hướng | **Westbound** (chỉ chiều Tây của I-94 — không đại diện chiều Đông hay toàn tuyến) |
| Vị trí | Giữa Minneapolis và St. Paul, Minnesota, Mỹ |
| Khoảng thời gian | 2012-10-02 09:00 → 2018-09-30 23:00 |
| Múi giờ | Local CST |
| Số dòng | 48.204 |
| Số biến đầu vào | 8 |
| Biến mục tiêu | `traffic_volume` (1 biến) |
| Missing values | UCI ghi "Không". Thực tế: cột `holiday` chứa **chuỗi `"None"`** (không phải NaN) ở 48.143 dòng — nghĩa là "không phải ngày lễ", KHÔNG phải dữ liệu thiếu. Vì vậy `src/data.py` đọc bằng `keep_default_na=False`. Xem `reports/figures/data_quality_report.md` |

## Cách tái tạo (vì không đưa file dữ liệu vào Git)

Cài môi trường trước: `py -m pip install -r requirements-lock.txt` (tái lập tuyệt đối)
hoặc `py -m pip install -r requirements.txt` (theo khoảng version).

1. Tải dữ liệu gốc: `py src\download_data.py` trên máy có Internet thường
   (môi trường sandbox bị giới hạn mạng), HOẶC tải thủ công từ link ở trên rồi giải nén.
2. Đặt file vào `data/raw/Metro_Interstate_Traffic_Volume.csv` và đối chiếu SHA256
   (xem mục *Tệp sử dụng & checksum*).
3. Chạy toàn bộ pipeline, **đúng thứ tự** (Windows):

   ```bat
   py src\download_data.py         :: tải dữ liệu gốc từ UCI (máy có Internet)
   py src\data.py                  :: audit + collapse trùng + đánh dấu giá trị vô lý -> data/processed/
   py src\eda.py                   :: EDA CHỈ trên TRAIN
   py src\train.py                 :: fit baseline + Ridge + preprocessing trên TRAIN, chọn alpha trên VALIDATION -> ĐÓNG BĂNG cấu hình
   py src\experiments.py           :: thí nghiệm phát triển, CHỈ trong 2012-2017 (có assert_no_final_test_rows() chặn 2018)
   py src\freeze_serving_policy.py :: ĐÓNG BĂNG chính sách max(0,·) — CHỈ dùng TRAIN + VALIDATION
   py src\evaluate.py              :: FINAL TEST 2018 — mở 2018 ra đánh giá SAU khi model/config/policy đã đóng băng
   py src\postprocess_audit.py     :: chỉ đo tác động của policy đã đóng băng lên 2018 — KHÔNG quyết định policy
   py src\uncertainty_audit.py     :: chỉ đo độ bất định / phân tích kết quả 2018 — không retrain, không tune
   py -m pytest tests\ -v
   ```

   Thứ tự này là bắt buộc vì **lý do học thuật**, không phải quy ước hình thức:

   | # | Bước | Vai trò |
   | --- | --- | --- |
   | 1–3 | `download_data` → `data` → `eda` | Mọi biến đổi là quy tắc tất định; EDA chỉ trên TRAIN |
   | 4 | `train` | Fit baseline + Ridge + bộ tiền xử lý trên TRAIN, chọn `alpha` trên VALIDATION |
   | 5 | `experiments` | Chỉ dùng development window 2012–2017; **không** dùng FINAL TEST 2018 để chọn mô hình |
   | 6 | `freeze_serving_policy` | **Đóng băng** policy `max(0,·)` chỉ từ TRAIN + VALIDATION — phải chạy TRƯỚC `evaluate.py` |
   | 7 | `evaluate` | Đánh giá FINAL TEST 2018 sau khi model, cấu hình và serving policy đã đóng băng |
   | 8 | `postprocess_audit` | Chỉ đo tác động của policy đã đóng băng lên metric 2018; không quyết định policy |
   | 9 | `uncertainty_audit` | Chỉ đo độ bất định (bootstrap theo ngày, MAE theo tháng) và phân tích kết quả |
   | 10 | `pytest` | Kiểm chứng lại toàn bộ |

   Nếu đảo bước 6 và 7 (chạy `evaluate` trước `freeze_serving_policy`) thì chính sách hậu xử lý
   sẽ thành *test-informed postprocessing* — tức chọn cách hậu xử lý *vì* đã nhìn thấy kết quả 2018.
   Vì vậy `postprocess_audit.py` **từ chối chạy** nếu policy chưa được đóng băng.

### Về FINAL TEST 2018

Năm 2018 không tham gia hyperparameter tuning hoặc model selection.
Mô hình, cấu hình và serving policy được đóng băng từ dữ liệu 2012–2017
trước khi đánh giá FINAL TEST. Kết quả 2018 không được dùng để tiếp tục
tối ưu mô hình.

Điều cần chứng minh không phải là “số lần chạy” mà là **không có vòng lặp tối ưu nào đi qua
2018** — và điều đó được cưỡng chế bằng `assert_no_final_test_rows()` trong toàn bộ mã nguồn
thí nghiệm phát triển.

## Kết quả audit đã xác minh trên đúng file này
| Mục | Số liệu |
| --- | --- |
| Raw rows | 48.204 |
| Timestamp duy nhất | 40.575 |
| Nhóm trùng `date_time` | 5.445 (kích thước 2–6 bản ghi) |
| Rows nằm trong nhóm trùng | 13.074 (27,12%) |
| Rows sau khi collapse | **40.575** |
| `holiday == "None"` (KHÔNG phải lễ) | 40.522 timestamp sau collapse |
| Ngày lễ thật | 53 ngày, 11 tên khác nhau |
| `temp <= 0 K` (vô lý) | 10 dòng → NaN |
| `rain_1h == 9831.3` (khớp chính xác sentinel) | 1 dòng → NaN — **không dùng ngưỡng** |
| Lịch tất định (`src/holidays.py`) khớp dataset | **53/53**, bỏ sót **0** |
| Giờ bị thiếu hẳn | 11.976 (22,79%) trong khoảng 2012-10-02 → 2018-09-30 |

Chi tiết đầy đủ (kèm bảng bất biến theo từng cột trong nhóm trùng): `reports/figures/data_quality_report.md`.

## Sử dụng có trách nhiệm
Không áp dụng cho điều khiển giao thông an toàn quan trọng. Dữ liệu một trạm, một hướng — không đại diện cho toàn thành phố hay cả hai chiều đường.

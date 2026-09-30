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
1. Chạy `python src/download_data.py` trên máy có Internet thường (bị giới hạn mạng như môi trường sandbox này), HOẶC tải thủ công từ link ở trên rồi giải nén.
2. Đặt file vào `data/raw/Metro_Interstate_Traffic_Volume.csv`.
3. Chạy toàn bộ pipeline, **đúng thứ tự**:
   ```bash
   python src/data.py          # audit + collapse trùng + đánh dấu giá trị vô lý -> data/processed/
   python src/eda.py           # EDA CHỈ trên TRAIN
   python src/train.py         # baseline + tune alpha trên VALIDATION -> ĐÓNG BĂNG cấu hình
   python src/experiments.py   # thí nghiệm phát triển, CHỈ trong 2012-2017 (có guard chặn 2018)
   python src/evaluate.py      # FINAL TEST 2018, đánh giá đúng 1 lần
   pytest tests/ -v
   ```

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

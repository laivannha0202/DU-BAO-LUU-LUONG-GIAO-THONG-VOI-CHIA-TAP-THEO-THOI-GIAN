# NHẬT KÝ DỰ ÁN

**Đề tài:** Dự báo lưu lượng giao thông I-94 (chiều westbound) — trạm ATR 301
**Môn học:** Data Mining / Machine Learning — Project 20, Bài 7 (rò rỉ dữ liệu & đánh giá trung thực)
**Cập nhật cuối:** bản nộp cuối cùng

---

## ⚠ CẦN NGƯỜI DÙNG ĐIỀN TRƯỚC KHI NỘP

> Bảng dưới đây đã điền sẵn **toàn bộ phần công việc và kết quả kỹ thuật** — phần này lấy từ
> lịch sử thật của dự án (mã nguồn, artifact trong `models/` và `reports/figures/`).
>
> Ba cột sau **cố ý để trống có marker**, vì đó là thông tin chỉ người thật mới có:
>
> | Cột | Marker cần điền |
> | --- | --- |
> | Người thực hiện | `[NGƯỜI THỰC HIỆN]` |
> | Thời gian thực tế | `[NGÀY]` |
> | Giờ ước lượng | `[GIỜ THỰC TẾ]` |
>
> **Không** bịa tên, **không** bịa số giờ, **không** bịa ngày làm. Nếu một tuần có hai người
> làm, ghi rõ từng người trong cùng một ô.

**Cách điền nhanh:** thay `[NGƯỜI THỰC HIỆN]`, `[NGÀY]`, `[GIỜ THỰC TẾ]` trong bảng bên dưới
bằng thông tin thật của nhóm. Tổng giờ nên khớp với phần báo cáo giờ công của đề.

---

## Nhật ký theo tuần

| Tuần | Người thực hiện | Thời gian | Giờ ước lượng | Công việc | Kết quả | Vấn đề / cách xử lý |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `[NGƯỜI THỰC HIỆN]` | `[NGÀY]` | `[GIỜ THỰC TẾ]` | Đọc đề; viết `reports/project_brief.md`; tải dataset từ UCI bằng `src/download_data.py`; viết `data/README.md` (nguồn, giấy phép CC BY 4.0, SHA256) và `data/data_dictionary.md` | Chốt 7 quy tắc chống rò rỉ **trước khi** động vào dữ liệu; xác minh 48.204 dòng thô, khung 2012-10-02 → 2018-09-30 | Dữ liệu có nhiều dòng trùng `date_time` ngay từ đầu → chưa xử lý ở tuần này, để nguyên cho tuần 2 audit kỹ |
| 2 | `[NGƯỜI THỰC HIỆN]` | `[NGÀY]` | `[GIỜ THỰC TẾ]` | `src/data.py` (audit + collapse trùng + đánh dấu giá trị vô lý); `src/holidays.py` (lịch tất định); `src/weather.py`; `src/features.py` (feature + time split có assert); `src/eda.py` | Còn **40.575** dòng mô hình hoá; lịch ngày lễ khớp **53/53** ngày dataset, bỏ sót 0; time split train 25.329 / val 8.713 / test 6.533 với 3 assert đều đạt | **`holiday == "None"` bị pandas đọc thành NaN** → dùng `keep_default_na=False`. **`drop_duplicates(keep="first")` bị loại** vì phụ thuộc thứ tự dòng và vứt bỏ thông tin thời tiết → thay bằng collapse tất định (median cho phép đo, multi-hot cho thời tiết). **10 dòng `temp ≤ 0 K` và 1 dòng `rain_1h = 9831,3`** → đánh dấu NaN bằng quy tắc vật lý / khớp sentinel, **không** dùng ngưỡng hậu nghiệm |
| 3 | `[NGƯỜI THỰC HIỆN]` | `[NGÀY]` | `[GIỜ THỰC TẾ]` | `src/train.py`: baseline mean theo `giờ × thứ` (fit train only) + Ridge pipeline (`ColumnTransformer` + `Pipeline`) + tune `alpha` trên validation | `alpha = 0,001`, `solver = lsqr`; validation MAE **272,12** so với baseline **278,66**; lưu `models/ridge_pipeline.joblib` + `model_metadata.json` + `run_config.json` | Hiệu năng gần như bằng nhau ở dải alpha nhỏ → chọn **biên nhỏ nhất** (ít giả định nhất) thay vì chọn theo sai số cộng thêm. Không mở 2018 ở bước này |
| 4 | `[NGƯỜI THỰC HIỆN]` | `[NGÀY]` | `[GIỜ THỰC TẾ]` | `src/experiments.py`: thí nghiệm 1 (random vs time split), 1b (tách kích thước train khỏi khoảng cách thời gian), 1c (khối liên tục), 3 (rolling-origin), 3b (PSI) — **chỉ trong 2012–2017** | Thí nghiệm 1: random split lạc quan **5,43 ± 0,81 MAE** khi so trên cùng tập dòng đánh giá. Thí nghiệm 1b/1c: sau khi thêm arm đối chứng **cùng kích thước**, kết luận cũ "mô hình nhìn thấy hàng xóm" **không còn đúng** — hiệu ứng không do kích thước, và mô hình không có lag nên không thể nhớ giá trị dòng lân cận; điều đo được là ảnh hưởng của khoảng cách thời gian (4,15 MAE) | Phải thêm `assert_no_final_test_rows()` ở **mọi** hàm sau khi phát hiện một lần code đọc cả năm 2018 trong lúc debug → giờ chặn ở mọi lối vào và có 31 test bảo vệ. Rolling-origin cho MAE giảm 53,86 từ 2015→2017 nhưng **không** kết luận "mô hình tiến bộ" vì có thể do 2017 dễ hơn. Bài học: **phải tách yếu tố trước khi gán cơ chế cho một con số** |
| 5 | `[NGƯỜI THỰC HIỆN]` | `[NGÀY]` | `[GIỜ THỰC TẾ]` | `src/freeze_serving_policy.py` (**đóng băng policy TRƯỚC khi mở 2018**) → `src/evaluate.py` (FINAL TEST) → `src/postprocess_audit.py` (đo tác động) | RAW MODEL: MAE **259,73** / RMSE **416,78** / R² **0,9554**; baseline 272,90 / 473,13 / 0,9426. DEPLOYED `max(0,·)`: 257,54 / 412,25 / 0,9564 — báo riêng, không gọi chung tên | Ridge là hồi quy tuyến tính nên **có thể trả dự báo âm**. Nguy cơ test-informed postprocessing → tách quyết định khỏi đo lường: policy chốt bằng 3 điều kiện D1–D3 trên TRAIN + VALIDATION, rồi mới chạy evaluate. `postprocess_audit.py` từ chối chạy nếu chưa có policy. Phân tích lỗi lộ ra điểm yếu thật: ngày lễ MAE 1.031 (gấp 4,3 lần), tuyết 524,50, sương mù 536,05 |
| 6 | `[NGƯỜI THỰC HIỆN]` | `[NGÀY]` | `[GIỜ THỰC TẾ]` | `app/` (FastAPI + 3 màn hình + 4 endpoint); `tests/` (bộ test đầy đủ); `reports/final_report.md`; `docs/slides-outline.md`, `demo-script.md`, `viva-questions.md`; `src/export_docs.py` + `requirements-export.txt`; `requirements-lock.txt` + `models/environment.json`; `docs/project-log.md` | **356 test pass, 0 fail**; web/API chạy thật (`/health` 200, dự báo trả số thật, input sai trả 422 và server không sập); xuất được `release/final_report.docx` / `.pdf` / `slides.pptx` | Rủi ro lớn nhất là **train-serving skew** → dùng chung hàm `build_features`, `is_holiday` tính bằng lịch tất định, và có test so từng cột feature trên dữ liệu thật. Số test bị ghi sai (197 / 238 / 265 / 294 lệch nhau) → thêm test tự đối chiếu `pytest --collect-only` để tài liệu **không thể** lệch số |

---

## Ghi chú

### Về cột "Người thực hiện" và "Giờ ước lượng"

Nhóm điền các cột này bằng thông tin thật. Repo **không** tự bịa tên hay số giờ — đó là
thông tin cá nhân và không thể suy ra từ mã nguồn.

### Về lịch sử Git

Lịch sử commit hiện tại thuộc **một tài khoản duy nhất**. Nhóm **không** tạo commit giả cho
thành viên thứ hai, không đổi tên tác giả, không backdate. Nếu đề yêu cầu mỗi thành viên có
phần đóng góp Git riêng, phần đó phải do **chính thành viên đó** tự commit.

### Mốc kỹ thuật tham chiếu

| Mốc | Nơi lưu |
| --- | --- |
| Số dòng / checksum dữ liệu gốc | `data/README.md`, `data/processed/data_audit.json` |
| Cấu hình train, seed, alpha | `models/run_config.json` |
| Số liệu FINAL TEST | `reports/figures/evaluation_results.json` |
| Số liệu thí nghiệm 2012–2017 | `reports/figures/experiments_results.json` |
| Chính sách hậu xử lý đã đóng băng | `models/serving_policy.json` |
| Tác động của policy lên 2018 | `reports/figures/postprocess_audit.json` |
| Môi trường đã dùng để sinh artifact | `models/environment.json` |

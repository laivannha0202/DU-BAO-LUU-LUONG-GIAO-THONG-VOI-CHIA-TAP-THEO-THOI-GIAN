# Dự báo lưu lượng giao thông I-94 (westbound) theo giờ

Project 20 — Bài 7: Rò rỉ dữ liệu, chia tập đúng và đánh giá trung thực.
Dữ liệu: Metro Interstate Traffic Volume (UCI, CC BY 4.0). Xem `data/README.md`.

## Trạng thái — Checkpoint 3.1 (ĐÃ CÓ WEB/API, sẵn sàng nghiệm thu)

- [x] Cấu trúc project
- [x] Dữ liệu thật trong `data/raw/` — SHA256 `749c90d7...85fce9e` khớp, 48.204 dòng
- [x] **Preprocessing** (`src/data.py` + `src/weather.py`)
  - `keep_default_na=False` → `holiday == "None"` nghĩa là **không phải ngày lễ**
  - **Không** `drop_duplicates(keep="first")`: 5.445 nhóm trùng `date_time` collapse tất định (median cho phép đo, multi-hot cho thời tiết) → 1 timestamp = 1 quan sát, còn **40.575** dòng
  - `traffic_volume` / `holiday` / `snow_1h` **assert bất biến 100%** trong nhóm trùng
  - `temp <= 0 K` (10 dòng) → NaN bằng quy tắc **vật lý** (0 K là nhiệt độ tuyệt đối)
  - `rain_1h` → NaN bằng **khớp chính xác sentinel 9831.3**, **không dùng ngưỡng hậu nghiệm**
  - **không nội suy**; NaN do `SimpleImputer` trong pipeline điền, fit TRAIN only
- [x] **`is_holiday` từ lịch tất định** (`src/holidays.py`) — tính từ ngày tháng, khớp 53/53 ngày lễ dataset, bỏ sót 0, Web/API tái tạo được không cần đọc dữ liệu
- [x] **Time split có assert**: train ≤2016 (25.329) / validation 2017 (8.713) / test 2018 (6.533)
- [x] EDA **chỉ trên TRAIN** (`src/eda.py`)
- [x] Baseline mean theo `hour × day_of_week`, fit train only
- [x] Ridge + `ColumnTransformer` + `Pipeline` với `SimpleImputer → StandardScaler`; imputer/scaler/encoder **fit train only**; alpha **tune trên validation**
- [x] **Độ nhạy alpha trên lưới mở rộng** (chỉ VALIDATION, có `alpha = 0` = OLS): vùng `alpha ∈ [0; 0,01]` chỉ chênh 0,09 MAE → Ridge ≈ OLS; giữ nguyên alpha đã đóng băng vì 2018 đã bị xem
- [x] **Bảo vệ FINAL TEST 2018** (`src/experiments.py`): mọi thí nghiệm phát triển chạy trong 2012–2017, có `assert_no_final_test_rows()` chặn ở mọi hàm
- [x] **Đo cải thiện 4,15 MAE do khoảng cách thời gian** — Thí nghiệm 1b/1c tách riêng *kích thước tập huấn luyện*, *mức năm* và *khoảng cách thời gian*; có arm đối chứng cùng kích thước, lặp 5 seed, báo trung bình ± độ lệch
- [x] **Drift công bằng**: rolling-origin chỉ trên cửa sổ out-of-sample; tách "performance by split" khỏi "drift"; thêm PSI cho drift phân bố
- [x] Final evaluation bundle duy nhất trên 2018 (`src/evaluate.py`): baseline vs model, MAE/RMSE/R², error analysis hour/day/holiday/weather
- [x] **FastAPI + 3 màn hình web** (`app/`) — chỉ load artifact, không train, không tune
- [x] **Model Card** (mục 10 của màn Dashboard + `reports/final_report.md`)
- [x] **Serving policy đã đóng băng** (`src/freeze_serving_policy.py` → `models/serving_policy.json`)
      — quyết định chỉ trên TRAIN + VALIDATION, có đường bỏ clamp nếu điều kiện không đạt
- [x] **Báo cáo tác động** của policy trên 2018 (`src/postprocess_audit.py`) — chạy SAU, chỉ đo
- [x] **Kiểm toán bất định** (`src/uncertainty_audit.py`) — block bootstrap theo khối ngày lịch (4.000 lần, seed cố định) cho hiệu MAE/RMSE, MAE theo từng tháng; chạy SAU `evaluate.py`, chỉ đo
- [x] **Kiểm chứng điểm yếu giờ đêm trên dữ liệu dev** (`src/experiments.py`) — Ridge kém ở toàn bộ 5 giờ đêm trong 3/3 cửa sổ rolling-origin; thử log-target như khám phá hậu nghiệm, **không** thay mô hình
- [x] **Thí nghiệm lag (độc lập, KHÔNG vào serving)** — lag 1/24/168 giờ ghép theo **thời điểm** (không `shift()` theo dòng); kết quả **không ủng hộ** giả thuyết "lag tự tạo rò rỉ", nhưng cách dựng lag sai thì có
- [x] **Test: `py -m pytest tests\ -v` → 317 passed, 0 failed**
      (con số này được **tự kiểm chứng** bởi `tests/test_report.py::test_documented_test_count_matches_real_collection`)
- [x] Báo cáo, slide, kịch bản demo, câu hỏi viva (`reports/final_report.md`, `docs/`)

---

## Chạy lại toàn bộ từ đầu (Windows)

```bat
py -m pip install -r requirements.txt
py src\download_data.py     :: chỉ cần 1 lần, trên máy có Internet
py src\data.py              :: audit + collapse + đánh dấu giá trị vô lý
py src\eda.py               :: EDA chỉ trên TRAIN
py src\train.py             :: baseline + tune alpha + lưu model  -> ĐÓNG BĂNG cấu hình
py src\experiments.py       :: thí nghiệm phát triển, CHỈ 2012-2017 (không đụng 2018)
py src\freeze_serving_policy.py :: ĐÓNG BĂNG chính sách max(0,·) — CHỈ dùng TRAIN + VALIDATION
py src\evaluate.py          :: FINAL TEST 2018 — mở 2018 ra đánh giá lần đầu
py src\postprocess_audit.py :: báo cáo tác động của policy trên 2018 — chạy SAU, KHÔNG quyết định gì
py src\uncertainty_audit.py :: bootstrap + MAE theo tháng trên 2018 — chạy SAU, CHỈ đo
py -m pytest tests\ -v
py -m uvicorn app.main:app --reload
```

Sau đó mở trình duyệt: **http://localhost:8000** (OpenAPI docs: http://localhost:8000/docs)

> **Thứ tự này là bắt buộc — và mang ý nghĩa học thuật, không phải quy ước hình thức.**
>
> | # | Bước | Vì sao đứng ở đây |
> | --- | --- | --- |
> | 1–3 | `download_data` → `data` → `eda` | Mọi biến đổi là **quy tắc tất định**, EDA chỉ trên TRAIN |
> | 4 | `train` | Fit mô hình + bộ tiền xử lý, chọn `alpha` trên VALIDATION |
> | 5 | `experiments` | Mọi thí nghiệm phát triển nằm trong 2012–2017 (có `assert_no_final_test_rows()`) |
> | 6 | **`freeze_serving_policy`** | **Đóng băng** chính sách `max(0,·)` **chỉ từ TRAIN + VALIDATION** |
> | 7 | **`evaluate`** | Mở FINAL TEST 2018 ra **sau khi** mọi lựa chọn đã chốt xong |
> | 8 | `postprocess_audit` | **Chỉ đo** tác động của policy đã đóng băng lên metric 2018 |
> | 9 | `uncertainty_audit` | **Chỉ đo** độ bất định quanh kết quả 2018 (bootstrap theo ngày, MAE theo tháng) |
> | 10–11 | `pytest` → `uvicorn` | Kiểm chứng rồi mới phục vụ |
>
> Nếu đảo bước 6 và 7 (chạy `evaluate` trước `freeze_serving_policy`) thì chính sách hậu xử lý
> sẽ trở thành **test-informed postprocessing** — tức ta chọn cách hậu xử lý *vì* nhìn thấy kết quả
> 2018, làm mất ý nghĩa của FINAL TEST. Vì vậy `postprocess_audit.py` còn **từ chối chạy** nếu
> policy chưa được đóng băng, và `tests/test_serving_policy.py` kiểm tra ở mức mã nguồn rằng
> `build_policy()` không bao giờ đọc dòng nào của năm 2018.
>
> Sau khi đọc kết quả 2018, **không** quay lại sửa mô hình, sửa `alpha`, hay sửa policy.
>
> Lưu ý PowerShell: dùng dấu gạch chéo ngược `\` cho đường dẫn script và `\ -v` cho pytest.

---

## Kết quả chính (FINAL TEST 2018)

| | MAE | RMSE | R² | n |
| --- | --- | --- | --- | --- |
| Baseline (hour × day_of_week) | 272,90 | 473,13 | 0,9426 | 6.533 |
| **Ridge pipeline** (alpha = 0,001) | **259,73** | **416,78** | **0,9554** | 6.533 |

Chi tiết: `reports/figures/evaluation_report.md` (2018), `reports/figures/experiments_report.md` (2012–2017),
`reports/figures/uncertainty_audit.md` (độ bất định), báo cáo đầy đủ: `reports/final_report.md`.

**Đọc kết quả này kèm điều kiện** — không được nói chung "mô hình luôn hơn baseline":

- Trên 2018, Ridge nhỉnh hơn baseline **13,16 MAE**; block bootstrap theo ngày lịch cho
  CI 95 % **[+3,80; +22,00]** (không chứa 0), Ridge thắng ở **8/9 tháng**.
- **Nhưng** trên dữ liệu dev 2012–2017, Ridge chỉ thắng ở **1/4 cửa sổ** (pseudo-test 2016–2017
  và 3 fold rolling-origin). Ở 3 cửa sổ kia, khoảng tin cậy nằm hẳn về phía baseline.

---

## Ứng dụng Web / API

### Khởi động

```bat
py -m uvicorn app.main:app --reload
```

| Đường dẫn | Nội dung |
| --- | --- |
| `http://localhost:8000/` | Màn 1 — Giới thiệu |
| `http://localhost:8000/du-bao` | Màn 2 — Dự báo (form gọi API thật) |
| `http://localhost:8000/dashboard` | Màn 3 — Dashboard + Model Card |
| `http://localhost:8000/docs` | OpenAPI docs (Swagger UI) |

### Endpoint

| Method | Path | Mô tả |
| --- | --- | --- |
| GET | `/health` | 200 khi model đã load; **503 + lý do rõ** khi thiếu artifact |
| GET | `/api/model-info` | Metadata mô hình, phạm vi dữ liệu, quy tắc validation, chính sách hậu xử lý |
| POST | `/api/traffic-forecast` | Dự báo `traffic_volume` (xe/giờ) cho một mốc thời gian |
| GET | `/api/dashboard-metrics` | Toàn bộ số liệu dashboard, đọc thẳng từ artifact |

### Ví dụ gọi API

```bat
curl -X POST http://localhost:8000/api/traffic-forecast ^
  -H "Content-Type: application/json" ^
  -d "{\"date_time\":\"2018-06-15T08:00:00\",\"temperature_celsius\":20.0,\"clouds_all\":20,\"weather\":[\"Clear\"]}"
```

```json
{
  "predicted_traffic_volume": 5525.76,
  "unit": "vehicles_per_hour",
  "unit_vi": "xe/giờ",
  "raw_model_output": 5525.7626,
  "clipped_to_zero": false,
  "postprocess_policy": "max(0, raw_prediction)",
  "calendar": { "hour": 8, "day_of_week_name": "Thứ 6", "month": 6, "is_weekend": false },
  "holiday": { "is_holiday": 0, "holiday_name": "None" },
  "weather": { "multi_hot": { "wm_clear": 1, "wm_rain": 0, "...": 0 } },
  "warnings": []
}
```

### Nguyên tắc bất di bất dịch của lớp serving

1. **Chỉ load artifact đã đóng băng** (`models/ridge_pipeline.joblib`, `models/model_metadata.json`).
2. **Không train, không tune alpha, không fit lại** encoder/imputer/scaler, **không đọc target** từ dataset.
3. Mọi feature kỹ thuật hoá sinh bằng **CHUNG hàm** `src.features.build_features` với lúc huấn luyện → **không có train-serving skew**
   (có test `test_no_train_serving_skew_on_real_rows` đối chiếu từng cột feature trên dữ liệu thật).
4. Backend tự suy ra: giờ, thứ, tháng, cuối tuần, `is_holiday` (lịch tất định), nhãn đại diện & multi-hot thời tiết, mức độ thời tiết.
   **Người dùng không cần (và không được phép) nhập `is_holiday` thủ công.**
5. Nhiệt độ nhập theo **°C**; backend tự quy đổi sang **Kelvin** (K = °C + 273,15).
6. Nhiều hiện tượng thời tiết cùng lúc được giữ đủ bằng **multi-hot** (không ép về một nhãn).
7. **Chính sách phục vụ đã đóng băng: `max(0, dự báo thô)`.**
   - Chốt bởi `src/freeze_serving_policy.py` → `models/serving_policy.json`,
     **chỉ dùng TRAIN + VALIDATION**. FINAL TEST 2018 **không** tham gia quyết định.
   - Ba điều kiện D1–D3 (miền giá trị · mô hình thực sự tràn xuống 0 · cắt không tệ hơn trên
     validation) — nếu một điều kiện FAIL thì script tự đặt `policy = "none"` và negative
     prediction thành limitation.
   - **Phân biệt bắt buộc:** **RAW MODEL** = Ridge trả về trực tiếp (MAE 259,73 — *metric của
     mô hình*, kết luận chính thức) · **DEPLOYED PREDICTOR** = `max(0,·)` ∘ Ridge
     (MAE 257,54 — *wrapper phục vụ*, báo riêng). **Không gọi hai số này là cùng một model metric.**
   - Xem `reports/figures/serving_policy.md` (§ quyết định) và
     `reports/figures/postprocess_audit.md` (tác động sau khi đóng băng).

### Validation

| Trường | Quy tắc | Sai thì trả |
| --- | --- | --- |
| `date_time` | ISO-8601, năm 1900–2200 | 422 |
| `temperature_celsius` | số, trong khoảng **−70 … 70 °C** | 422 |
| `rain_1h_mm` | số, **≥ 0**, ≤ 500 | 422 |
| `snow_1h_mm` | số, **≥ 0**, ≤ 500 | 422 |
| `clouds_all` | số nguyên, **0 … 100** | 422 |
| `weather` | danh sách **không rỗng**, chỉ 11 danh mục hỗ trợ | 422 kèm danh sách hợp lệ |
| `state_fair_start_date` (tuỳ chọn) | cùng năm với `date_time` | 422 |
| trường lạ | bị từ chối, **không bỏ qua âm thầm** | 422 |

Body lỗi luôn có cấu trúc:

```json
{
  "error": "validation_error",
  "message": "Input không hợp lệ: ...",
  "details": [{ "field": "clouds_all", "message": "...", "type": "less_than_equal" }]
}
```

Server **không crash** khi nhận input sai (đã có test dồn nhiều request lỗi rồi kiểm tra server vẫn phục vụ bình thường).

### Phạm vi sử dụng & cảnh báo trung thực

- Dữ liệu chỉ phủ **2012-10-02 09:00 → 2018-09-30 23:00**. Dự báo ngoài khoảng này: API vẫn trả kết quả
  nhưng kèm cảnh báo `in_dataset_range = false`.
- **FINAL TEST chỉ có 9 tháng đầu 2018.** Đo trên validation 2017 cho thấy chấm Jan–Sep cho MAE
  thấp hơn cả năm **21,16** (250,96 so với 272,12) — tức số 2018 được đo trên **phần dễ hơn**
  của năm. Mọi kết luận về 2018 chỉ áp dụng cho 9 tháng đầu năm.
- **Ngày lễ trong FINAL TEST chỉ có 7 ngày lịch** (167 giờ), và MAE từng ngày lệch nhau gần 3 lần
  (518,33 → 1.508,73) → con số "MAE ngày lễ" có độ bất định lớn, không đọc như giá trị ổn định.
- Bảng lịch **State Fair** chỉ có ngày cho các năm **2012–2020**. Ngoài khoảng đó API **không tự đoán**:
  trả cảnh báo `state_fair_calendar_unknown` và cho phép người dùng truyền
  `state_fair_start_date` để tính đúng.
- Mô hình tuyến tính, một trạm đo, dữ liệu lịch sử 2012–2018 — **không dùng cho mục đích safety-critical**.

---

## Deploy lên Render

Web Service public chạy trực tiếp artifact đã đóng băng. **Không train lại**, **không cần dataset**.

Repo có sẵn [`render.yaml`](render.yaml) (Render Blueprint), nên Render tự điền Build/Start/Health
từ file đó — không phải nhập tay.

- Python: **3.14.7** (Render đọc file [`.python-version`](.python-version); đây là đúng phiên bản
  đã sinh ra `models/ridge_pipeline.joblib` — xem `models/environment.json`).
- Dependency: [`requirements-deploy.txt`](requirements-deploy.txt) — chỉ phần phục vụ web/API,
  pin đúng version của môi trường sinh model (đổi version sklearn/numpy có thể làm hỏng artifact).

Build Command:

```
pip install -r requirements-deploy.txt
```

Start Command:

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Health Check:

```
/health
```

Các điểm cần biết:

- `PORT` do Render cấp — **không hard-code** trong `render.yaml`.
- `/health` trả **200** khi `models/ridge_pipeline.joblib` load được, **503 + lý do rõ ràng** khi thiếu artifact.
- Không cần `data/raw/` và `data/processed/traffic_clean.csv` (2 tệp này bị `.gitignore` loại khỏi repo).
  Các JSON báo cáo trong `reports/figures/` và `data/processed/data_audit.json` đã được commit nên
  Dashboard & Model Card hiển thị đầy đủ.
- Render dùng `uvicorn` một tiến trình, 1 worker — tải nhẹ, phù hợp demo/nghiệm thu.
- Nếu dùng gói **Free**, service ngủ sau ~15 phút không có truy cập; lần mở lại đầu tiên chậm vài giây.

Các bước tạo service (làm trực tiếp trên render.com):

1. Đăng nhập <https://render.com> (GitHub → Authorize).
2. **New + → Blueprint** → chọn repo `laivannha0202/DU-BAO-LUU-LUONG-GIAO-THONG-VOI-CHIA-TAP-THEO-THOI-GIAN`.
3. Render đọc `render.yaml`: tên `traffic-forecast-i94`, Runtime `python`, Region, Plan.
4. **Apply** → chờ Build & Deploy.
5. Deploy xong, Render hiển thị URL public dạng `https://<ten-service>.onrender.com` trên trang
   Service → **Copy URL**. Mở URL đó và kiểm tra `/health` trả 200.

> Địa chỉ public thật chỉ tồn tại **sau khi** Render tạo service; README này cố tình không ghi trước URL.

---

## Cấu trúc thư mục

```
data/
  README.md              # nguồn, giấy phép, checksum, kết quả audit
  data_dictionary.md      # mô tả từng biến + các bẫy đã xác minh
  raw/                    # CSV gốc (không commit — .gitignore)
  processed/
    traffic_clean.csv     # đã collapse + đánh dấu giá trị vô lý
    data_audit.json       # số liệu audit dạng máy đọc được
src/
  download_data.py        # tải dữ liệu (chạy trên máy có Internet)
  weather.py              # định nghĩa tất định về thời tiết (không học thống kê)
  holidays.py             # lịch ngày lễ tất định — tính từ ngày, KHÔNG đọc dữ liệu
  data.py                 # load, audit, collapse trùng, đánh dấu giá trị vô lý
  features.py             # feature engineering + time split có assert
  eda.py                  # EDA chỉ trên TRAIN
  train.py                # baseline + Ridge pipeline, tune alpha, lưu artifact
  experiments.py          # thí nghiệm phát triển — CHỈ 2012-2017, có guard 2018
  freeze_serving_policy.py # ĐÓNG BĂNG chính sách max(0,·) — chỉ TRAIN + VALIDATION
  evaluate.py             # FINAL TEST 2018 — chạy SAU khi policy đã đóng băng
  postprocess_audit.py    # báo cáo tác động trên 2018 — chạy SAU, không quyết định gì
  uncertainty_audit.py    # bootstrap + MAE theo tháng trên 2018 — chạy SAU, chỉ đo
app/
  __init__.py
  config.py               # hằng số + đường dẫn artifact (có thể override bằng biến môi trường)
  registry.py             # nạp artifact đã đóng băng (nơi DUY NHẤT chạm vào models/)
  schemas.py              # Pydantic: request/response + validation
  serving.py              # dựng feature (CHUNG hàm với train) + dự báo
  main.py                 # FastAPI: route, exception handler, static/template
  templates/
    index.html            # Màn 1 — Giới thiệu
    predict.html          # Màn 2 — Dự báo
    dashboard.html        # Màn 3 — Dashboard + Model Card
  static/
    css/style.css
    js/predict.js
    js/dashboard.js
models/
  ridge_pipeline.joblib   # pipeline đã đóng băng (ĐÃ commit — xem "Artifact mô hình" bên dưới)
  baseline_table.csv      # bảng mean hour × day_of_week (fit train only)
  baseline_meta.json
  run_config.json         # cấu hình run
  model_metadata.json     # metadata + thống kê imputer/scaler đã học
  environment.json        # phiên bản Python & thư viện dùng để sinh artifact
  serving_policy.json     # chính sách phục vụ đã đóng băng + bằng chứng chọn policy
reports/
  project_brief.md
  final_report.md         # BÁO CÁO (15-25 trang khi xuất DOCX/PDF)
  figures/
    data_quality_report.md
    eda_train_only.md
    experiments_report.md      # 2012-2017
    experiments_results.json
    evaluation_report.md       # 2018
    evaluation_results.json
    alpha_tuning.json
    alpha_sensitivity.json  # độ nhạy alpha trên lưới mở rộng (chỉ VALIDATION, có OLS)
    serving_policy.md          # quyết định policy (train+val) + lập luận
    postprocess_audit.md       # tác động policy lên metric sau khi đóng băng
    postprocess_audit.json
    uncertainty_audit.md       # độ bất định: bootstrap theo ngày + MAE theo tháng
    uncertainty_audit.json
    uncertainty_bootstrap.png
    *.png
docs/
  project-log.md         # NHẬT KÝ DỰ ÁN (thời gian · người · giờ · kết quả · vấn đề)
  slides-outline.md       # 11 slide
  demo-script.md          # kịch bản demo 5-7 phút
  viva-questions.md       # 35 câu hỏi + đáp án
tests/
  conftest.py             # fixture dùng chung
  test_data.py            # holiday, duplicate, outlier, invariant, lịch tất định
  test_features.py        # feature, holiday semantics, time split
  test_pipeline.py        # imputer trong pipeline, không rò rỉ, artifact
  test_experiments.py     # bảo vệ FINAL TEST 2018, drift out-of-sample
  test_serving.py         # °C->K, holiday tự tính, multi-weather, KHÔNG train-serving skew
  test_api.py             # route, validation, lỗi, dashboard lấy số từ artifact
  test_serving_policy.py  # policy không test-informed, D1-D3, RAW vs DEPLOYED
  test_report.py          # tài liệu khớp artifact, số test đồng bộ, thứ tự pipeline
  fixtures/sample_traffic.csv
requirements.txt          # khoảng version tương thích (>=) — dùng khi cài mới
requirements-lock.txt     # phiên bản chính xác của môi trường đã sinh artifact
requirements-deploy.txt   # chỉ phần phục vụ web/API — dùng cho deploy Render
render.yaml               # Render Blueprint (build/start/health check)
.python-version           # 3.14.7 — Render đọc file này để chọn Python runtime
release/
  final_report.docx       # bản nộp (đã commit)
  final_report.pdf        # bản nộp (đã commit)
  slides.pptx             # bản nộp (đã commit)
```

**Phân bổ test:** xem bảng ở §13 của `reports/final_report.md`.
Con số tổng **356 test** được tự kiểm chứng: `test_report.py` chạy `pytest --collect-only`
và bắt tài liệu phải khớp đúng số đó — nên tài liệu **không thể** nói sai số test.

---

## Cài đặt & môi trường tái lập

Có hai cách cài, dùng cho hai mục đích khác nhau:

| Cách | Lệnh | Khi nào dùng |
| --- | --- | --- |
| **Tái lập tuyệt đối** | `py -m pip install -r requirements-lock.txt` | Khi cần **đúng** kết quả đã báo cáo. Đây là môi trường đã sinh ra `models/ridge_pipeline.joblib` và mọi con số trong báo cáo. |
| Theo khoảng version | `py -m pip install -r requirements.txt` | Khi cài trên máy mới / máy khác và chấp nhận bản patch-minor mới hơn. Ít rủi ro hơn khi môi trường cũ không còn cài được. |

- `requirements.txt` dùng toán tử `>=` → dễ cài, nhưng **không** bảo đảm cùng kết quả.
- `requirements-lock.txt` là output của `py -m pip freeze` → đúng từng phiên bản đã chạy thật.

Phiên bản thật của môi trường đã dùng để sinh artifact được ghi ở
[`models/environment.json`](models/environment.json) — đọc được bằng máy, không phải gõ tay.
Toàn bộ script dùng `seed = 42` (`models/run_config.json`).

---

## Artifact mô hình (bàn giao sẵn)

`models/ridge_pipeline.joblib` **đã được commit vào repo** (8.133 bytes) — nhỏ hơn nhiều
so với giới hạn của Git, nên người chấm `git clone` là chạy được ngay, **không cần train lại**:

```bat
py -m pip install -r requirements-lock.txt
py -m uvicorn app.main:app --reload
```

| | |
| --- | --- |
| Đường dẫn | `models/ridge_pipeline.joblib` |
| Kích thước | 8.133 bytes |
| SHA256 | `4619678a28b8f47eb047c060b5c27984c6fe4e62c72081d2c756d2ea3bc1abe0` |

Kiểm tra artifact còn nguyên:

```bat
py -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('models/ridge_pipeline.joblib').read_bytes()).hexdigest())"
```

Nếu vì lý do nào đó file không tồn tại (đề yêu cầu không commit file nhị phân), chạy lại
`py src\train.py` để sinh ra — script dùng `seed = 42` và `solver = "lsqr"`, cho ra cùng
kết quả. **Không** tune lại bằng FINAL TEST 2018.

> `.gitignore` vẫn giữ luật `models/*.joblib` cho các lần train sau; file bàn giao được
> thêm bằng `git add -f`. Tương tự, thư mục `release/` vẫn bị ignore nhưng 3 tệp phát
> hành cuối cùng được thêm bằng `git add -f`.

---

## Nhật ký dự án

Xem [`docs/project-log.md`](docs/project-log.md) — bảng theo tuần gồm: người thực hiện,
giờ ước lượng, công việc, kết quả, vấn đề/cách xử lý.


## Xuất tài liệu phát hành (DOCX / PDF / PPTX)

Báo cáo và slide được soạn bằng Markdown để **kiểm chứng được đối chiếu artifact**
(`tests/test_report.py`). Khi nộp, sinh bản DOCX/PDF/PPTX **từ chính Markdown đó** —
không ai chép số tay.

```bat
py -m pip install -r requirements-export.txt
py src\export_docs.py all          :: sinh cả 3
py src\export_docs.py docx         :: chỉ sinh DOCX
py src\export_docs.py pdf          :: chỉ sinh PDF
py src\export_docs.py pptx         :: chỉ sinh PPTX
```

Kết quả:

| Tệp | Nguồn | Công cụ | Quy mô hiện thời điểm |
| --- | --- | --- | --- |
| `release/final_report.docx` | `reports/final_report.md` | pandoc (pypandoc-binary) | 17 mục, 45 bảng, **7 ảnh nhúng** |
| `release/final_report.pdf` | `reports/final_report.md` | reportlab + font Arial | **21 trang A4**, dấu tiếng Việt đầy đủ |
| `release/slides.pptx` | `docs/slides-outline.md` | python-pptx | **12 slide** (1 bìa + 11 nội dung), 4 ảnh thật |

- Báo cáo **21 trang A4** — nằm trong khoảng mục tiêu 15–25 trang.
- Nếu muốn định dạng đẹp hơn cho DOCX: đặt file mẫu `reference.docx` (mẫu định dạng của
  nhà trường) vào thư mục gốc rồi chạy lại — script tự dùng làm `--reference-doc`.
- Nếu thiếu một công cụ, script in **"BỎ QUA"** kèm lý do và **không** tạo file rỗng.
- Muốn cập nhật bản phát hành: sửa Markdown → chạy lại `export_docs.py`. Số liệu không
  bao giờ tự đổi nếu không chạy lại script.

## Bằng chứng chống train-serving skew

Ba lớp kiểm chứng trong `tests/`:

1. `test_no_train_serving_skew_on_real_rows` — dựng feature từ một dòng thật của dataset theo đường dẫn
   serving, rồi so **từng cột** với feature sinh ra từ đường dẫn huấn luyện. Không cột nào được sai lệch.
2. `test_serving_uses_the_same_feature_columns_as_training` — danh sách cột đưa vào pipeline lúc serving
   phải bằng `src.features.FEATURE_COLUMNS_ALL`.
3. `test_pipeline_is_not_refit_at_serving` — gọi API nhiều lần rồi kiểm tra `scaler.mean_` và
   `encoder.categories_` **không đổi**, tức không có fit nào xảy ra lúc phục vụ.

Ngoài ra `test_server_reads_no_target_at_serving` chỉ vào thư mục dữ liệu rỗng và chứng minh API
vẫn dự báo được — tức server không cần (và không đọc) dataset lúc phục vụ.

## Quy tắc chống rò rỉ (áp dụng xuyên suốt)

1. Mọi biến đổi trước khi tách tập phải là **quy tắc tất định**, không học thống kê.
2. Mọi thống kê học được (imputer, scaler, encoder) nằm trong `Pipeline` và **fit trên TRAIN duy nhất**.
3. **Không dùng ngưỡng hậu nghiệm** rút ra từ phân bố toàn bộ tập dữ liệu.
4. **FINAL TEST 2018 không tham gia tuning / model selection.** Mọi thí nghiệm phát triển nằm trong 2012–2017
   và được bảo vệ bằng `assert_no_final_test_rows()`.
5. Không `interpolate` trước khi tách tập.
6. `traffic_volume` không bao giờ làm feature và không bao giờ được đọc lúc phục vụ.
7. Không trộn metric in-sample (train) với out-of-sample (val/test) để kết luận drift.
8. Ứng dụng web/API chỉ nạp artifact; mọi tính năng hiển thị đều đọc từ artifact, không hard-code số liệu.

## Nhóm & phân công

> ### ⚠ CẦN NGƯỜI DÙNG ĐIỀN TRƯỚC KHI NỘP
>
> Mục này **cố ý để trống có marker**. Nhóm **không** tự bịa tên, không bịa phân công,
> không bịa số giờ. Ba dòng dưới đây phải do chính thành viên điền bằng thông tin thật:
>
> | Trường | Cần điền |
> | --- | --- |
> | Thành viên 1 | `[TÊN THÀNH VIÊN 1]` |
> | Thành viên 2 | `[TÊN THÀNH VIÊN 2]` |
> | Phân công thực tế theo tuần | `[PHÂN CÔNG THỰC TẾ]` — xem khuôn `docs/project-log.md` |
>
> Thông tin kỹ thuật của từng tuần **đã có sẵn** trong `docs/project-log.md`; chỉ cần bổ sung
> cột *Người thực hiện* và *Giờ ước lượng* bằng số thật.
>
> **Lưu ý về Git:** lịch sử commit hiện tại thuộc về một tài khoản duy nhất. Nếu đề yêu cầu
> mỗi thành viên có commit riêng, phần đó **phải do chính thành viên đó tự commit** — không
> thay tên tác giả, không tạo commit giả, không backdate.

## Công cụ AI đã sử dụng

**Công cụ AI đã sử dụng:**
- ChatGPT
- Pi Agent

**Mục đích sử dụng:**
- hỗ trợ phân tích yêu cầu đề bài
- rà soát phương pháp chống data leakage
- hỗ trợ viết/sửa mã nguồn
- hỗ trợ xây dựng test
- hỗ trợ kiểm tra Web/API
- hỗ trợ chuẩn bị báo cáo, slide và câu hỏi vấn đáp

**Cách kiểm chứng lại:**
- chạy pipeline trên dataset UCI gốc (SHA256 đối chiếu ở `data/README.md`)
- chạy `py -m pytest tests\ -v`
- đối chiếu mọi metric với artifact do code sinh ra (`models/*.json`, `reports/figures/*.json`)
- smoke-test FastAPI/Web (kể cả các ca input sai)
- đọc lại mã nguồn và tài liệu trước khi bảo vệ

> AI **không** được dùng để quyết định giá trị nghiệm vụ (quy tắc ngày lễ, định nghĩa
> danh mục thời tiết, ngưỡng giá trị vô lý) và **không** được dùng để tạo số liệu.
> Mọi con số trong báo cáo đến từ artifact do mã nguồn sinh ra, và `tests/test_report.py`
> tự động đối chiếu tài liệu với artifact.

# DÀN Ý SLIDE — 11 slide

**Đề tài:** Dự báo lưu lượng giao thông I-94 (chiều westbound) — trạm ATR 301
**Thời lượng trình bày:** ~10 phút · **Phần hỏi đáp:** ~5 phút
**Quy tắc:** mọi biểu đồ phải lấy từ `reports/figures/*.png` (sinh bằng mã nguồn thật),
mọi con số phải lấy từ artifact. Không vẽ lại bằng tay.

> **Danh mục hình có sẵn trong `reports/figures/`:**
> `eda_target_distribution.png` · `eda_traffic_by_hour_dow.png` · `eda_traffic_by_month_year.png` ·
> `eda_weather_distribution.png` · `eda_weather_numeric.png` · `eda_holiday_effect.png` ·
> `final_mae_by_hour_dow.png` · `final_mae_by_weather.png`

---

## Slide 1 — Bìa đề tài

**Nội dung**
- Tên đề tài: *Dự báo lưu lượng giao thông I-94 (chiều westbound) — trạm ATR 301*
- Project 20 — Bài 7: Rò rỉ dữ liệu, chia tập đúng và đánh giá trung thực
- Nhóm · thành viên · ngày bảo vệ
- Dữ liệu: Metro Interstate Traffic Volume (UCI, CC BY 4.0)

**Nói:** "Nhóm em xây dựng mô hình dự báo lưu lượng xe theo giờ tại một trạm đo I-94
chiều westbound, và trọng tâm của bài này là làm sao để **đánh giá** mô hình đó một cách
trung thực."

**Hình:** không (hoặc logo nhóm).

---

## Slide 2 — Bài toán & dữ liệu

**Nội dung**
- Đơn vị quan sát: **1 giờ tại một trạm đo**; biến mục tiêu `traffic_volume` — **lưu lượng, xe/giờ**
- Khoảng thời gian: **2012 – 2018** (dữ liệu kết thúc 30/09/2018)
- Số dòng thô **48.204** → số dòng dùng để mô hình hoá **40.575** (collapse 5.445 nhóm trùng timestamp)
- Ba bẫy đã xử lý:
  1. `holiday == "None"` là **không phải** ngày lễ (dùng `keep_default_na=False`)
  2. `temp ≤ 0 K` (10 dòng) → NaN theo **quy tắc vật lý**
  3. `rain_1h = 9831,3` (1 dòng) → NaN theo **khớp sentinel**, không dùng ngưỡng

**Hình:** `eda_target_distribution.png` (phân bố mục tiêu trên TRAIN)

**Nói:** "Điểm quan trọng: 48.204 dòng thô nhưng chỉ còn 40.575 giờ quan sát, vì một giờ có thể
có nhiều bản ghi thời tiết khác nhau. Chúng em **không** lấy dòng đầu tiên — mà collapse tất định
và giữ đủ bằng multi-hot."

---

## Slide 3 — Ba quy tắc chống rò rỉ đã chốt TRƯỚC khi động vào dữ liệu

**Nội dung**
1. Mọi biến đổi trước khi tách tập là **quy tắc tất định** (giờ, thứ, `is_holiday` từ lịch),
   **không học thống kê**
2. Imputer / scaler / encoder nằm trong `Pipeline`, **fit trên TRAIN duy nhất**
3. `traffic_volume` **không bao giờ** làm feature và **không bao giờ** được đọc lúc phục vụ

**Nói:** "Ba quy tắc này các nhóm em chốt ở tuần đầu, **trước khi** nhìn vào kết quả.
Nếu chốt sau, rất dễ vô tình biến một lựa chọn thành thiên lệch."

**Hình:** sơ đồ nhỏ `Raw → Data cleaning → Features → Time split → Train (fit) / Val / Test`.
Vẽ bằng SmartArt/box, không cần ảnh chụp.

---

## Slide 4 — Chia tập theo thời gian

**Nội dung**

| Tập | Khoảng | n | Vai trò |
| --- | --- | --- | --- |
| TRAIN | 2012-10-02 → 2016-12-31 | 25.329 | fit mô hình + bộ tiền xử lý |
| VALIDATION | 2017 cả năm | 8.713 | chọn `alpha` |
| FINAL TEST | 2018-01-01 → 2018-09-30 | 6.533 | **chỉ đánh giá** |

Assert chạy mỗi lần: `max(train) < min(val)` ✓ · `max(val) < min(test)` ✓ · 0 timestamp chung ✓

**Hình:** `eda_traffic_by_month_year.png` (thấy rõ 3 vùng thời gian tách nhau) hoặc dải thời gian tự vẽ.

**Nói:** "Dữ liệu này có 22,79 % số giờ bị thiếu, và em **không** nội suy — một giờ không có quan
sát thì không tạo quan sát giả."

---

## Slide 5 — Baseline và mô hình

**Nội dung**
- **Baseline:** trung bình lưu lượng theo `giờ × thứ trong tuần`, fit **chỉ trên TRAIN**
- **Mô hình:** Ridge (L2) trong `Pipeline`
  `ColumnTransformer[OneHotEncoder | SimpleImputer(median) → StandardScaler | passthrough] → Ridge(lsqr)`
- 4 cột categorical · 5 cột numeric · 12 cột nhị phân → **217** cột sau tiền xử lý
- `alpha` chọn theo **MAE trên validation 2017** → `alpha = 0,001`

**Hình:** `eda_traffic_by_hour_dow.png` — cho thấy baseline khai thác đúng cấu trúc mạnh nhất.

**Nói:** "Baseline này mạnh vì nó nắm đúng cấu trúc hai đỉnh sáng/tối. Ridge bắt đầu từ cùng cấu
trúc `hour_dow` rồi **cộng thêm** thời tiết và tháng."

---

## Slide 6 — FINAL TEST 2018 (kết quả chính thức)

**Nội dung**

| Mô hình | MAE | RMSE | R² | n |
| --- | --- | --- | --- | --- |
| Baseline (giờ × thứ) | 272,90 | 473,13 | 0,9426 | 6.533 |
| **Ridge (alpha = 0,001)** | **259,73** | **416,78** | **0,9554** | 6.533 |
| Cải thiện | **−13,17 (−4,8 %)** | −56,35 (−11,9 %) | +0,0128 | — |

- Cùng một tập test, cùng một tập huấn luyện → phép so sánh **công bằng**
- 2018 **không** dùng để chọn mô hình hay `alpha`
- **Kèm điều kiện:** block bootstrap theo ngày lịch → hiệu MAE **+13,16**,
  CI 95 % **[+3,80; +22,00]** (không chứa 0), Ridge thắng **8/9 tháng**
- **Nhưng** ở 3/4 cửa sổ dev 2012–2017 thì Ridge **thua** baseline → không nói
  chung "mô hình luôn hơn baseline"

**Hình:** `uncertainty_bootstrap.png` (phân phối bootstrap + MAE theo tháng)

**Nói:** "MAE giảm 13,16 so với baseline, và khoảng tin cậy 95 % không chứa 0 — nhưng ở các
cửa sổ 2012–2017 thì hướng ngược lại. Nên con số này đúng cho **cửa sổ 2018 với tập huấn
luyện 2012–2016**, không phải một tuyên bố chung. Và như phần tiếp theo cho thấy, câu hỏi
quan trọng hơn là: **làm sao biết 13,16 đó là thật** chứ không phải do ta tự lừa mình."

---

## Slide 7 — Thí nghiệm 1b/1c: đo cái đo được, không đoán cơ chế

**Nội dung**

Bốn arm dùng **CHUNG một tập test** (nửa 2017, n = 4.357), lặp trên 5 seed cố định:

| Arm | Train đến | n train | MAE (TB ± SD) |
| --- | --- | --- | --- |
| A | 2015-12-31 | 17.491 | 270,30 ± 2,32 |
| B | 2016-12-31 | 25.329 | 269,74 ± 2,53 |
| C | 2016 + nửa 2017 | 29.685 | 263,19 ± 2,19 |
| **D** | **ngẫu nhiên từ C, đúng số dòng của B** | **25.329** | **262,98 ± 2,21** |

- ΔMAE (C − B) = **−6,55 ± 0,37** · ΔMAE (D − B) = **−6,76 ± 0,39** → chênh **+0,21**
- Thí nghiệm 1c (tháng chẵn → train, tháng lẻ → test): ΔMAE = **−2,40**
  → chênh giữa "2017 rải rác toàn năm" và "2017 chỉ ở tháng khác" là **4,15 MAE**

**Nói:** "Nhóm em **không** nói 6,55 điểm này là do mô hình nhìn thấy hàng xóm. Mô hình của em
không có đặc trưng lag, nên nó không thể *nhớ* giá trị của dòng lân cận. Điều đo được là: dữ liệu
2017 nằm ở **cùng tháng** với dòng cần dự báo thì giúp ích nhiều hơn 4,15 MAE so với nằm ở tháng
khác — và điều đó **không** phải do nhiều dòng hơn, vì arm D cùng kích thước với arm B mà vẫn
giữ nguyên hiệu ứng. Đây là mô tả, không phải bằng chứng nhân quả."

**Hình:** bốn cột MAE (vẽ từ bảng) + hai cột ΔMAE. Không cần ảnh chụp.

**Câu hỏi dự kiến:** "Vậy random split lạc quan bao nhiêu?" → **5,43 ± 0,81 MAE** khi so trên
*cùng một tập dòng đánh giá* (§Slide 7 / §11.1 báo cáo). Nhỏ hơn nhiều so với bài toán chuỗi thời
gian có lag, vì mô hình không có lag.

---

## Slide 8 — Mô hình hỏng ở đâu? (phân tích lỗi, kèm số mẫu)

**Nội dung**

| Phân khúc | n | MAE Ridge | MAE baseline |
| --- | --- | --- | --- |
| Ngày thường | 6.366 | 239,49 | 250,58 |
| **Ngày lễ** | **167** | **1.031,44** | 1.123,43 |
| Không có hiện tượng cực đoan | 5.749 | 235,96 | 245,32 |
| **Có tuyết** | **521** | **524,50** | 585,36 |
| **Có sương mù** | 192 | **536,05** | 640,17 |

- Ngày lễ: MAE **gấp 4,3 lần**, thiên lệch **+58,68** (mô hình dự báo cao hơn thực tế)
- Chỉ **7 ngày lễ** trong 2018 → ước lượng có độ bất định lớn
- Phân khúc **Squall có n = 0** → báo cáo ghi "không đánh giá được", **không** báo số 0

**Hình:** `final_mae_by_weather.png` và `final_mae_by_hour_dow.png`

**Nói:** "Ridge nhỉnh hơn baseline ở **cả 7/7 ngày trong tuần** và **17/24 giờ**. Nhưng ở giờ đêm
00–04 thì **baseline lại thắng**, ví dụ 3 giờ: baseline 31,14 so với Ridge 142,14.

Và quan trọng: nhóm em **đã kiểm chứng lại trên dữ liệu 2012–2017**, không chỉ nhìn 2018 —
Ridge kém ở **toàn bộ 5 giờ đêm trong cả 3/3 cửa sổ**. Nên đây là **đặc tính của mô hình**, không
phải đặc điểm của năm 2018. Giả thuyết: mô hình cộng tuyến tính nên hiệu ứng thời tiết cộng
một lượng *tuyệt đối* giống nhau ở mọi giờ, trong khi thực tế nó *tương đối* theo lưu lượng.
Nhóm thử log-target: MAE giờ đêm giảm gần một nửa, nhưng MAE tổng lại xấu hơn — và nhóm **không**
thay mô hình, vì giả thuyết này nêu ra sau khi đã nhìn 2018."

---

## Slide 9 — Ứng dụng Web / API: chỉ nạp artifact, không train lại

**Nội dung**
- 3 màn hình: **Giới thiệu** (`/`) · **Dự báo** (`/du-bao`, gọi `POST /api/traffic-forecast` thật) ·
  **Dashboard & Model Card** (`/dashboard`)
- 4 endpoint: `GET /health` · `GET /api/model-info` · `POST /api/traffic-forecast` ·
  `GET /api/dashboard-metrics`
- **Không** train, **không** tune alpha, **không** fit lại encoder/imputer/scaler,
  **không** đọc target từ dataset
- Mọi feature sinh bằng **CHUNG hàm `src.features.build_features`** với lúc huấn luyện
  → **không có train-serving skew**
- Số liệu dashboard **đọc từ artifact**, không hard-code trong HTML/JS

**Hình:** ảnh chụp màn 2 (form dự báo) và màn 3 (dashboard) — chụp từ `http://localhost:8000`.

**Nói:** "Đây là chỗ dễ sai nhất. Nếu backend tự tính `is_holiday` bằng cách quét dataset, thì lúc
dự báo chỉ có một dòng dữ liệu — không làm được, và còn là rò rỉ. Nhóm em dùng lịch tất định để
tính được từ ngày tháng."

---

## Slide 10 — Model Card (tóm tắt) & Hạn chế

**Nội dung**

| | |
| --- | --- |
| Mô hình | Ridge, `alpha = 0,001`, solver `lsqr`, 217 đặc trưng |
| Dữ liệu | 48.204 dòng thô → 40.575 dòng mô hình hoá |
| Train / Val / Test | 2012–2016 (25.329) / 2017 (8.713) / 2018 (6.533) |
| FINAL TEST | MAE 259,73 · RMSE 416,78 · R² 0,9554 |

**Hạn chế (nói thẳng):**
1. Một trạm I-94 westbound — **không đại diện toàn thành phố**
2. Ngày lễ: MAE gấp 4,3 lần
3. Thời tiết cực đoan: MAE gấp 1,84 lần
4. Test 2018 chỉ tới **30/09** — và đây lại là phần **dễ hơn** của năm: chấm Jan–Sep 2017 cho
   MAE **250,96** so với **272,12** cả năm (chênh **−21,16**). Tháng 11–12 là hai tháng tệ nhất.
5. Ngày lễ chỉ **7 ngày lịch** (167 giờ), MAE từng ngày lệch nhau gần 3 lần (518 → 1.509)
6. Dữ liệu lịch sử **2012–2018**
7. Lịch State Fair chỉ có **2012–2020**; ngoài phạm vi hệ thống **báo rõ, không tự đoán**
8. **Không dùng cho mục đích safety-critical**
9. Dự báo thô âm ở 34/6.533 dòng → API chặn về 0 theo policy `max(0, ·)`
10. Mô hình ≈ OLS: `alpha ∈ [0; 0,01]` chỉ chênh 0,09 MAE trên validation
11. Ridge **thua** baseline ở 3/4 cửa sổ dev 2012–2017

**Nói:** "Điểm 7 là ví dụ của nguyên tắc mà nhóm em theo: khi ngoài phạm vi, hệ thống **báo rõ
ra** thay vì âm thầm đoán. Điểm 4 nhóm em muốn nhấn mạnh: test 2018 thiếu quý IV, mà quý IV lại là
phần khó nhất — nhóm em **định lượng** được việc này bằng validation chứ không chỉ nói miệng.
Điểm 9 nhóm em phân biệt **RAW MODEL** với **DEPLOYED PREDICTOR** — số 259,73 là của mô hình,
số 257,54 là của lớp phục vụ, và hai thứ đó không được gọi chung tên. Chính sách cắt về sàn được
đóng băng trên TRAIN + VALIDATION trước khi nhìn kết quả 2018, nên nó không phải test-informed
postprocessing."

---

## Slide 11 — Kết luận

**Nội dung**
1. **Mô hình nhỉnh hơn baseline trên 2018** — có điều kiện: MAE 259,73 so với 272,90,
   CI 95 % của hiệu MAE [+3,80; +22,00], thắng 8/9 tháng; **nhưng** ở 3/4 cửa sổ
   dev 2012–2017 thì ngược lại
2. **Cách đánh giá cũng làm lệch kết luận**: random split lạc quan **5,43 ± 0,81 MAE** trên cùng
   tập dòng đánh giá (§Slide 7); khoảng cách thời gian thêm **4,15 MAE**
3. **Biết mô hình hỏng ở đâu** cũng là kết quả: ngày lễ, tuyết, sương mù, giờ đêm
4. **Nói trung thực, kể cả khi bất tiện**: không gán cơ chế cho một con số khi chưa tách được
   các yếu tố — bản đầu của báo cáo đã từng kết luận sai, và nhóm đã sửa
5. **Minh bạch**: 385 test pass, 0 fail; web/API chạy thật, chỉ nạp artifact

**Câu kết:**
> "Trong bài toán chuỗi thời gian, **cách ta chia tập quan trọng ngang với việc ta chọn mô hình** —
> và nói trung thực về giới hạn là một phần của kết quả, không phải điểm trừ."

**Hình:** ảnh chụp dashboard (mục Model Card) hoặc logo nhóm.

---

## Phụ lục trình chiếu (không tính vào 11 slide)

| Mã | Nội dung | Khi nào dùng |
| --- | --- | --- |
| A1 | Bảng alpha đầy đủ (13 giá trị) + độ nhạy cảm lưới mở rộng (17 giá trị, có OLS) | GV hỏi "sao chọn alpha nhỏ vậy" |
| A2 | Bảng MAE đầy đủ 24 giờ | GV hỏi chi tiết theo giờ |
| A3 | Bảng PSI theo năm | GV hỏi về drift |
| A4 | Rolling-origin 3 fold | GV hỏi "có drift không" |
| A5 | Bảng so sánh feature train vs serving | GV hỏi về train-serving skew |
| A6 | Danh sách 385 test theo nhóm | GV hỏi về kiểm thử |
| A7 | Chi tiết collapse 5.445 nhóm trùng | GV hỏi về tiền xử lý |

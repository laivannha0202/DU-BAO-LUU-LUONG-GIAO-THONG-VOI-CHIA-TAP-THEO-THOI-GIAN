# CÂU HỎI VIVA — 33 câu

> **Cách dùng:** mỗi câu có *đáp án ngắn* đủ để trả lời trong 30–60 giây, kèm *gợi ý mở rộng*
> nếu GV hỏi sâu. Số liệu trong đáp án đều lấy từ artifact.

**Mục lục**
- [A. Bài toán & dữ liệu (1–7)](#a-bài-toán--dữ-liệu-17)
- [B. Tiền xử lý (8–13)](#b-tiền-xử-lý-813)
- [C. Rò rỉ dữ liệu & chia tập (14–21)](#c-rò-rỉ-dữ-liệu--chia-tập-1421)
- [D. Mô hình & đánh giá (22–27)](#d-mô-hình--đánh-giá-2227)
- [E. Web/API & triển khai (28–32)](#e-webapi--triển-khai-2832)

**Cách chạy lại toàn bộ pipeline (thứ tự bắt buộc):**

```bat
py src\download_data.py
py src\data.py
py src\eda.py
py src\train.py
py src\experiments.py
py src\freeze_serving_policy.py   :: đóng băng policy max(0,·) TRƯỚC khi mở năm 2018
py src\evaluate.py                :: FINAL TEST 2018
py src\postprocess_audit.py       :: chỉ đo tác động của policy đã đóng băng
py -m pytest tests\ -v
py -m uvicorn app.main:app --reload
```

> Nếu làm ngược lại (chạy `evaluate.py` trước `freeze_serving_policy.py`) thì chính sách hậu xử
> lý sẽ trở thành **test-informed postprocessing** — đó là điều phải tránh. Xem Câu 32.

---

# A. Bài toán & dữ liệu (1–7)

### Câu 1
**Mô hình của nhóm dự báo cái gì, đơn vị ra sao?**
**Đáp án:** Dự báo `traffic_volume` — lưu lượng xe đi qua trạm đo ATR 301 trên I-94 chiều westbound, **đơn vị xe/giờ**. Quan sát là **1 giờ tại một trạm đo**, không phải lưu lượng cả tuyến.
*Gợi ý mở rộng:* vì sao chỉ một trạm? Vì dữ liệu UCI chỉ có một trạm — xem mục hạn chế ở Câu 30.

### Câu 2
**Dữ liệu gồm bao nhiêu dòng? Vì sao 48.204 dòng thành 40.575 dòng?**
**Đáp án:** 48.204 dòng thô. Có **5.445 nhóm trùng `date_time`** (mỗi giờ 2–6 bản ghi thời tiết khác nhau) chứa 13.074 dòng. Collapse tất định còn **40.575 giờ quan sát**, loại 7.629 dòng thừa.
*Gợi ý mở rộng:* nói tiếp rằng `traffic_volume` và `holiday` **bất biến 100 %** trong mọi nhóm trùng, nên collapse không đổi mục tiêu ở dòng nào.

### Câu 3
**Vì sao KHÔNG dùng `drop_duplicates(keep="first")`?**
**Đáp án:** Vì `keep="first"` phụ thuộc **thứ tự dòng trong file** (không tất định, khó tái lập), và nó **vứt bỏ** các bản ghi thời tiết còn lại — tức mất thông tin. Nhóm dùng: phép đo thì lấy **trung vị**, thời tiết thì giữ **multi-hot** kèm nhãn đại diện tất định.

### Câu 4
**Một giờ có thể có nhiều hiện tượng thời tiết. Nhóm xử lý thế nào?**
**Đáp án:** Mã hoá **multi-hot / multi-label** — giữ **11 cột nhị phân** `wm_clear` … `wm_thunderstorm`, hiện 1 nếu giờ đó có hiện tượng đó. Thêm một **nhãn đại diện** tất định (giá trị xuất hiện nhiều nhất, hoà thì alphabet) để làm feature categorical, và một **mức độ** nghiêm trọng để gộp nhiều mô tả.

### Câu 5
**Vì sao cột `holiday` cần đọc `keep_default_na=False`?**
**Đáp án:** Vì cột chứa **chuỗi `"None"`** (nghĩa là *không phải* ngày lễ), không phải ô trống. Đọc bằng mặc định của pandas, `"None"` bị hiểu thành NaN → **mọi dòng trông như ngày lễ**. Audit xác nhận 0 dòng bị hiểu sai sau khi sửa.

### Câu 6
**Nhóm xử lý bao nhiêu phần trăm số giờ bị thiếu? Có nội suy không?**
**Đáp án:** **11.976 giờ thiếu, tức 22,79 %** so với 52.551 giờ lý thuyết. **Không nội suy** — vì nội suy là học thống kê từ hàng xóm, mà hàng xóm có thể nằm ở tập test, tức là rò rỉ.

### Câu 7
**Feature nào chủ yếu giải thích lưu lượng, và vì sao?**
**Đáp án:** **Lịch** — `hour_dow` (tương tác giờ × thứ), rồi `is_holiday` và `month`. EDA trên TRAIN cho thấy hai đỉnh sáng ~07–08h và chiều ~16–17h, và hình dạng ngày làm việc khác hẳn cuối tuần (00:00 Chủ nhật 1.335 xe so với 617 xe thứ Hai). Thời tiết là yếu tố phụ trợ.

---

# B. Tiền xử lý (8–13)

### Câu 8
**Giá trị `temp ≤ 0 K` và `rain_1h = 9831,3` được xử lý thế nào? Vì sao không dùng ngưỡng thống kê?**
**Đáp án:** 10 dòng `temp ≤ 0` → NaN theo **quy tắc vật lý** (0 K là nhiệt độ tuyệt đối); 1 dòng mưa → NaN theo **khớp chính xác** giá trị sentinel đã audit. Không dùng ngưỡng hậu nghiệm (ví dụ "trên 99,9 phân vị") vì ngưỡng đó rút từ phân bố **toàn bộ** tập, tức đã học thống kê trên cả validation và test — đó là rò rỉ.

### Câu 9
**Ai điền giá trị thiếu cho `temp`, `rain_1h`, `snow_1h`?**
**Đáp án:** `SimpleImputer(strategy="median")` **bên trong `Pipeline`**, và chỉ `fit` trên **TRAIN**. Median lấy từ TRAIN: `temp` = 282,08 · `rain_1h` = 0,0 · `snow_1h` = 0,0 · `clouds_all` = 40,0 · `weather_severity` = 1,0. Giá trị này được lưu trong `model_metadata.json` để kiểm chứng.

### Câu 10
**Vì sao phải có `SimpleImputer` bên trong pipeline?**
**Đáp án:** Vì bước đánh dấu giá trị vô lý ở `src/data.py` tạo ra NaN **có chủ đích**. Nếu điền NaN trước khi tách tập thì đó là học thống kê trên toàn bộ dữ liệu. Đặt imputer trong `Pipeline` giữ đúng nguyên tắc: mọi thống kê học được chỉ đến từ TRAIN.

### Câu 11
**Vì sao `StandardScaler` cũng phải nằm trong pipeline?**
**Đáp án:** Vì mean và std cũng là thống kê học được. Nếu chuẩn hoá trên toàn bộ dữ liệu rồi mới tách tập, thông tin phân bố của test đã "rò" vào quá trình huấn luyện. Test `test_pipeline.py` kiểm tra imputer đứng **trước** scaler trong pipeline.

### Câu 12
**Feature categorical xử lý thế nào? Nếu gặp giá trị chưa từng thấy thì sao?**
**Đáp án:** `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` — chỉ tạo cột cho danh mục có trong TRAIN; giá trị lạ thì **toàn bộ cột one-hot bằng 0** thay vì lỗi. Nhờ vậy API không bao giờ crash vì một giá trị thời tiết mới.

### Câu 13
**`temp` lưu đơn vị gì? API nhận đơn vị gì?**
**Đáp án:** Dataset lưu **Kelvin** (nhiệt độ tuyệt đối). API nhận **°C** và backend tự cộng **273,15**. Phép biến đổi này là quy tắc tất định, không học gì, nên không gây train-serving skew.

---

# C. rò rỉ dữ liệu & chia tập (14–21)

### Câu 14
**Vì sao chọn time split thay vì random split?**
**Đáp án:** Vì mục tiêu là **dự báo tương lai**. Random split trộn các năm vào tập huấn luyện, nên khi dự báo một giờ, mô hình đã được huấn luyện trên dữ liệu **cùng năm và gần thời điểm đó** — làm kết quả **lạc quan giả**. Lượng lạc quan đo được: **5,43 ± 0,81 MAE** khi so hai mô hình trên **cùng một tập dòng đánh giá** (Câu 17).

### Câu 15
**Ba tập được chia như thế nào, có bảo đảm gì không?**
**Đáp án:** TRAIN 2012-10-02 → 2016-12-31 (**25.329**), VALIDATION 2017 cả năm (**8.713**), FINAL TEST 2018-01-01 → 2018-09-30 (**6.533**). Có 3 assert chạy mỗi lần: `max(train) < min(validation)`, `max(validation) < min(test)`, và **0 timestamp trùng** giữa các tập.

### Câu 16
**2018 có được dùng để chọn `alpha` hay chọn mô hình không?**
**Đáp án:** **Không.** Toàn bộ lựa chọn về tiền xử lý, đặc trưng, mô hình và `alpha` chốt trên **2012–2017**. Quy tắc này được **cưỡng chế bằng mã nguồn**: `src/experiments.py` gọi `assert_no_final_test_rows()` ở mọi hàm và sẽ dừng chương trình nếu dòng 2018 lọt vào; `tests/test_experiments.py` có 26 test bảo vệ điều đó.
*Gợi ý mở rộng:* phát biểu trung thực — "2018 không tham gia tuning. Pipeline được đánh giá lại sau các sửa lỗi phương pháp, và kết quả 2018 không được dùng để tiếp tục tối ưu."

### Câu 17
**Thí nghiệm 1 cho kết quả gì — random split và đánh giá trên tương lai chênh nhau bao nhiêu?**
**Đáp án:** Lặp trên **5 seed cố định**, báo trung bình ± độ lệch chuẩn. Time split (train ≤ 2015, test 2016–2017) MAE **306,48**; random split MAE dao động 284,94–296,75 theo seed.

- So kiểu thông thường (mỗi arm tập test riêng): **−14,79 ± 3,83** — nhưng **không dùng được**, vì hai tập test khác thành phần năm.
- So **trên cùng một tập dòng đánh giá** (đây là con số trả lời câu hỏi): **−5,43 ± 0,81**.

**Vì sao nhỏ?** Mô hình chỉ dùng đặc trưng lịch và thời tiết, **không có lag** nên không thể *nhớ* giá trị dòng lân cận. Phần lạc quan còn lại đến từ việc random split nhìn thấy được cả tháng 11–12/2017, trong khi time-split train chỉ tới 2015.

### Câu 18
**Thí nghiệm 1b là gì và kết quả ra sao?**
**Đáp án:** Bốn arm dùng **CHUNG một tập test** (nửa 2017, n = 4.357), lặp 5 seed:

| Arm | Train | n train | MAE (TB ± SD) |
| --- | --- | --- | --- |
| A | ≤ 2015 | 17.491 | 270,30 ± 2,32 |
| B | ≤ 2016 | 25.329 | 269,74 ± 2,53 |
| C | ≤ 2016 + nửa 2017 | 29.685 | 263,19 ± 2,19 |
| **D** | **ngẫu nhiên từ C, đúng số dòng của B** | **25.329** | **262,98 ± 2,21** |

ΔMAE (C − B) = **−6,55 ± 0,37**; ΔMAE (D − B) = **−6,76 ± 0,39** → chênh **+0,21**.

**Điểm mấu chốt:** arm D **cùng kích thước** với arm B, nên hiệu ứng **không** phải do "nhiều dòng hơn" mà là do *có* dữ liệu 2017 trong tập huấn luyện.

### Câu 19
**Ý nghĩa quan trọng nhất của thí nghiệm 1b/1c là gì — và có phải "mô hình nhìn thấy hàng xóm" không?**
**Đáp án (trả lời thẳng):** **Không**, nhóm không khẳng định điều đó, vì mô hình **không có đặc trưng lag** nên không thể nhớ *giá trị* của dòng lân cận.

Điều đo được là **khoảng cách thời gian** ảnh hưởng tới con số: Thí nghiệm 1c dùng **khối liên tục** (tháng chẵn 2017 → train, tháng lẻ → test) cho ΔMAE chỉ **−2,40**, so với **−6,55** khi rải rác toàn năm → chênh **4,15 MAE**. Hiệu ứng này gắn với việc tập huấn luyện có dữ liệu ở **cùng tháng và gần giờ** với dòng cần dự báo.

Đây là **mô tả hiệu ứng, không phải bằng chứng nhân quả** (arm C và P2 khác nhau cả về tháng được thay vào train). Bài học phương pháp: **đừng gán một cơ chế cho một con số khi chưa tách được các yếu tố.** Bản đầu của báo cáo đã từng kết luận sai theo hướng này và nhóm đã sửa.

*Nếu GV hỏi "vậy thí nghiệm này còn giá trị gì":* nó vẫn chứng minh điều cần thiết để bảo vệ FINAL TEST — **khoảng cách thời gian làm thay đổi đáng kể con số đánh giá**, nên kết luận trên 2018 bắt buộc phải đến từ mô hình chưa từng thấy năm 2018 (tức arm B).

### Câu 20
**Làm sao biết `is_holiday` ở thời điểm dự báo mà không bị rò rỉ?**
**Đáp án:** Nhóm viết **lịch ngày lễ tất định** trong `src/holidays.py`, tính thuần từ ngày tháng: 10 ngày lễ liên bang theo quy tắc lịch + bảng ngày khai mạc Minnesota State Fair do bang công bố. Lịch khớp **53/53** ngày lễ dataset, **bỏ sót 0**, sai tên 0. Lịch là **thông tin công cộng biết trước** — ai cũng biết 4/7 là lễ trước khi nó tới — nên dùng làm feature là hợp lệ, không phải rò rỉ.
*Gợi ý mở rộng:* nói cách **không** làm: quét các dòng khác trong cùng ngày để biết cột `holiday` khác "None" — vừa cần toàn bộ bảng dữ liệu (lúc dự báo chỉ có một dòng), vừa là rò rỉ thông tin của chính ngày đó.

### Câu 21
**Vì sao không dùng lag feature của `traffic_volume` trong mô hình chính?**
**Đáp án:** Lag của chính biến mục tiêu là **đường nghiệm dễ rơi vào rò rỉ thời gian** nhất — mô hình học "tương lai gần nhất giống hiện tại" từ dữ liệu liền kề. Làm đúng thì phải tính lag theo **thời điểm** (`date_time − k giờ`, không `shift` theo dòng vì dữ liệu thiếu giờ) rồi mới drop NaN, và phải backtest nhiều kỳ. Đề tài này tập trung vào **phương pháp đánh giá**, nên nhóm giữ nguyên bộ feature không dùng lag ở mô hình chính; phần mở rộng lag được chạy riêng như một thí nghiệm độc lập, **không** đưa vào phục vụ.

---

# D. Mô hình & đánh giá (22–28)

### Câu 22
**Baseline là gì, và vì sao nó là baseline hợp lý?**
**Đáp án:** Trung bình `traffic_volume` theo `giờ × thứ trong tuần`, fit **chỉ trên TRAIN**, lưu ở `models/baseline_table.csv` (kèm `global_fallback` = 3.252,51). Hợp lý vì EDA cho thấy cấu trúc giờ × thứ là mạnh nhất của bài toán, nên đây là đối thủ mạnh, không phải một baseline dễ thắng.
*Gợi ý mở rộng:* nhấn mạnh cả hai mô hình **cùng fit trên 2012–2016, cùng test trên 2018** → phép so sánh công bằng.

### Câu 23
**Kết quả FINAL TEST 2018 là gì — và nói thật, mô hình có hơn baseline không?**
**Đáp án:** Ridge (alpha = 0,001): **MAE 259,73 · RMSE 416,78 · R² 0,9554** trên 6.533 giờ. Baseline: MAE 272,90 · RMSE 473,13 · R² 0,9426. Nhỉnh hơn **13,16 MAE, tức 4,8 %**.

**Phải kèm điều kiện:** đây là kết luận **cho cửa sổ 2018** với tập huấn luyện 2012–2016, **không phải** tuyên bố chung. Xem Câu 28.

### Câu 24
**MAE hay RMSE phù hợp hơn ở đây? Vì sao chọn MAE làm tiêu chí chọn alpha?**
**Đáp án:** Cả hai đều báo cáo. Nhóm chọn **MAE** làm tiêu chí vì đơn vị cùng với mục tiêu (xe/giờ) và **diễn giải trực tiếp** là "trung bình sai lệch bao nhiêu xe/giờ" — quan trọng với người lập kế hoạch. RMSE nhạy với outlier hơn; ở đây RMSE của Ridge (416,78) thấp hơn baseline (473,13) nên cả hai chỉ số cùng kết luận. **Lưu ý:** trên pseudo-test 2016–2017 thì hai chỉ số **không** đồng ý (baseline thắng MAE, Ridge thắng RMSE) — vì vậy nhóm không bao giờ dùng một chỉ số đơn lẻ.

### Câu 25
**`alpha` được chọn thế nào? Kết quả có nhạy cảm không?**
**Đáp án:** Thử 13 giá trị từ 0,001 đến 1.000, chọn theo **MAE trên validation 2017** → `alpha = 0,001`. Không nhạy cảm ở vùng nhỏ: 0,001 → 272,12; 0,01 → 272,14; 0,1 → 272,35; 0,3 → 272,81. Nhạy cảm mạnh khi alpha lớn: 30 → 429,00; 1.000 → 1.461,15. Dữ liệu không quá nhiễu nên hiệu chuẩn không cần mạnh; nhóm chọn biên nhỏ nhất vì ít giả định nhất.

### Câu 26
**Mô hình hỏng ở đâu? Nêu kèm số mẫu.**
**Đáp án:** Ở **ngày lễ**: MAE **1.031,44** với n = **167** giờ, so với 239,49 ở ngày thường (n = 6.366) — gấp 4,3 lần, thiên lệch +58,68. Ở **thời tiết cực đoan**: có tuyết MAE 524,50 (n = 521), có sương mù 536,05 (n = 192), so với 235,96 khi không có (n = 5.749). Ngoài ra ở **giờ đêm** 00–04h baseline lại **thắng** Ridge — chi tiết ở Câu 34.
*Gợi ý mở rộng:* phân khúc Squall có n = 0 — nhóm báo "không đánh giá được" chứ không báo số 0; phân khúc Smoke chỉ 2 mẫu, được gắn nhãn "mẫu nhỏ — thận trọng".

### Câu 27
**Có drift không? Làm sao biết mà không trộn metric in-sample với out-of-sample?**
**Đáp án:** Rolling-origin 3 fold, tất cả đều out-of-sample: 2015 → 325,98; 2016 → 343,77; 2017 → 272,12. MAE giảm 53,86 (16,5 %) nhưng **không kết luận được là mô hình tiến bộ** — có thể chỉ vì 2017 dễ hơn. Nhóm còn đo **PSI** cho phân bố biến đầu vào: `traffic_volume` rất ổn định (PSI < 0,035 mọi năm), còn `temp` và `clouds_all` dịch chuyểch mạnh ở 2012 và 2015 (PSI 0,70–1,10). Hai khái niệm được tách bạch: *performance theo tập* vs *drift phân bố*.

### Câu 28
**Chắc chắn 13,16 MAE cải thiện đó là thật chứ không phải ngẫu nhiên? Nhóm kiểm tra thế nào?**
**Đáp án:** Nhóm chạy **block bootstrap theo khối ngày lịch** (`src/uncertainty_audit.py`, 4.000 lần, seed cố định) cho **hiệu cặp** baseline − Ridge.

- Vì sao theo **ngày** chứ không theo dòng: các giờ trong cùng một ngày lịch không độc lập (cùng thời tiết, cùng ngày làm việc), lấy mẫu lại từng dòng sẽ cho CI **hẹp hơn thực tế**.
- Kết quả 2018: hiệu MAE **+13,16**, CI 95 % **[+3,80; +22,00]** — **không chứa 0**; 99,58 % lần lấy mẫu Ridge thắng. Hiệu RMSE +56,36, CI [+31,07; +80,63].
- Theo tháng: Ridge thắng **8/9 tháng**, thua ở tháng 8 (−18,77).

**Quan trọng — hiệu ứng này không nhất quán:** áp dụng đúng phép đó cho dữ liệu dev 2012–2017, Ridge chỉ thắng ở **1/4 cửa sổ**:

| Cửa sổ | Hiệu MAE | CI 95 % |
| --- | --- | --- |
| pseudo-test 2016–2017 | −11,53 | [−19,72; −3,79] |
| fold 1 (2015) | −43,13 | [−60,91; −25,92] |
| fold 2 (2016) | −26,48 | [−38,32; −14,72] |
| fold 3 (2017) | +6,54 | [−4,50; +16,50] |

**Kết luận trung thực:** "Ridge hơn baseline **trên 2018**" thì đúng; "mô hình luôn hơn baseline" thì **sai**. Các cửa sổ mà Ridge thua đều có tập huấn luyện rất thưa (2014 kết thúc 08/08, 2015 bắt đầu 11/06) — Ridge cần dữ liệu đủ dày, baseline thì không.

*Nếu GV hỏi "vậy có nên dùng mô hình không":* vẫn nên, vì 2012–2016 là tập huấn luyện đầy đủ nhất mà dữ liệu cho phép, và đó đúng là tình huống sử dụng thật. Nhưng phải nói kèm điều kiện.

---

# E. Web/API & triển khai (29–33)

### Câu 29
**Ứng dụng web có huấn luyện lại mô hình không?**
**Đáp án:** **Không.** Ứng dụng chỉ `predict` trên `models/ridge_pipeline.joblib` đã đóng băng. Không tune alpha, không fit lại encoder/imputer/scaler, không đọc target từ dataset. Có test `test_pipeline_is_not_refit_at_serving` gọi API 8 lần rồi kiểm tra `scaler.mean_` và `encoder.categories_` **không đổi**, và test `test_server_reads_no_target_at_serving` trỏ thư mục dữ liệu vào chỗ trống rồi chứng minh API vẫn dự báo được.

### Câu 30
**Làm sao chắc chắn không có train-serving skew?**
**Đáp án:** Mọi feature kỹ thuật hoá ở tầng serving được sinh bằng **CHUNG hàm `src.features.build_features`** với lúc huấn luyện. Có test `test_no_train_serving_skew_on_real_rows` chọn 3 timestamp thật (gồm một ngày lễ), dựng feature theo đường dẫn serving, rồi so **từng cột** với đường dẫn huấn luyện — không cột nào lệch.
*Gợi ý mở rộng:* giải thích 3 nơi dễ sai: `is_holiday` (phải từ lịch, không quét dataset), nhiệt độ (phải cùng phép °C→K), và multi-hot thời tiết (phải cùng cách gộp).

### Câu 31
**Ứng dụng có dùng được cho nhiều trạm / nhiều thành phố không?**
**Đáp án:** **Không.** Dữ liệu chỉ có một trạm (ATR 301, I-94 westbound). Mô hình học đặc trưng riêng của trạm đó, và việc mở rộng sang trạm khác cần huấn luyện lại từ đầu với dữ liệu trạm đó. Đây là hạn chế được ghi rõ trong Model Card.

### Câu 32
**Nếu người dùng nhập sai (mây 150 %, thời tiết gõ sai) thì hệ thống xử lý ra sao?**
**Đáp án:** Trả **HTTP 422** kèm JSON nêu rõ trường và lý do, ví dụ `{"field": "clouds_all", "message": "Input should be less than or equal to 100"}`. Hệ thống **không tự sửa** — không tự đặt mây về 100, không tự sửa `clear` thành `Clear` — vì sửa âm thầm là che lỗi nhập. Server không crash: có test gửi liên tiếp 5 payload sai rồi xác nhận request hợp lệ vẫn trả 200.
*Gợi ý mở rộng:* nếu xóa file model, `/health` trả **503** kèm đúng tên file thiếu và lệnh cần chạy, **không** phải 500 mơ hồ, và **không** tự huấn luyện lại.

### Câu 33
**Nếu dự báo cho năm 2025, hệ thống có báo không?**
**Đáp án:** **Có, và nó báo rõ thay vì đoán.** Dataset chỉ phủ 2012-10-02 → 2018-09-30, nên API trả kèm `in_dataset_range: false` và cảnh báo rằng độ tin cậy ngoài phạm vi không được đảm bảo. Ngoài ra bảng lịch **State Fair** chỉ có năm **2012–2020**; nếu năm nằm ngoài, API trả cảnh báo `state_fair_calendar_unknown` và cho phép người dùng truyền `state_fair_start_date` để tính đúng — **không tự suy đoán** ngày khai mạc.
*Gợi ý mở rộng — câu GV hay đặt ra nhất về chính sách hậu xử lý:*

Trả lời theo thứ tự 4 bước:

1. **Phân biệt trước:** `predictor.raw_model_output` là **RAW MODEL** (Ridge trả về trực tiếp) —
   đó là metric của mô hình. `predicted_traffic_volume` là **DEPLOYED PREDICTOR** =
   `max(0, ·)` ∘ Ridge — là *wrapper phục vụ*, **không phải cùng một model metric**.
2. **Vì sao cắt:** `traffic_volume` là số xe/giờ nên không thể âm; min trên TRAIN = 0,00.
3. **Vì sao đây KHÔNG phải test-informed postprocessing:** chính sách được chốt trong
   `src/freeze_serving_policy.py` chỉ dựa trên **TRAIN + VALIDATION** (FINAL TEST không được đọc
   khi quyết định, có test kiểm tra ở mức mã nguồn). Trên VALIDATION 2017: MAE 272,12 → 269,55,
   R² 0,9548 → 0,9558. Cộng lập luận toán học không cần dữ liệu: với `y ≥ 0` thì
   `|y − max(0, ŷ)| ≤ |y − ŷ|` cho mọi `ŷ`, tức cắt về sàn **không bao giờ** làm tăng sai số ở
   bất kỳ dòng nào. Nếu một trong ba điều kiện D1–D3 không đạt, script tự đặt `policy = "none"`.
4. **Số liệu:** metric chính thức 259,73 là RAW. Số 257,54 là DEPLOYED, báo riêng, không trích
   dẫn như "hiệu năng mô hình". Nhóm **không** sửa gói đánh giá đã đóng băng cho khớp API.

---

## Câu bổ sung hay gặp (không tính vào 33 câu chính)

### Câu 34
**Tại sao Ridge lại kém ở giờ đêm? Có phải chỉ do 2018 không?**
**Đáp án:** **Không chỉ do 2018** — nhóm đã kiểm chứng lại trên 3 cửa sổ dev out-of-sample
(fold rolling-origin 2015, 2016, 2017) và thấy Ridge kém ở **toàn bộ 5 giờ đêm trong cả 3/3
cửa sổ**. Đây là **đặc tính của mô hình**.

**Giả thuyết:** mô hình cộng tuyến tính nên hiệu ứng tháng/thời tiết được cộng thêm với một
lượng **tuyệt đối** gần như không đổi ở mọi giờ. Nhưng thực tế tác động đó **tương đối**: 20 %
của 3.500 xe/giờ là 700 xe, còn 20 % của 400 xe/giờ chỉ 80 xe. Ở giờ đêm lưu lượng thấp, cùng
một hệ số tuyệt đối lại quá lớn.

**Kiểm chứng:** MAE **tương đối** (MAE / lưu lượng thực TB) ban đêm là 0,2674 với Ridge so với
0,1208 với baseline — gần gấp đôi; ban ngày hai bên gần nhau (0,0712 vs 0,0807). Nghĩa là vấn
đề **không** chỉ là hiệu ứng quy mô.

**Thử log-target (khám phá hậu nghiệm, KHÔNG thay mô hình chính):**

| Cửa sổ dev | Biến đích | MAE tổng | MAE giờ đêm |
| --- | --- | --- | --- |
| VALIDATION 2017 | tuyến tính | **272,12** | 160,81 |
| VALIDATION 2017 | log1p | 295,00 | **85,84** |
| pseudo-test 2016–2017 | tuyến tính | **306,48** | 176,97 |
| pseudo-test 2016–2017 | log1p | 320,09 | **93,52** |

→ Giả thuyết **được ủng hộ một nửa**: MAE giờ đêm giảm gần một nửa, đúng như dự đoán. Nhưng MAE
**tổng lại xấu hơn** vì `log1p` nén thang lưu lượng cao. Và nhóm **không** đổi mô hình chính vì
hai lý do: (1) giả thuyết này nêu ra **sau khi đã nhìn 2018** → chọn log-target lúc này là
test-informed model selection; (2) cần một quyết định dài hạn về ưu tiên vận hành mà dữ liệu
hiện có không đủ để đưa ra. Muốn dùng thì phải đánh giá lại trên **holdout mới**.

**Hướng phát triển:** tương tác `giờ × thời tiết` (hệ số thời tiết khác nhau theo khung giờ —
sát giả thuyết hơn log-target và không phá thang cao), hoặc `HistGradientBoosting` tự học tương
tác. **Không** đưa vào mô hình chính ở checkpoint này.

## Câu bổ sung dạng bảng

| Câu | Trả lời ngắn |
| --- | --- |
| Vì sao dùng Ridge mà không dùng mô hình cây? | Đề tài thiên về minh hoạ **phương pháp đánh giá**. Ridge dễ giải thích, chịu đa cộng tuyến, và cho phép so sánh ý nghĩa từng nhóm feature. Gradient boosting sẽ có MAE thấp hơn nhưng làm lệch trọng tâm bài toán. |
| Vì sao dùng `solver="lsqr"`? | Ma trận sau one-hot rộng (217 cột) nhưng mẫu chỉ 25.329; `lsqr` ổn định, nhanh, tránh lập phân rã ma trận. |
| Có thể dùng mô hình này để điều khiển tín hiệu giao thông không? | **Không** — tuyệt đối không dùng cho mục đích safety-critical. Sai số ở ngày lễ gấp 4,3 lần, mẫu chỉ 7 ngày lễ; chỉ có một trạm; dữ liệu là lịch sử 2012–2018. |
| Tại sao không nội suy 22,79 % số giờ thiếu? | Nội suy là học thống kê từ hàng xóm, mà hàng xóm có thể nằm ở tập test → rò rỉ. Thiếu dữ liệu được báo cáo như hạn chế thay vì che bằng thống kê học từ tập lớn. |
| Hạn chế lớn nhất của nghiên cứu này? | Chỉ có một trạm đo, một hướng đi, dữ liệu lịch sử 2012–2018, và test 2018 chỉ 9 tháng. Kết luận **không** suy rộng được ra toàn thành phố. |
| 343 test đảm bảo điều gì? | Đảm bảo hành vi, không đảm bảo độ đúng của mô hình. Đáng chú ý nhất là test chứng minh **không** train-serving skew và test chứng minh pipeline **không bị refit** lúc phục vụ. |
| Nếu được làm lại, nhóm sẽ đổi gì? | (1) Thêm lag feature nhưng **backtest nghiêm** trên nhiều kỳ; (2) thử mô hình phi tuyến nhưng giữ nguyên time split; (3) mô hình riêng cho ngày lễ — dù có nguy cơ overfit vì chỉ 7 ngày lễ trong 2018; (4) dự báo xác suất (quantile) để có khoảng tin cậy cho lập kế hoạch. |
| Nếu bị hỏi "có chắc 2018 chỉ chạy đúng 1 lần không"? | Nói thẳng: **không khẳng định điều đó**. Phát biểu đúng là: **năm 2018 không tham gia hyperparameter tuning hoặc model selection; pipeline và serving policy được đóng băng từ dữ liệu 2012–2017; kết quả 2018 không được dùng để tiếp tục tối ưu mô hình.** Điều cần chứng minh là **không có vòng lặp tối ưu nào đi qua 2018** — và điều đó có `assert_no_final_test_rows()` bảo chứng. |

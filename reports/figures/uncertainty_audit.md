# KIỂM TOÁN BẤT ĐỊNH — so sánh Ridge vs baseline

> Sinh tự động bởi `python src/uncertainty_audit.py` — **chạy SAU**
> `python src/evaluate.py`.

> **Script này KHÔNG quyết định gì.** Nó chỉ đo độ bất định quanh những con số
> đã có. Không ghi đè `evaluation_results.json`, không ghi vào `models/`,
> không train lại mô hình đã đóng băng.

- Số lần lấy mẫu lại: **4000** · đơn vị khối: **calendar_day** · seed: **20240501** · mức tin cậy: **95.0 %**
- Metric: **RAW MODEL (Ridge trả về trực tiếp)**
- Quy ước hiệu: **baseline − Ridge**; dương = Ridge tốt hơn.

## 1. FINAL TEST 2018 — hiệu MAE và hiệu RMSE

- Cửa sổ: 2018-01-01 00:00:00 → 2018-09-30 23:00:00, n = 6.533, 273 khối ngày lịch
- Baseline: MAE 272.9 · Ridge: MAE 259.73

| Chỉ số | Ước lượng | CI 95 % | Độ lệch bootstrap | % lần lấy mẫu Ridge thắng | CI có chứa 0? |
| --- | --- | --- | --- | --- | --- |
| Hiệu MAE | +13.16 | [3.8; 22.0] | 4.76 | 99.58 % | **không** |
| Hiệu RMSE | +56.36 | [31.07; 80.63] | 12.53 | 100.0 % | **không** |

## 2. MAE theo từng tháng của 2018

| Tháng | n | MAE baseline | MAE Ridge | Hiệu MAE | Ridge thắng |
| --- | --- | --- | --- | --- | --- |
| Jan | 742 | 374.01 | 358.18 | +15.82 | ✓ |
| Feb | 672 | 255.91 | 240.49 | +15.43 | ✓ |
| Mar | 733 | 281.57 | 259.75 | +21.82 | ✓ |
| Apr | 720 | 388.21 | 338.82 | +49.39 | ✓ |
| May | 743 | 261.27 | 243.43 | +17.84 | ✓ |
| Jun | 719 | 205.61 | 194.61 | +11.0 | ✓ |
| Jul | 744 | 243.18 | 240.93 | +2.25 | ✓ |
| Aug | 740 | 195.87 | 214.65 | -18.77 | ✗ |
| Sep | 720 | 249.44 | 244.75 | +4.69 | ✓ |

- **Ridge thắng ở 8/9 tháng** của FINAL TEST 2018.

## 3. Có nhất quán qua các năm khác không? (chỉ dữ liệu dev 2012–2017)

| Cửa sổ | n | MAE baseline | MAE Ridge | Hiệu MAE | CI 95 % | % Ridge thắng | Tháng Ridge thắng |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pseudo-test 2016-2017 (train <= 2015) | 16.551 | 294.95 | 306.48 | -11.53 | [-19.72; -3.79] | 0.18 % | 4/12 |
| fold 1 — test year 2015 | 3.593 | 282.86 | 325.98 | -43.13 | [-60.91; -25.92] | 0.0 % | 0/7 |
| fold 2 — test year 2016 | 7.838 | 317.29 | 343.77 | -26.48 | [-38.32; -14.72] | 0.0 % | 5/12 |
| fold 3 — test year 2017 | 8.713 | 278.66 | 272.12 | +6.54 | [-4.5; 16.5] | 88.75 % | 8/12 |

## 4. Kết luận — sinh từ số liệu trên

- **Trên 2018:** Ridge tốt hơn baseline trên 2018, và khoảng tin cậy 95 % của hiệu MAE không chứa 0.
- **Số tháng Ridge thắng:** 8/9 · **số cửa sổ dev mà Ridge thắng:** 1/4 · **chiều nhất quán qua các cửa sổ dev:** False
- **Kết luận tổng:** Trên FINAL TEST 2018, Ridge vượt baseline 13.16 MAE (CI 95 % [3.8; 22.0], không chứa 0) và thắng ở 8/9 tháng. Trên dữ liệu dev 2012–2017, Ridge thắng ở 1/4 cửa sổ (pseudo-test 2016–2017 và 3 fold rolling-origin), nên chiều **không nhất quán** giữa các cửa sổ — vì vậy không được nói chung 'mô hình vượt baseline' một cách tuyệt đối.

**Điều kiện đi kèm bắt buộc khi trích dẫn kết luận này:**
- Kết luận này chỉ nói về FINAL TEST 2018 (01/01–30/09), không suy rộng ra năm khác.
- Nó là so sánh trên cùng một tập đánh giá và cùng một tập huấn luyện 2012–2016.
- Metric là RAW MODEL; DEPLOYED PREDICTOR (max(0,·)) được báo riêng ở postprocess_audit.

![Bootstrap và MAE theo tháng](reports/figures/uncertainty_bootstrap.png)

*Hình — nguồn: `src/uncertainty_audit.py`.*

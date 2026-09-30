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

## 3. Mô hình kém ở giờ nào trên 2018? (mô tả, không quyết định)

- Ridge kém hơn baseline ở **7/24 giờ**.
- Giờ ban đêm [0, 1, 2, 3, 4]: MAE Ridge 151.45 so với baseline 68.39 (n = 1352).
- **MAE tương đối** (MAE / lưu lượng thực TB) ban đêm: Ridge 0.2674 so với baseline 0.1208; ban ngày: Ridge 0.0712 so với baseline 0.0807.

| Giờ | n | Lưu lượng thực TB | MAE Ridge | MAE baseline | MAE tương đối R / B |
| --- | --- | --- | --- | --- | --- |
| 00:00 | 273 | 830.62 | 176.39 | 110.72 | 0.2124 / 0.1333 |
| 01:00 | 273 | 505.97 | 160.99 | 69.01 | 0.3182 / 0.1364 |
| 02:00 | 264 | 380.31 | 157.06 | 52.95 | 0.413 / 0.1392 |
| 03:00 | 270 | 371.57 | 142.14 | 31.14 | 0.3825 / 0.0838 |
| 04:00 | 272 | 735.43 | 120.63 | 77.22 | 0.164 / 0.105 |
| 05:00 | 272 | 2211.31 | 197.44 | 240.62 | 0.0893 / 0.1088 |
| 06:00 | 272 | 4166.76 | 260.24 | 328.21 | 0.0625 / 0.0788 |
| 07:00 | 271 | 4813.29 | 355.75 | 422.46 | 0.0739 / 0.0878 |
| 08:00 | 272 | 4629.79 | 376.07 | 434.41 | 0.0812 / 0.0938 |
| 09:00 | 272 | 4434.17 | 300.01 | 345.74 | 0.0677 / 0.078 |
| 10:00 | 273 | 4234.76 | 218.97 | 274.46 | 0.0517 / 0.0648 |
| 11:00 | 273 | 4505.28 | 232.58 | 282.56 | 0.0516 / 0.0627 |
| 12:00 | 273 | 4746.04 | 235.93 | 277.28 | 0.0497 / 0.0584 |
| 13:00 | 273 | 4722.25 | 248.65 | 268.41 | 0.0527 / 0.0568 |
| 14:00 | 273 | 4905.73 | 275.51 | 283.75 | 0.0562 / 0.0578 |
| 15:00 | 273 | 5287.31 | 282.25 | 342.04 | 0.0534 / 0.0647 |
| 16:00 | 273 | 5793.82 | 408.8 | 485.32 | 0.0706 / 0.0838 |
| 17:00 | 273 | 5329.43 | 373.38 | 422.64 | 0.0701 / 0.0793 |
| 18:00 | 273 | 4285.79 | 276.55 | 327.25 | 0.0645 / 0.0764 |
| 19:00 | 273 | 3345.84 | 240.94 | 295.68 | 0.072 / 0.0884 |
| 20:00 | 273 | 2939.14 | 242.32 | 298.71 | 0.0824 / 0.1016 |
| 21:00 | 273 | 2732.79 | 289.47 | 290.31 | 0.1059 / 0.1062 |
| 22:00 | 273 | 2233.71 | 348.61 | 323.83 | 0.1561 / 0.145 |
| 23:00 | 273 | 1512.34 | 308.77 | 256.17 | 0.2042 / 0.1694 |

> Phần **kiểm chứng trên dữ liệu dev 2012–2017** (các fold rolling-origin) và phần thử log-target nằm ở `experiments_report.md` mục 4 và 5 — vì 2018 không được dùng để quyết định bất cứ điều gì.

## 4. Có nhất quán qua các năm khác không? (chỉ dữ liệu dev 2012–2017)

| Cửa sổ | n | MAE baseline | MAE Ridge | Hiệu MAE | CI 95 % | % Ridge thắng | Tháng Ridge thắng |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pseudo-test 2016-2017 (train <= 2015) | 16.551 | 294.95 | 306.48 | -11.53 | [-19.72; -3.79] | 0.18 % | 4/12 |
| fold 1 — test year 2015 | 3.593 | 282.86 | 325.98 | -43.13 | [-60.91; -25.92] | 0.0 % | 0/7 |
| fold 2 — test year 2016 | 7.838 | 317.29 | 343.77 | -26.48 | [-38.32; -14.72] | 0.0 % | 5/12 |
| fold 3 — test year 2017 | 8.713 | 278.66 | 272.12 | +6.54 | [-4.5; 16.5] | 88.75 % | 8/12 |

## 5. Kết luận — sinh từ số liệu trên

- **Trên 2018:** Ridge tốt hơn baseline trên 2018, và khoảng tin cậy 95 % của hiệu MAE không chứa 0.
- **Số tháng Ridge thắng:** 8/9 · **số cửa sổ dev mà Ridge thắng:** 1/4 · **chiều nhất quán qua các cửa sổ dev:** False
- **Kết luận tổng:** Trên FINAL TEST 2018, Ridge vượt baseline 13.16 MAE (CI 95 % [3.8; 22.0], không chứa 0) và thắng ở 8/9 tháng. Trên dữ liệu dev 2012–2017, Ridge thắng ở 1/4 cửa sổ (pseudo-test 2016–2017 và 3 fold rolling-origin), nên chiều **không nhất quán** giữa các cửa sổ — vì vậy không được nói chung 'mô hình vượt baseline' một cách tuyệt đối.

**Điều kiện đi kèm bắt buộc khi trích dẫn kết luận này:**
- Kết luận này chỉ nói về FINAL TEST 2018 (01/01–30/09), không suy rộng ra năm khác.
- Nó là so sánh trên cùng một tập đánh giá và cùng một tập huấn luyện 2012–2016.
- Metric là RAW MODEL; DEPLOYED PREDICTOR (max(0,·)) được báo riêng ở postprocess_audit.

![Bootstrap và MAE theo tháng](reports/figures/uncertainty_bootstrap.png)

*Hình — nguồn: `src/uncertainty_audit.py`.*

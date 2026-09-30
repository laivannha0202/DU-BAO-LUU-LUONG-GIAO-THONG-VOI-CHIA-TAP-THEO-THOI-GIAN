# FINAL EVALUATION — TEST 2018 (đánh giá đúng MỘT LẦN)

> Sinh tự động bởi `python src/evaluate.py`. **Không con số nào được hard-code.**
> Mọi số liệu tính lại từ dữ liệu sau khi sửa toàn bộ preprocessing.

## 0. Cấu hình đã đóng băng TRƯỚC khi đánh giá 2018

- Mô hình: Ridge, **alpha = 0.001** (chọn theo MAE trên VALIDATION 2017)
- Solver: `lsqr`
- Pipeline: `ColumnTransformer[OneHotEncoder | SimpleImputer(median)→StandardScaler | passthrough] → Ridge`
- Imputer / scaler / encoder: **fit trên TRAIN 2012–2016 duy nhất**
- Baseline: mean traffic theo (hour, day_of_week), **fit trên TRAIN duy nhất**
- `is_holiday`: tính từ **lịch tất định** `src/holidays.py` (không phụ thuộc dữ liệu)

| Tập | Khoảng thời gian | Số dòng |
| --- | --- | --- |
| train | 2012-10-02 09:00:00 → 2016-12-31 23:00:00 | 25.329 |
| validation | 2017-01-01 00:00:00 → 2017-12-31 23:00:00 | 8.713 |
| test | 2018-01-01 00:00:00 → 2018-09-30 23:00:00 | 6.533 |

Assert: `max(train) < min(validation)` ✅ · `max(validation) < min(test)` ✅ · 0 timestamp trùng ✅

> **Bằng chứng 2018 chưa bị dùng để lựa chọn gì:** toàn bộ tuning alpha và các thí nghiệm phát triển (random vs time split, kiểm soát rò rỉ, rolling-origin drift) đều chạy trong `src/train.py` và `src/experiments.py`, với cửa sổ dữ liệu **2012–2017**. `src/experiments.py` gọi `assert_no_final_test_rows()` ở mọi hàm và sẽ dừng chương trình nếu bất kỳ dòng 2018 nào lọt vào.

## 1. Baseline vs Model

| Mô hình | MAE | RMSE | R² | n |
| --- | --- | --- | --- | --- |
| Baseline (hour × day_of_week) | 272.9 | 473.13 | 0.9426 | 6.533 |
| **Ridge pipeline** | **259.73** | **416.78** | **0.9554** | 6.533 |

- Ridge cải thiện **13.17** MAE (4.8%) so với baseline.

## 2. MAE / RMSE / R²

| Tập | Đánh giá | MAE | RMSE | R² | n |
| --- | --- | --- | --- | --- | --- |
| VALIDATION 2017 (dùng để chọn alpha) | out-of-sample | 272.12 | 421.46 | 0.9548 | 8.713 |
| FINAL TEST 2018 | out-of-sample | **259.73** | **416.78** | **0.9554** | 6.533 |
| FINAL TEST 2018 (baseline) | out-of-sample | 272.9 | 473.13 | 0.9426 | 6.533 |

- ⚠️ Không đưa MAE trên tập train vào bảng này: train là **in-sample** còn hai dòng trên là **out-of-sample**, trộn chúng là so sánh không cùng đối tượng. Phân tích drift riêng nằm ở `experiments_report.md` (rolling-origin, chỉ out-of-sample).

## 3. Phân tích lỗi trên FINAL TEST

> Mọi phân khúc đều kèm **MAE + số mẫu**. Ngưỡng cảnh báo mẫu nhỏ: n < 100.

### 3a. Theo giờ

| Phân khúc | MAE (Ridge) | n mẫu | MAE (baseline) | Lưu lượng thực | Dự báo TB | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| hour_00 | 176.39 | 273 | 110.72 | 830.62 | 864.91 | mẫu đủ |
| hour_01 | 160.99 | 273 | 69.01 | 505.97 | 550.01 | mẫu đủ |
| hour_02 | 157.06 | 264 | 52.95 | 380.31 | 432.24 | mẫu đủ |
| hour_03 | 142.14 | 270 | 31.14 | 371.57 | 407.84 | mẫu đủ |
| hour_04 | 120.63 | 272 | 77.22 | 735.43 | 730.79 | mẫu đủ |
| hour_05 | 197.44 | 272 | 240.62 | 2211.31 | 2118.64 | mẫu đủ |
| hour_06 | 260.24 | 272 | 328.21 | 4166.76 | 4186.51 | mẫu đủ |
| hour_07 | 355.75 | 271 | 422.46 | 4813.29 | 4810.83 | mẫu đủ |
| hour_08 | 376.07 | 272 | 434.41 | 4629.79 | 4609.56 | mẫu đủ |
| hour_09 | 300.01 | 272 | 345.74 | 4434.17 | 4379.92 | mẫu đủ |
| hour_10 | 218.97 | 273 | 274.46 | 4234.76 | 4207.02 | mẫu đủ |
| hour_11 | 232.58 | 273 | 282.56 | 4505.28 | 4495.5 | mẫu đủ |
| hour_12 | 235.93 | 273 | 277.28 | 4746.04 | 4741.25 | mẫu đủ |
| hour_13 | 248.65 | 273 | 268.41 | 4722.25 | 4761.2 | mẫu đủ |
| hour_14 | 275.51 | 273 | 283.75 | 4905.73 | 4986.87 | mẫu đủ |
| hour_15 | 282.25 | 273 | 342.04 | 5287.31 | 5268.2 | mẫu đủ |
| hour_16 | 408.8 | 273 | 485.32 | 5793.82 | 5678.46 | mẫu đủ |
| hour_17 | 373.38 | 273 | 422.64 | 5329.43 | 5349.94 | mẫu đủ |
| hour_18 | 276.55 | 273 | 327.25 | 4285.79 | 4294.09 | mẫu đủ |
| hour_19 | 240.94 | 273 | 295.68 | 3345.84 | 3293.88 | mẫu đủ |
| hour_20 | 242.32 | 273 | 298.71 | 2939.14 | 2849.78 | mẫu đủ |
| hour_21 | 289.47 | 273 | 290.31 | 2732.79 | 2688.41 | mẫu đủ |
| hour_22 | 348.61 | 273 | 323.83 | 2233.71 | 2213.77 | mẫu đủ |
| hour_23 | 308.77 | 273 | 256.17 | 1512.34 | 1494.66 | mẫu đủ |

- Giờ khó nhất: **hour_16** (MAE 408.8, n=273); giờ dễ nhất: **hour_04** (MAE 120.63, n=272).

### 3b. Theo thứ trong tuần

| Phân khúc | MAE (Ridge) | n mẫu | MAE (baseline) | Lưu lượng thực | Dự báo TB | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| Mon | 347.28 | 936 | 390.36 | 3260.87 | 3341.28 | mẫu đủ |
| Tue | 226.68 | 933 | 239.49 | 3542.38 | 3575.78 | mẫu đủ |
| Wed | 210.36 | 935 | 224.95 | 3630.01 | 3633.51 | mẫu đủ |
| Thu | 230.6 | 932 | 234.97 | 3729.58 | 3703.6 | mẫu đủ |
| Fri | 221.27 | 935 | 236.16 | 3772.11 | 3713.24 | mẫu đủ |
| Sat | 321.27 | 927 | 322.4 | 2879.52 | 2820.68 | mẫu đủ |
| Sun | 260.93 | 935 | 262.05 | 2450.87 | 2406.38 | mẫu đủ |

- Ngày khó nhất: **Mon** (MAE 347.28, n=936).

### 3c. Ngày lễ

| Phân khúc | MAE (Ridge) | n mẫu | MAE (baseline) | Lưu lượng thực | Dự báo TB | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| non_holiday | 239.49 | 6.366 | 250.58 | 3346.74 | 3334.88 | mẫu đủ |
| holiday | 1031.44 | 167 | 1123.43 | 2453.41 | 2512.09 | mẫu đủ |

- Chênh lệch MAE (ngày lễ − ngày thường) = **791.95** xe/giờ; ngày lễ chiếm 167 giờ trên 7 ngày lịch.

### 3d. Thời tiết (weather_main, multi-hot)

| Phân khúc | MAE (Ridge) | n mẫu | MAE (baseline) | Lưu lượng thực | Dự báo TB | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| Clear | 245.15 | 2.328 | 256.64 | 2989.63 | 2966.36 | Mẫu đủ. |
| Clouds | 243.39 | 1.996 | 254.11 | 3924.41 | 3867.11 | Mẫu đủ. |
| Drizzle | 207.16 | 278 | 211.26 | 3569.06 | 3600.56 | Mẫu đủ. |
| Fog | 536.05 | 192 | 640.17 | 2224.91 | 2540.66 | Mẫu đủ. |
| Haze | 313.56 | 272 | 332.9 | 3213.19 | 3216.78 | Mẫu đủ. |
| Mist | 295.41 | 1.099 | 319.87 | 2840.57 | 2920.02 | Mẫu đủ. |
| Rain | 246.88 | 992 | 254.91 | 3391.65 | 3406.48 | Mẫu đủ. |
| Smoke | 137.93 | 2 | 88.11 | 4443.0 | 4580.93 | Mẫu nhỏ — thận trọng. |
| Snow | 524.5 | 521 | 585.36 | 2800.6 | 3039.48 | Mẫu đủ. |
| Squall | — | 0 | — | — | — | KHÔNG CÓ MẪU — không đánh giá được. |
| Thunderstorm | 266.76 | 268 | 269.18 | 3246.82 | 3326.57 | Mẫu đủ. |

![MAE theo weather](reports/figures/final_mae_by_weather.png)

### 3e. Thời tiết cực đoan

| Phân khúc | MAE (Ridge) | n mẫu | MAE (baseline) | Lưu lượng thực | Dự báo TB | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| Snow | 524.5 | 521 | 585.36 | 2800.6 | 3039.48 | Mẫu đủ. |
| Thunderstorm | 266.76 | 268 | 269.18 | 3246.82 | 3326.57 | Mẫu đủ. |
| Squall | — | 0 | — | — | — | KHÔNG CÓ MẪU trong FINAL TEST — không đánh giá được. |
| any_extreme_weather | 434.08 | 784 | 475.09 | 2958.81 | 3139.75 | mẫu đủ |
| no_extreme_weather | 235.96 | 5.749 | 245.32 | 3373.69 | 3337.59 | mẫu đủ |

- Các phân khúc dùng multi-hot nên Snow/Thunderstorm/Squall có thể trùng nhau; tổng n có thể vượt số dòng của FINAL TEST.
- ⚠️ **KẾT LUẬN YẾU:** `Squall` không có mẫu nào trong FINAL TEST. Không phát biểu mạnh về các phân khúc này.

### 3f. Theo giờ × thứ

![MAE theo giờ và thứ](reports/figures/final_mae_by_hour_dow.png)

## 4. Kết luận

1. FINAL TEST 2018 (n=6.533): Ridge đạt **MAE=259.73, RMSE=416.78, R²=0.9554**, so với baseline MAE=272.9.
2. Điểm yếu rõ nhất là **ngày lễ** (MAE=1031.44 so với 239.49 ở ngày thường, n=167). `is_holiday` chỉ là cờ nhị phân nên mô hình chỉ học được một mức dịch chuyển trung bình, không học được dạng hình giờ đặc thù của ngày lễ. Hướng sửa hợp lý cho checkpoint sau: thêm tương tác `hour × is_holiday`.
3. Thời tiết cực đoan tổng hợp: MAE=434.08 (n=784) so với 235.96 ở thời tiết bình thường (n=5.749).
4. Giờ khó nhất: **hour_16** (MAE 408.8, n=273); ngày khó nhất: **Mon** (MAE 347.28, n=936).
5. Mọi phân khúc đều báo kèm số mẫu; phân khúc không có mẫu được đánh dấu rõ và không dùng để kết luận.
6. **Đối chiếu với giai đoạn phát triển:** trên pseudo-test 2016–2017 (train chỉ tới 2015, dữ liệu rất thưa) baseline thắng Ridge về MAE nhưng thua về RMSE. Trên FINAL TEST này (train tới 2016, dữ liệu dày hơn nhiều) **Ridge thắng ở cả ba chỉ số**. Phần đuôi hơn của Ridge vì vậy phụ thuộc vào mật độ dữ liệu huấn luyện — và là lý do không nên kết luận chỉ từ một cửa sổ đánh giá duy nhất.

# EDA — CHỈ TRÊN TẬP TRAIN (2012-2016)

> Sinh tự động bởi `python src/eda.py`. Mọi thống kê dưới đây chỉ dùng dữ liệu TRAIN. Validation (2017) và Test (2018) không được nhìn vào trước khi cấu hình được đóng băng.

## 1. Quy mô các tập (thống kê tối thiểu, chỉ để ghi nhận ranh giới thời gian)

| Tập | Khoảng thời gian | Số dòng | Năm |
| --- | --- | --- | --- |
| TRAIN | 2012-10-02 09:00:00 → 2016-12-31 23:00:00 | 25.329 | 2012, 2013, 2014, 2015, 2016 |
| VALIDATION | 2017-01-01 00:00:00 → 2017-12-31 23:00:00 | 8.713 | 2017 |
| TEST | 2018-01-01 00:00:00 → 2018-09-30 23:00:00 | 6.533 | 2018 |

> Dòng trên chỉ ghi nhận mốc thời gian/số dòng để chứng minh split không chồng; không có phân bố nào của validation/test được tính ở đây.

## 2. Target `traffic_volume` trên TRAIN

- Số quan sát: **25.329**
- min = 0 | max = 7.260 | mean = 3.252.51 | std = 1.987.14
- median = 3.339 | p05 = 336 | p95 = 6.199
- Số giờ có lưu lượng = 0: 2

![Phân bố target](reports/figures/eda_target_distribution.png)

### Theo giờ × thứ (trung bình)

![Lưu lượng theo giờ x thứ](reports/figures/eda_traffic_by_hour_dow.png)

| Giờ | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 617 | 590 | 615 | 641 | 759 | 1.222 | 1.335 |
| 01 | 375 | 372 | 389 | 389 | 457 | 761 | 835 |
| 02 | 286 | 281 | 303 | 295 | 345 | 597 | 629 |
| 03 | 332 | 353 | 370 | 360 | 387 | 407 | 375 |
| 04 | 790 | 829 | 846 | 819 | 770 | 412 | 324 |
| 05 | 2.548 | 2.722 | 2.755 | 2.725 | 2.470 | 758 | 516 |
| 06 | 5.158 | 5.503 | 5.570 | 5.397 | 5.119 | 1.303 | 867 |
| 07 | 5.858 | 6.132 | 6.182 | 6.119 | 5.938 | 1.950 | 1.201 |
| 08 | 5.229 | 5.581 | 5.767 | 5.512 | 5.285 | 2.779 | 1.857 |
| 09 | 4.561 | 4.985 | 5.022 | 4.995 | 4.785 | 3.477 | 2.652 |
| 10 | 4.106 | 4.344 | 4.480 | 4.452 | 4.528 | 3.898 | 3.395 |
| 11 | 4.323 | 4.564 | 4.674 | 4.761 | 4.952 | 4.270 | 3.711 |
| 12 | 4.573 | 4.778 | 4.863 | 4.939 | 5.195 | 4.561 | 4.077 |
| 13 | 4.617 | 4.807 | 4.887 | 4.945 | 5.158 | 4.501 | 4.172 |
| 14 | 4.938 | 5.141 | 5.223 | 5.298 | 5.436 | 4.487 | 4.141 |
| 15 | 5.345 | 5.628 | 5.664 | 5.698 | 5.732 | 4.455 | 4.175 |
| 16 | 5.944 | 6.332 | 6.343 | 6.236 | 5.991 | 4.446 | 4.183 |
| 17 | 5.624 | 5.966 | 6.027 | 5.889 | 5.511 | 4.271 | 3.951 |
| 18 | 4.131 | 4.455 | 4.548 | 4.545 | 4.621 | 3.961 | 3.609 |
| 19 | 3.022 | 3.189 | 3.286 | 3.358 | 3.586 | 3.294 | 3.097 |
| 20 | 2.563 | 2.792 | 2.868 | 2.963 | 2.976 | 2.860 | 2.683 |
| 21 | 2.269 | 2.619 | 2.612 | 2.896 | 2.934 | 2.939 | 2.350 |
| 22 | 1.673 | 1.977 | 1.882 | 2.329 | 2.655 | 2.956 | 1.790 |
| 23 | 1.102 | 1.267 | 1.230 | 1.395 | 1.881 | 2.103 | 1.197 |

- Hình dạng lưu lượng rất rõ: hai đỉnh buổi sáng (~08:00) và buổi tối (~17:00), đường cong ngày làm việc khác hẳn cuối tuần. Đây là lý do `hour_dow` là feature chính và cũng chính là cấu trúc mà baseline khai thác.

### Theo tháng × năm

![Lưu lượng theo tháng x năm](reports/figures/eda_traffic_by_month_year.png)

| Năm | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | - | - | - | - | - | - | - | - | - | 3.456 | 3.263 | 2.983 |
| 2013 | 3.111 | 3.230 | 3.261 | 3.383 | 3.335 | 3.361 | 3.406 | 3.569 | 3.320 | 3.659 | 3.282 | 2.948 |
| 2014 | 2.931 | 3.072 | 3.284 | 3.308 | 3.426 | 3.412 | 3.470 | 3.810 | - | - | - | - |
| 2015 | - | - | - | - | - | 3.302 | 3.222 | 3.361 | 3.349 | 3.303 | 3.116 | 3.076 |
| 2016 | 3.011 | 3.196 | 3.400 | 3.480 | 3.329 | 3.481 | 2.823 | 3.227 | 3.183 | 3.162 | 3.021 | 3.078 |

## 3. Ngày lễ trên TRAIN

- Số giờ thuộc ngày lễ: **826** (3.26% của TRAIN)
- Số ngày lễ: **38**
- Lưu lượng trung bình ngày thường: 3.276
- Lưu lượng trung bình ngày lễ: 2.542

![Tác động ngày lễ](reports/figures/eda_holiday_effect.png)

## 4. Thời tiết trên TRAIN

![Phân bố weather_main](reports/figures/eda_weather_distribution.png)

| weather_main | Số timestamp (multi-hot) | Tỉ lệ |
| --- | --- | --- |
| Clear | 8.013 | 31.64% |
| Clouds | 10.118 | 39.95% |
| Drizzle | 889 | 3.51% |
| Fog | 481 | 1.90% |
| Haze | 739 | 2.92% |
| Mist | 3.471 | 13.70% |
| Rain | 3.449 | 13.62% |
| Smoke | 17 | 0.07% |
| Snow | 1.695 | 6.69% |
| Squall | 4 | 0.02% |
| Thunderstorm | 474 | 1.87% |

- Đây là phân bố **multi-hot** (một giờ có thể thuộc nhiều hiện tượng), nên tổng các tỉ lệ có thể vượt 100%. Không dùng được `drop_duplicates(keep='first')` ở đây — nó sẽ chỉ giữ đúng 1 hiện tượng đầu tiên và làm mất phần còn lại.

![Phân bố biến thời tiết số](reports/figures/eda_weather_numeric.png)

| Biến | n NaN trong TRAIN | Ghi chú |
| --- | --- | --- |
| `temp` | 10 | sẽ do SimpleImputer (median, fit TRAIN) điền |
| `rain_1h` | 1 | sẽ do SimpleImputer (median, fit TRAIN) điền |
| `snow_1h` | 0 | không thiếu |
| `clouds_all` | 0 | không thiếu |

## 5. Mật độ quan sát theo năm (trên TRAIN) — cảnh báo về khoảng trống

| Năm | Số giờ có dữ liệu | Số giờ lẽ ra có (nếu đủ 8760/8784) | Tỉ lệ phủ |
| --- | --- | --- | --- |
| 2012 | 2.103 | 8.784 | 23.9% |
| 2013 | 7.294 | 8.760 | 83.3% |
| 2014 | 4.501 | 8.760 | 51.4% |
| 2015 | 3.593 | 8.760 | 41.0% |
| 2016 | 7.838 | 8.784 | 89.2% |

- Năm 2015 đặc biệt thưa (chỉ từ 2015-06-11). Vì vậy kết luận 'mô hình generalize tốt qua các năm' phải dựa vào VALIDATION 2017 và TEST 2018, không dựa vào việc khớp tốt trên TRAIN.

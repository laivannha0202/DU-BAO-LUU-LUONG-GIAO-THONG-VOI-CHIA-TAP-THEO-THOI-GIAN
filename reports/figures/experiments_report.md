# Thí nghiệm phát triển — cửa sổ 2012–2017 (FINAL TEST 2018 CHƯA BỊ CHẠM)

> Sinh tự động bởi `python src/experiments.py`.

- Cửa sổ phát triển: 2012-10-02 09:00:00 → 2017-12-31 23:00:00 — **34.042** dòng
- 2018 (FINAL TEST) — chưa được chạm vào ở giai đoạn này
- Mọi hàm trong module này đều gọi `assert_no_final_test_rows()`; nếu một dòng 2018 lọt vào thì chương trình dừng ngay với `FinalTestLeakError`.

## 0. Baseline vs Ridge trên pseudo-test 2016–2017

- Train: 2012-10-02 09:00:00 → 2015-12-31 23:00:00
- Pseudo-test: 2016-01-01 00:00:00 → 2017-12-31 23:00:00

| Mô hình | MAE | RMSE | R² | n |
| --- | --- | --- | --- | --- |
| baseline | 294.95 | 495.71 | 0.9364 | 16.551 |
| ridge | 306.48 | 472.61 | 0.9422 | 16.551 |

- ⚠️ **Hai chỉ số không đồng ý — báo cáo trung thực, không chọn lợi hơn:** so với baseline, Ridge có MAE -11.53 (KÉM HƠN) nhưng RMSE +23.10 (TỐT HƠN), và R² 0.9422 so với 0.9364 (cao hơn).

  - Cách đọc đúng: Ridge **giảm sai số lớn** (RMSE, R² tốt hơn) nhưng lại **tăng sai số ở các giờ bình thường** (MAE kém hơn). Baseline là trung bình theo hour × day_of-week — một đường bằng phẳng rất mượt, nên hiếm khi sai lệch lớn, nhưng cũng không bám được mức lưu lượng thực tế khi năm đổi mức.
  - Nguyên nhân hợp lý: tập train 2012–2015 có dữ liệu rất thưa (2014 kết thúc 08/08, 2015 bắt đầu 11/06), nên Ridge thiếu dữ liệu để học mức lưu lượng ổn định, trong khi baseline không cần học mức nào.
  - **Vì vậy không được dùng một chỉ số đơn lẻ để kết luận.** Báo cáo cuối sẽ trình bày MAE, RMSE và R² cùng lúc, và nêu rõ chỗ nào mô hình thắng, chỗ nào thua.

## 1. Thí nghiệm 1 — Random split vs Time split (CHỈ MINH HỌA)

| Cách chia | MAE | RMSE | R² | n test | Khoảng test |
| --- | --- | --- | --- | --- | --- |
| Time split | 306.48 | 472.61 | 0.9422 | 16.551 | 2016-01-01 → 2017-12-31 |
| Random split | 298.61 | 481.99 | 0.9415 | 5.107 | mẫu ngẫu nhiên rải rác 2012–2017 |

- Chênh lệch MAE (random − time) = **-7.87**
- **Không khẳng định trước** random split sẽ tốt hơn hay xấu hơn. Ở lần chạy này random split cho MAE **thấp hơn** time split.
- ⚠️ Hai tập test KHÁC thành phần năm (time split test = 2016-2017 còn nguyên; random split test = mẫu ngẫu nhiên rải rác 2012-2017), nên chênh lệch KHÔNG chứng minh được rò rỉ. Thí nghiệm 1b mới là phép so sánh công bằng.

## 1b. Thí nghiệm 1b — Kiểm soát rò rễ trên MỘT tập test cố định

- Tập test dùng CHUNG cho cả 3 arm: **4.357** giờ. 2017-01-01 → 2017-12-31
- Dùng CHUNG cho cả 3 arm — đây là điểm làm phép so sánh công bằng.

| Arm | Tập train kết thúc | Cách xa test | n train | MAE | RMSE | R² | n test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A: chỉ tới 2015 | 2015-12-31 | 2 năm | 17.491 | **270.32** | 421.3 | 0.955 | 4.357 |
| B: chỉ tới 2016 | 2016-12-31 | 1 năm | 25.329 | **269.32** | 420.73 | 0.9551 | 4.357 |
| C: tới 2016 + nửa 2017 | 2017-12-31 | 0 năm | 29.685 | **262.77** | 414.02 | 0.9565 | 4.357 |

- MAE giảm **7.55** điểm khi đưa train sát test hơn (A → C), trên **cùng một tập test**.
- Càng đưa dữ liệu sát thời điểm dự báo vào train, MAE càng giảm — nhưng phần giảm đó KHÔNG đến từ năng lực mô hình mà từ việc mô hình đã nhìn thấy 'hàng xóm' của chính dòng cần dự báo. Đây chính là rò rễ mà time split loại bỏ, và là lý do kết luận trên 2018 phải được công bố từ một mô hình chưa từng thấy năm 2018.

## 3. Thí nghiệm 3 — Rolling-origin evaluation (chỉ cửa sổ OUT-OF-SAMPLE)

- Thiết kế: expanding-window rolling origin; mọi fold đều out-of-sample
- ⚠️ **Không** dùng MAE trên tập train (in-sample) để kết luận drift: trộn in-sample với out-of-sample là so sánh không cùng đối tượng.

| Fold | Năm test | Khoảng train | Khoảng test | Loại đánh giá | MAE | RMSE | R² | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2015 | 2012-10-02 → 2014-08-08 | 2015-06-11 → 2015-12-31 | out_of_sample | **325.98** | 497.93 | 0.9362 | 3.593 |
| 2 | 2016 | 2012-10-02 → 2015-12-31 | 2016-01-01 → 2016-12-31 | out_of_sample | **343.77** | 523.66 | 0.9274 | 7.838 |
| 3 | 2017 | 2012-10-02 → 2016-12-31 | 2017-01-01 → 2017-12-31 | out_of_sample | **272.12** | 421.46 | 0.9548 | 8.713 |

- Chuỗi MAE theo năm test: [325.98, 343.77, 272.12] — MAE giảm 53.9 (16.5%) — chất lượng được cải thiện, có thể do năm gần nhất (2017) dễ hơn, không nhất thiết là mô hình tiến bộ.

### Gộp các cửa sổ out-of-sample (2015. 2016. 2017). n=20.144. MAE=309.61

| Mùa | MAE | n |
| --- | --- | --- |
| Autumn | 328.72 | 5.864 |
| Spring | 240.93 | 3.988 |
| Summer | 313.19 | 6.014 |
| Winter | 342.39 | 4.278 |

| Tháng | MAE | n |
| --- | --- | --- |
| Jan | 319.16 | 1.205 |
| Feb | 282.53 | 1.166 |
| Mar | 238.85 | 1.212 |
| Apr | 248.77 | 1.312 |
| May | 235.61 | 1.464 |
| Jun | 217.36 | 1.582 |
| Jul | 389.87 | 2.216 |
| Aug | 304.93 | 2.216 |
| Sep | 308.3 | 2.079 |
| Oct | 320.32 | 1.955 |
| Nov | 360.89 | 1.830 |
| Dec | 393.66 | 1.907 |

## 3b. Drift DỮ LIỆU (phân bố) — khác với drift hiệu năng ở trên

- Năm tham chiếu: **2013** (năm TRAIN đầy đủ nhất)
- PSI: < 0.1 ổn định, 0.1–0.25 dịch chuyển vừa, >= 0.25 dịch chuyển lớn

| Năm | mean traffic | median | std | n |
| --- | --- | --- | --- | --- |
| 2012 | 3226.7 | 3202.0 | 2005.6 | 2.103 |
| 2013 | 3309.55 | 3369.0 | 2030.71 | 7.294 |
| 2014 | 3270.14 | 3336.0 | 1992.65 | 4.501 |
| 2015 | 3258.04 | 3385.0 | 1970.84 | 3.593 |
| 2016 | 3193.7 | 3306.5 | 1943.76 | 7.838 |
| 2017 | 3376.59 | 3593.0 | 1982.58 | 8.713 |

| Biến | PSI 2012 | PSI 2013 | PSI 2014 | PSI 2015 | PSI 2016 | PSI 2017 |
| --- | --- | --- | --- | --- | --- | --- |
| `traffic_volume` | 0.021 | 0.0 | 0.0041 | 0.0298 | 0.0345 | 0.0196 |
| `temp` | 1.1035 | 0.0 | 0.0794 | 0.7389 | 0.1619 | 0.1027 |
| `clouds_all` | 0.3329 | 0.0 | 0.0593 | 0.698 | 0.0715 | 0.3532 |

- PSI đo **mức dịch chuyển phân bố** của dữ liệu đầu vào so với năm tham chiếu; nó không nói gì về chất lượng dự báo. Drift hiệu năng nằm ở mục 3, drift phân bố nằm ở đây.
- ⚠️ **Cảnh báo về cách đọc PSI:** các năm có độ phủ thời gian khác nhau sẽ cho PSI cao mà **không phải do drift thật**. 2012 chỉ có từ 10/02, 2014 kết thúc 08/08, 2015 bắt đầu 11/06 — tức thiếu hẳn một số mùa so với năm tham chiếu 2013 (đủ 12 tháng). PSI cao ở `temp` (2012 và 2015) và `clouds_all` (2015) phần lớn phản ánh **khác biệt về thành phần mùa trong mẫu**, không phải hệ thống đã đổi hành vi.
- Vì vậy chỉ nên so PSI giữa các năm có **độ phủ tương đương** (ví dụ 2013 vs 2017), hoặc chuẩn hoá theo tháng trước khi kết luận.

## Kết luận giai đoạn phát triển

1. Trên pseudo-test 2016–2017: ⚠️ Ridge **kém baseline về MAE** (306.48 vs 294.95) nhưng **tốt hơn về RMSE** (472.61 vs 495.71) — đã biết **trước khi** nhìn vào 2018.
2. Thí nghiệm 1b cho thấy đưa dữ liệu sát test hơn làm MAE giảm 7.55 điểm trên cùng tập test — cơ sở để giữ 2018 hoàn toàn nguyên vẹn.
3. Rolling-origin (chỉ out-of-sample): MAE giảm 53.9 (16.5%) — chất lượng được cải thiện, có thể do năm gần nhất (2017) dễ hơn, không nhất thiết là mô hình tiến bộ.
4. Cấu hình (alpha, feature, quy tắc tiền xử lý) đã được chốt. Bước kế tiếp là `python src/evaluate.py` — đánh giá FINAL TEST 2018 đúng một lần. **Sau khi đọc kết quả 2018, không được quay lại sửa mô hình.**

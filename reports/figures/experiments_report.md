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

## 1. Thí nghiệm 1 — Random split vs Time split

Lặp lại trên 5 seed cố định ([11, 23, 37, 53, 71]); báo cáo trung bình ± độ lệch chuẩn.

**Time split** (train ≤ 2015, test 2016-01-01 → 2017-12-31, n=16.551): MAE 306.48, RMSE 472.61, R² 0.9422

**Random split** (mỗi seed một tập test ngẫu nhiên rải rác 2012–2017):

| seed | n test | MAE | RMSE | R² | Thành phần năm của tập test |
| --- | --- | --- | --- | --- | --- |
| 11 | 5.107 | 296.75 | 479.64 | 0.9415 | 2012: 288. 2013: 1.148. 2014: 679. 2015: 515. 2016: 1.192. 2017: 1.285 |
| 23 | 5.107 | 292.09 | 475.82 | 0.943 | 2012: 306. 2013: 1.099. 2014: 708. 2015: 559. 2016: 1.164. 2017: 1.271 |
| 37 | 5.107 | 291.58 | 476.43 | 0.942 | 2012: 333. 2013: 1.105. 2014: 643. 2015: 535. 2016: 1.202. 2017: 1.289 |
| 53 | 5.107 | 284.94 | 456.88 | 0.9476 | 2012: 316. 2013: 1.062. 2014: 678. 2015: 545. 2016: 1.233. 2017: 1.273 |
| 71 | 5.107 | 293.11 | 469.48 | 0.9432 | 2012: 305. 2013: 1.070. 2014: 701. 2015: 565. 2016: 1.193. 2017: 1.273 |

### Câu hỏi nghiên cứu: hai cách đánh giá chênh nhau bao nhiêu?

| Phép đo | Trung bình ± độ lệch (MAE) | min | max |
| --- | --- | --- | --- |
| (a) Mỗi arm dùng tập test riêng | -14.79 ± 3.83 | -21.54 | -9.73 |
| (b) **Cùng một tập dòng đánh giá** | **-5.43 ± 0.81** | -6.72 | -4.65 |

- **(a)** là cách so sánh *tự nhiên* khi mỗi arm dùng tập test của chính nó. Dấu âm = random split trông **tốt hơn**.
- **(b)** là phép so sánh **công bằng về cách chọn tập huấn luyện**: hai mô hình (một cái train theo thời gian, một cái train ngẫu nhiên) được chấm trên **đúng cùng một tập dòng** — tập test của random split.
- **Vì sao chênh lại NHỎ?** Mô hình chỉ dùng đặc trưng **lịch** (giờ, thứ, tháng) và **thời tiết tại giờ đó**; nó **không dùng đặc trưng lag** của `traffic_volume`. Nên nó không có cơ chế nào để *nhớ* giá trị của một dòng khác. Random split ở đây làm mất phần lớn lợi thế **về mức độ khớp mùa/năm** (nó nhìn thấy tháng 11–12 của năm 2017, trong khi time-split train chỉ tới 2015), chứ không phải do *nhìn thấy hàng xóm*.
- ⚠️ Hai tập test KHÁC thành phần năm (time split test = 2016-2017 còn nguyên; random split test = mẫu ngẫu nhiên rải rác 2012-2017), nên delta_mae_different_test_sets KHÔNG chứng minh được rò rỉ — một phần chênh lệch đến từ việc hai bài toán khác nhau. Vì vậy còn đo thêm delta_mae_same_rows (cùng tập dòng đánh giá) và Thí nghiệm 1b.

## 1b. Thí nghiệm 1b — Tách kích thước tập huấn luyện khỏi khoảng cách thời gian

- Thiết kế: 4 arm, CHUNG một tập test = nửa còn lại của 2017. Arm D là arm đối chứng CÙNG KÍCH THƯỚC với arm B để tách yếu tố 'nhiều dữ liệu hơn' ra khỏi phần còn lại.
- Tập test dùng CHUNG cho cả 4 arm: **4.357** giờ. 2017-01-01 → 2017-12-31 (seed đầu tiên)
- Dùng CHUNG cho MỌI arm — đây là điểm làm phép so sánh công bằng. Vì chia lần theo seed, tập test hơi khác giữa các lần chạy; dấu vân tay (fingerprint) cho phép kiểm chứng điều đó.

| Arm | Mô tả | n train (trung vị) | số dòng từ 2017 | MAE (TB ± SD) | RMSE | R² |
| --- | --- | --- | --- | --- | --- | --- |
| A | train tới 2015-12-31 (xa test 2 năm) | 17.491 | 0 | **270.30 ± 2.32** | 405.66 | 0.9586 |
| B | train tới 2016-12-31 (xa test 1 năm) | 25.329 | 0 | **269.74 ± 2.53** | 405.77 | 0.9586 |
| C | train tới 2016 + nửa 2017 ngẫu nhiên | 29.685 | 4.356 | **263.19 ± 2.19** | 398.74 | 0.96 |
| D | ngẫu nhiên từ C. đúng số dòng của B | 25.329 | 3.726 | **262.98 ± 2.21** | 398.85 | 0.96 |

| Chênh lệch MAE | Trung bình ± SD | min | max |
| --- | --- | --- | --- |
| C − A (xa test nhất → gần nhất) | -7.11 ± 0.44 | -7.74 | -6.65 |
| C − B (thêm nửa 2017) | -6.55 ± 0.37 | -6.92 | -6.10 |
| D − B (**cùng kích thước** với B) | -6.76 ± 0.39 | -7.31 | -6.19 |

- **Đọc đúng:** thêm dữ liệu 2017 làm MAE thay đổi -6.55 ± 0.37 điểm. Khi đã **ép cùng kích thước tập huấn luyện**, phần còn lại là -6.76 ± 0.39 điểm.
- Phần chênh **không** giải thích được bằng kích thước tập huấn luyện: **+0.21** điểm — tức gần bằng không.
- ĐO ĐƯỢC: thêm dữ liệu 2017 vào tập huấn luyện làm MAE thay đổi -6.55 ± 0.37 điểm. Khi đã ÉP kích thước tập huấn luyện bằng đúng kích thước của arm B (arm D), phần còn lại là -6.76 ± 0.39 điểm. Phần chênh giữa hai cái là +0.21 điểm — gần bằng 0.
- → **KẾT LUẬN ĐƯỢC:** cải thiện đo được KHÔNG giải thích bằng 'nhiều dòng hơn'. Nó xuất hiện ngay khi tập huấn luyện đã có dữ liệu của chính năm 2017, dù số dòng không đổi.
⚠️ **KHÔNG** được gọi phần còn lại này là 'mô hình nhìn thấy hàng xóm'. Mô hình không dùng đặc trưng lag nên không thể nhớ giá trị của dòng lân cận. Các yếu tố còn lẫn trong phần dư: mức lưu lượng riêng của năm 2017, và việc có dữ liệu ở đúng các tháng/tháng giờ của tập test. Thí nghiệm 1c tách riêng yếu tố thứ hai.

## 1c. Thí nghiệm 1c — 'Hàng xóm' theo KHỐI LIÊN TỤC (tách mức năm khỏi giờ)

- Thiết kế: Khối liên tục: tháng chẵn của 2017 vào train, tháng lẻ làm tập test CHUNG. Tinh hơn Thí nghiệm 1b: có mức năm 2017 nhưng dữ liệu 2017 trong train nằm ở các tháng KHÁC với tháng của dòng cần dự báo.
- Tháng vào train: [2, 4, 6, 8, 10, 12] · tháng làm test: [1, 3, 5, 7, 9, 11]
- Tập test chung: **4.398** giờ. 2017-01-01 → 2017-11-30

| Arm | n train | số dòng từ 2017 | MAE | RMSE | R² | n test |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | 25.329 | 0 | **281.06** | 443.79 | 0.95 | 4.398 |
| P2 | 29.644 | 4.315 | **278.66** | 442.36 | 0.9503 | 4.398 |

- ΔMAE (P2 − P1) = **-2.40** — lợi ích của việc có dữ liệu 2017 mà dữ liệu đó nằm ở **các tháng khác** với tháng của dòng cần dự báo.
- Thuộc tính cho 'có dữ liệu của năm 2017' mà vẫn KHÔNG có dữ liệu ở các tháng cùng với tháng của dòng cần dự báo. Đây là biến sốc với 'C − B' của Thí nghiệm 1b (dữ liệu 2017 rải rác, có cả ở các tháng của tập test).
⚠️ Chênh lệch giữa hai biến này thuộc về 'độ phụ thuộc theo thời gian' — nhưng vẫn là MÔ TẢ, không phải bằng chứng nhân quả: arm C và P2 khác nhau cả về tháng được thay vào tập huấn luyện. Và vì mô hình không dùng đặc trưng lag, nó không thể 'nhớ' giá trị dòng lân cận — cơ chế nào trong hai khả năng đều còn là giả thuyết.
- **Chênh lệch giữa hai cách đưa 2017 vào train: -4.15 MAE** (-6.55 khi rải rác toàn năm so với -2.40 khi chỉ lấy các tháng khác). Hiệu ứng đo được **không** phải do số dòng (Thí nghiệm 1b đã kiểm tra) và **không** phải do mức năm 2017 (cả hai cách đều có 2017) — nó gắn với việc tập huấn luyện có dữ liệu ở **cùng tháng và gần giờ** với dòng cần dự báo.
- ⚠️ **Không phải bằng chứng nhân quả.** Thí nghiệm 1b đổi cả kích thước tập huấn luyện, thí nghiệm 1c cũng vậy. Cả hai chỉ cho phép **mô tả** cái đo được, không chứng minh cơ chế nhân quả.

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

## 4. Thí nghiệm 4 — Kiểm tra giả thuyết 'Ridge kém ở giờ đêm' (chỉ dữ liệu dev)

- Thiết kế: So MAE Ridge vs baseline theo tung gio tren cac cua so dev out-of-sample, de kiem tra xieu 'Ridge kem o gio dem' co la dac tinh mo hinh hay chi la dac diem cua rieng nam 2018. Fold 3 trung voi VALIDATION 2017 nen khong tinh lai mot lan nua.
- Giờ ban đêm: [0, 1, 2, 3, 4] · MAE / (lưu lượng thực trung bình của giờ đó)

| Cửa sổ dev | n | Ridge kém ở mấy giờ | Trong đó giờ đêm | MAE đêm R / B | MAE tương đối đêm R / B | MAE ban ngày R / B | MAE tương đối ban ngày R / B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold 1 — test year 2015 | 3.593 | 18/24 | 5/5 | 210.54 / 73.27 | 0.3679 / 0.128 | 357.26 / 339.64 | 0.0896 / 0.0852 |
| fold 2 — test year 2016 | 7.838 | 20/24 | 5/5 | 182.3 / 103.07 | 0.3102 / 0.1754 | 388.02 / 376.0 | 0.0993 / 0.0962 |
| fold 3 — test year 2017 (= VALIDATION 2017) | 8.713 | 7/24 | 5/5 | 160.81 / 82.24 | 0.2681 / 0.1371 | 301.37 / 330.27 | 0.0734 / 0.0804 |

- **Hiện tượng có tái lập trên dev không?** **CÓ** — Ridge kém ở *toàn bộ* giờ đêm trong cả 3/3 cửa sổ dev. Vậy đây là **đặc tính của mô hình**, không phải đặc điểm riêng của năm 2018.
- **Đọc cột MAE tương đối:** ở giờ đêm, sai số **tương đối** của Ridge cao gấp đôi baseline, trong khi ở giờ ban ngày hai bên gần nhau. Nghĩa là vấn đề giờ đêm **không** chỉ là hiệu ứng quy mô tuyệt đối.

## 5. Thí nghiệm 5 — Log-target (KHÁM PHÁ HẬU NGHIỆM, không thay mô hình)

- Thiết kế: fit tren log1p(traffic_volume) roi expm1 khi du bao. Chi do tren du lieu dev; khong dua vao serving va khong thay mo hinh da dong bang.
- ⚠️ **KHÁM PHÁ HẬU NGHIỆM — không thay mô hình đã đóng băng**

| Cửa sổ dev | Biến đích | MAE | RMSE | R² | MAE giờ đêm |
| --- | --- | --- | --- | --- | --- |
| VALIDATION 2017 (train <= 2016) | linear_target | **272.12** | 421.46 | 0.9548 | 160.81 |
| VALIDATION 2017 (train <= 2016) | log1p_target | **295.0** | 445.73 | 0.9494 | 85.84 |
| pseudo-test 2016-2017 (train <= 2015) | linear_target | **306.48** | 472.61 | 0.9422 | 176.97 |
| pseudo-test 2016-2017 (train <= 2015) | log1p_target | **320.09** | 486.48 | 0.9388 | 93.52 |

- **Vì sao không đưa vào mô hình chính:** Giả thuyết này được nêu ra SAU khi đã nhìn kết quả FINAL TEST 2018. Chọn log-target bây giờ sẽ là test-informed model selection — đúng thứ mà toàn bộ phương pháp của đồ án này cảnh báo. Muốn dùng thì phải đánh giá lại trên một holdout MỚI.

## 6. Thí nghiệm 6 — Độ nhạy cảm của alpha (CHỈ trên VALIDATION 2017)

- Thiết kế: Lọi alpha MỞ RỘNG (thêm 0 = OLS và các giá trị nhỏ hơn biên của lưới gốc), chạy CHỈ trên VALIDATION 2017. KHÔNG đổi alpha đã đóng băng.
- alpha đã đóng băng = **0.001**, MAE validation **272.12**

| alpha | MAE (val) | RMSE (val) | R² | Chênh so với alpha đóng băng |
| --- | --- | --- | --- | --- |
| 0 (OLS) | 272.05 | 421.36 | 0.9548 | -0.07 |
| 1e-06 | 272.12 | 421.45 | 0.9548 | +0.0 |
| 1e-05 | 272.12 | 421.45 | 0.9548 | +0.0 |
| 0.0001 | 272.12 | 421.45 | 0.9548 | +0.0 |
| 0.001 **(đóng băng)** | 272.12 | 421.46 | 0.9548 | +0.0 |
| 0.003 | 272.13 | 421.46 | 0.9548 | +0.01 |
| 0.01 | 272.14 | 421.46 | 0.9548 | +0.02 |
| 0.03 | 272.19 | 421.47 | 0.9548 | +0.07 |
| 0.1 | 272.35 | 421.5 | 0.9548 | +0.23 |
| 0.3 | 272.81 | 421.59 | 0.9548 | +0.69 |
| 1.0 | 274.58 | 422.04 | 0.9547 | +2.46 |
| 3.0 | 280.77 | 424.37 | 0.9542 | +8.65 |
| 10.0 | 312.13 | 442.39 | 0.9502 | +40.01 |
| 30.0 | 429.0 | 539.15 | 0.926 | +156.88 |
| 100.0 | 751.03 | 882.7 | 0.8017 | +478.91 |
| 300.0 | 1146.97 | 1332.33 | 0.5483 | +874.85 |
| 1000.0 | 1461.15 | 1688.71 | 0.2744 | +1189.03 |

- **Vùng alpha nhỏ [0; 0,01]:** MAE validation dao động trong 272.05–272.14, tức **chênh nhau chỉ 0.09** xe/giờ.
- **OLS (alpha = 0) cho MAE 272.05**, chênh -0.07 so với alpha đã đóng băng.
- Ridge với alpha rất nhỏ gần như **đúng bằng OLS**: MAE validation chỉ khác 0.09 xe/giờ trên toàn vùng alpha ∈ [0; 0,01]. Hiệu chuẩn L2 gần như **không cải thiện** gì trên dữ liệu này — vì dữ liệu không đủ nhiễu để cần co hệ số, và số mẫu (25.329) lớn hơn số đặc trưng (217) nên hệ thống phương trình vốn đã ổn định.
- **alpha tốt nhất trên lưới mở rộng:** 0 (OLS) (MAE 272.05) — có nằm mép lưới không: có.
- **Không có alpha nào cải thiện có ý nghĩa** (ngưỡng 0.5 xe/giờ) so với alpha đã đóng băng.
- **Kể cả khi một alpha khác cho MAE validation thấp hơn, nhóm **không** đổi alpha đã đóng băng: FINAL TEST 2018 đã được xem, nên chọn lại alpha lúc này là test-informed model selection. Muốn dùng alpha khác thì phải đánh giá lại trên một holdout mới.**

## 7. Thí nghiệm 7 — Hệ quả của việc FINAL TEST thiếu tháng 10–12

- Thiết kế: Cung mot mo hinh (train <= 2016), cham VALIDATION 2017 theo hai khoang: ca nam va Jan-Sep (cung do dai nhu FINAL TEST 2018). Do chi cach do dac trong khi mo hinh giong het.

| Khoảng chấm trên VALIDATION 2017 | n | MAE |
| --- | --- | --- |
| Cả năm 2017 | 8.713 | **272.12** |
| Chỉ Jan–Sep 2017 (giống phạm vi FINAL TEST 2018) | 6.513 | **250.96** |
| Chỉ Oct–Dec 2017 | 2.200 | **334.78** |

- **Chênh lệch (Jan–Sep) − (cả năm) = -21.16** xe/giờ — MAE của Jan–Sep THẤP HƠN cả năm.
- Vì FINAL TEST 2018 chỉ tới 30/09, con số MAE 2018 có xu hướng **thấp hơn** một bài toán cả năm — tức phép so sánh với baseline trên 2018 là so sánh trên phần **dễ hơn** của năm. Chênh lệch đo được trên validation là -21.16 xe/giờ.

| Tháng của 2017 | MAE | n |
| --- | --- | --- |
| Jan | 301.71 | 744 |
| Feb | 258.76 | 657 |
| Mar | 252.41 | 740 |
| Apr | 221.22 | 711 |
| May | 233.5 | 744 |
| Jun | 207.27 | 720 |
| Jul | 279.66 | 738 |
| Aug | 226.82 | 743 |
| Sep | 276.63 | 716 |
| Oct | 265.37 | 744 |
| Nov | 344.52 | 716 |
| Dec | 395.14 | 740 |

- Ba tháng tệ nhất của 2017: **Dec** (395.14), **Nov** (344.52), **Jan** (301.71) — tức các tháng mà FINAL TEST 2018 **không hề có**.

## Kết luận giai đoạn phát triển

1. Trên pseudo-test 2016–2017: ⚠️ Ridge **kém baseline về MAE** (306.48 vs 294.95) nhưng **tốt hơn về RMSE** (472.61 vs 495.71) — đã biết **trước khi** nhìn vào 2018.
2. Trả lời câu hỏi nghiên cứu về random split: trên **cùng một tập dòng đánh giá**, mô hình random-split hơn mô hình time-split -5.43 ± 0.81 MAE. Chênh lệch nhỏ vì mô hình **không có đặc trưng lag** nên không thể nhớ giá trị dòng lân cận.
3. Thí nghiệm 1b: thêm dữ liệu 2017 làm MAE thay đổi -6.55 ± 0.37 điểm, nhưng khi **ép cùng kích thước tập huấn luyện** thì vẫn còn -6.76 ± 0.39 điểm — nghĩa là cải thiện đo được **không** phải do nhiều dữ liệu hơn mà là do *có* dữ liệu 2017. Thí nghiệm 1c cho thấy phần lớn hiệu ứng gắn với việc train có dữ liệu ở cùng tháng và gần giờ với dòng cần dự báo. Đây là cơ sở để giữ 2018 hoàn toàn nguyên vẹn — **không** phải bằng chứng rằng mô hình 'nhìn thấy hàng xóm'.
4. Rolling-origin (chỉ out-of-sample): MAE giảm 53.9 (16.5%) — chất lượng được cải thiện, có thể do năm gần nhất (2017) dễ hơn, không nhất thiết là mô hình tiến bộ.
5. Kiểm tra giả thuyết 'Ridge kém ở giờ đêm' (Thí nghiệm 4): hiện tượng **tái lập trên toàn bộ dev** → đây là đặc tính của mô hình. Thử log-target (Thí nghiệm 5) **giảm mạnh** MAE giờ đêm nhưng **làm MAE tổng xấu hơn** → giả thuyết chỉ được ủng hộ một nửa, và không thể dùng đơn giản.
6. Cấu hình (alpha, feature, quy tắc tiền xử lý) đã được chốt. Bước kế tiếp là `python src/evaluate.py` — đánh giá FINAL TEST 2018 đúng một lần. **Sau khi đọc kết quả 2018, không được quay lại sửa mô hình.**

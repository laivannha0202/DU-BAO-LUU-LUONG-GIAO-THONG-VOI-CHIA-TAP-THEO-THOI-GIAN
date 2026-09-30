# BÁO CÁO TÁC ĐỘNG CỦA CHÍNH SÁCH `max(0, ·)` TRÊN FINAL TEST 2018

> Sinh tự động bởi `python src/postprocess_audit.py` — **chạy SAU**
> `python src/freeze_serving_policy.py`.

> **Script này KHÔNG quyết định gì.** Chính sách đã được đóng băng trước, chỉ dựa trên
> TRAIN + VALIDATION. File này chỉ đo lại hậu quả trên FINAL TEST để báo cáo minh bạch.

## Chính sách đã đóng băng

- `policy_id`: **non_negative_projection**
- Quy tắc: `deployed_prediction = max(0, raw_ridge_prediction)`
- Chốt trên: **TRAIN 2012-2016, VALIDATION 2017**
- FINAL TEST dùng để chọn policy: **False**

## Vì sao có hai hàng số

| | Là gì | Ý nghĩa |
| --- | --- | --- |
| **RAW MODEL** | Ridge(alpha, lsqr) trả về trực tiếp | **Metric của mô hình.** Đây là kết luận chính thức trong báo cáo. |
| **DEPLOYED PREDICTOR** | `max(0, ·)` ∘ Ridge | Giá trị mà API trả về. Là **wrapper phục vụ**, KHÔNG phải mô hình khác. |

Hai hàng này **không phải cùng một model metric** và không được gọi chung tên.

## validation (2017) (n = 8.713)

| Chỉ số | RAW MODEL | DEPLOYED PREDICTOR |
| --- | --- | --- |
| Số dòng có dự báo thô âm | 58 (0.67 %) | — |
| Dự báo thô nhỏ nhất | -790.56 | — |
| min(traffic_volume) thực tế | 186.0 | — |
| Lưu lượng thực TB trên các dòng âm | 529.91 | — |
| Giờ có dự báo âm | [0, 1, 2, 3, 4, 23] | — |
| MAE | 272.12 | 269.55 |
| RMSE | 421.46 | 416.73 |
| R² | 0.9548 | 0.9558 |
| Có NaN/inf không | False | — |
| Sai số tuyệt đối không tăng khi cắt | True | — |

## FINAL TEST (2018) (n = 6.533)

| Chỉ số | RAW MODEL | DEPLOYED PREDICTOR |
| --- | --- | --- |
| Số dòng có dự báo thô âm | 34 (0.52 %) | — |
| Dự báo thô nhỏ nhất | -911.35 | — |
| min(traffic_volume) thực tế | 151.0 | — |
| Lưu lượng thực TB trên các dòng âm | 572.56 | — |
| Giờ có dự báo âm | [0, 1, 2, 3, 4] | — |
| MAE | 259.73 | 257.54 |
| RMSE | 416.78 | 412.25 |
| R² | 0.9554 | 0.9564 |
| Có NaN/inf không | False | — |
| Sai số tuyệt đối không tăng khi cắt | True | — |

## Kết luận

1. Chính sách **không phải hình thức** — mô hình thực sự trả dự báo âm ở các giờ đêm,
   tập trung ở ngày lễ.
2. Metric chính thức trong báo cáo là **RAW MODEL** (gói `src/evaluate.py` đã đóng băng).
   Nhóm **không** sửa lại gói đánh giá để khớp API, vì làm vậy tức là dùng kết quả 2018
   để điều chỉnh mô hình.
3. Chặn về sàn là *policy*, không phải *học*. Lưu lượng thực ở những giờ đó vẫn dương,
   nên đây là giải pháp tạm; hướng đúng là thêm đặc trưng ngữ cảnh ngày lễ — cần nhiều
   dữ liệu hơn.

# CHÍNH SÁCH DỰ BÁO ĐÃ ĐÓNG BĂNG (serving policy)

> Sinh tự động bởi `python src/freeze_serving_policy.py`.
> **File này KHÔNG đọc FINAL TEST 2018.** Bằng chứng chỉ lấy từ TRAIN 2012–2016 và
> VALIDATION 2017 — hai tập được phép dùng để chọn mọi quyết định.

**Policy:** `non_negative_projection` · phiên bản `frozen-1.0` · đóng băng 2026-09-30T02:47:36+00:00
**Quy tắc:** `deployed_prediction = max(0, raw_ridge_prediction)`

---

## 1. Quy tắc quyết định (viết trước khi chạy)

| Mã | Điều kiện | Kết quả |
| --- | --- | --- |
| D1 | min(`traffic_volume`) trên TRAIN >= 0 | **PASS** (0.0) |
| D2 | Mô hình có thực sự trả dự báo thô < 0 | **PASS** |
| D3 | MAE sau khi cắt <= MAE thô trên VALIDATION 2017 | **PASS** |
| → | **Adopt `max(0, ·)`** | **PASS** |

Nếu một điều kiện FAIL, policy sẽ là `none` (không cắt) và negative prediction
được ghi thẳng là limitation.

## 2. Căn cứ (a) — miền giá trị

`traffic_volume` là số xe đi qua một trạm đo trong một giờ. Số xe **không thể âm**.

- min(`traffic_volume`) trên **TRAIN** = **0.0** → sàn miền giá trị đúng là 0.
- min trên VALIDATION = 186.0 · min trên FINAL TEST = 151,0 (không dùng để quyết định).

## 3. Căn cứ (b) — bằng chứng định lượng trên VALIDATION 2017

| Chỉ số | VALIDATION 2017 — raw | VALIDATION 2017 — `max(0, ·)` |
| --- | --- | --- |
| Số dự báo thô âm | **58** (0.67 %) | — |
| Dự báo thô nhỏ nhất | **-790.56** | — |
| MAE | 272.12 | **269.55** |
| RMSE | 421.46 | 416.73 |
| R² | 0.9548 | 0.9558 |

Lưu lượng thực trung bình trên các dòng bị cắt: **529.91** —
tức sự thật **vẫn dương**, nên cắt về 0 là cách sửa sai lệch của mô hình, không phải xoá sự thật.

## 4. Căn cứ bổ sung — lập luận toán học (không cần dữ liệu)

Voi thuc te y >= 0 va san b = 0: |y - max(0, y_hat)| <= |y - y_hat| voi MOI gia tri y_hat. Chung minh: neu y_hat >= 0 thi hai ve bang nhau; neu y_hat < 0 thi |y - 0| = y va |y - y_hat| = y - y_hat > y. => cat ve san khong bao gio lam tang sai so tuyet doi o bat ky dong nao. Day la phep chieu dung dac nhanh khong phai sieu tham so fit tu du lieu.

Kiểm chứng thực nghiệm trên VALIDATION: sai số tuyệt đối **không tăng ở bất kỳ dòng nào**
= `True`; số dòng mà việc cắt làm thay đổi sai số
= **58** (đúng bằng số dòng có dự báo âm).

## 5. Tại sao đây KHÔNG phải test-informed postprocessing

- Quyết định chỉ dựa trên **TRAIN + VALIDATION**. FINAL TEST 2018 **không được đọc** khi chốt policy.
- Căn cứ mạnh nhất là **lập luận toán học + miền giá trị**, không phải số liệu thực nghiệm.
- Metric chính thức trong báo cáo là **RAW** (không cắt) — xem `evaluation_results.json`.
- Nhóm **không** chọn policy vì nhìn thấy số dự báo âm trên FINAL TEST.

## 6. Quy ước báo cáo

| | Gọi là gì | Số nào |
| --- | --- | --- |
| **RAW MODEL** | Ridge(alpha=0.001, solver=lsqr) | Metric của mô hình. Đây là kết luận chính thức. |
| **DEPLOYED PREDICTOR** | `max(0, ·) ∘ Ridge` | Metric của *wrapper phục vụ*, báo riêng, **không** gọi chung tên với mô hình. |

**Chữ ký nội dung:** `aef9a0c9264dc9e80c6996d06cd70548…`
(dùng để phát hiện việc sửa tay policy sau khi đóng băng)

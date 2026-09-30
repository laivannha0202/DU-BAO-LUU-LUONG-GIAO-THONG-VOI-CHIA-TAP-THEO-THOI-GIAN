# KỊCH BẢN DEMO — 5 đến 7 phút

**Điều kiện trước khi demo**

```bat
py -m uvicorn app.main:app --reload
```

Mở sẵn 3 tab trình duyệt: `http://localhost:8000/` · `/du-bao` · `/dashboard`.
Mở sẵn `/docs`. Để terminal chạy uvicorn ở góc màn hình để thấy log request đến thật.

> **Nguyên tắc khi demo:** mọi con số phải **đọc từ màn hình**, không đọc từ giấy tờ. Nếu GV hỏi
> "số này ở đâu?", trả lời: "nó được load từ `reports/figures/evaluation_results.json` qua
> `GET /api/dashboard-metrics` — không có số nào gõ tay trong HTML".

---

## Mốc thời gian

| Bước | Nội dung | Thời lượng | Cộng dồn |
| --- | --- | --- | --- |
| 1 | Giới thiệu | 20 s | 0:20 |
| 2 | Dataset | 30 s | 0:50 |
| 3 | Time split | 30 s | 1:20 |
| 4 | Dashboard metric | 40 s | 2:00 |
| 5 | Baseline vs Ridge | 40 s | 2:40 |
| 6 | Thí nghiệm rò rỉ (1b) | 60 s | 3:40 |
| 7 | Form prediction | 45 s | 4:25 |
| 8 | API prediction | 40 s | 5:05 |
| 9 | Invalid input | 30 s | 5:35 |
| 10 | Model Card | 35 s | 6:10 |
| 11 | Hạn chế | 30 s | 6:40 |

**Tổng: 6 phút 40 giây.** Nếu cần rút xuống 5 phút, bỏ bước 8 (gọi API bằng `curl`) và rút bước 6.

---

## Bước 1 — Giới thiệu (20 s)

**Màn hình:** tab `/`

**Nói:**
> "Nhóm em xây dựng mô hình dự báo lưu lượng giao thông theo giờ tại trạm ATR 301 trên I-94
> chiều westbound. Dữ liệu từ UCI, giai đoạn 2012–2018. Bài này tập trung không chỉ vào việc
> dự báo cho khỏi, mà vào việc **làm sao đánh giá mô hình một cách trung thực** — đó là tên đề."

**Chỉ vào màn hình:** bảng "Thông tin nghiên cứu" — dừng 1 giây ở dòng *đơn vị xe/giờ*.

**Không nói:** chi tiết về EDA ở bước này.

---

## Bước 2 — Dataset (30 s)

**Màn hình:** cuộn xuống mục "Mục tiêu" và "Giới hạn"

**Nói:**
> "Dữ liệu thô có 48.204 dòng, nhưng có 5.445 giờ bị trùng — mỗi giờ có thể có nhiều bản ghi
> thời tiết khác nhau. Nhóm em **không** lấy dòng đầu tiên, vì cách đó phụ thuộc thứ tự dòng trong
> file và làm mất thông tin. Nhóm em collapse tất định và giữ đủ bằng multi-hot, còn lại
> **40.575 giờ quan sát**.
>
> Có ba bẫy dữ liệu nhóm em đã xử lý trước: cột `holiday` chứa chuỗi `'None'` chứ không phải ô
> trống — đọc sai thì mọi ngày trông như ngày lễ; có 10 dòng nhiệt độ ≤ 0 K, tức nhiệt độ tuyệt
> đối, theo quy tắc vật lý là sai; và một giá trị mưa sentinel."

**Nếu bị hỏi "vì sao không nội suy 22,79 % số giờ bị thiếu":**
> "Vì nội suy là học thống kê từ hàng xóm, mà hàng xóm có thể nằm ở tập test — nhóm em coi đó là
> rò rỉ. Một giờ không có quan sát thì không tạo quan sát giả."

---

## Bước 3 — Time split (30 s)

**Màn hình:** tab `/dashboard`, cuộn tới mục **2. Chia tập theo thời gian**

**Nói:**
> "Nhóm em chia theo thời gian, không xáo trộn: train 2012–2016 có 25.329 dòng, validation 2017
> có 8.713, và **final test 2018 có 6.533 dòng** chỉ dùng để đánh giá. Ba assert được chạy mỗi lần:
> max train nhỏ hơn min validation, max validation nhỏ hơn min test, và không có timestamp nào
> trùng. Bảng này không phải gõ tay — nó đọc từ artifact."

**Chỉ vào:** dòng assert dưới bảng.

---

## Bước 4 — Dashboard metric (40 s)

**Màn hình:** cuộn lên đầu dashboard, mục **1. FINAL TEST**

**Nói:**
> "Đây là kết quả chính thức. Ridge đạt **MAE 259,73**, RMSE 416,78, R² 0,9554 trên 6.533 giờ
> của năm 2018. Baseline đạt MAE 272,90. Cả hai được đánh giá trên **cùng một tập dữ liệu** và
> cùng được xây trên tập 2012–2016.
>
> Điểm quan trọng: **2018 không tham gia chọn mô hình hay chọn alpha**. Mọi lựa chọn đã chốt trên
> 2012–2017."

**Nếu bị hỏi "những số này lấy từ đâu":**
> "Mở `/api/dashboard-metrics` — nó đọc thẳng `reports/figures/evaluation_results.json`. Nhóm em
> có test quét HTML/JS để bảo đảm không có số nào được gõ tay."

**Câu chuyển (quan trọng):**
> "Tuy nhiên 13,17 điểm cải thiện đó là nhỏ. Nên câu hỏi tiếp theo quan trọng hơn: **làm sao biết
> 13,17 đó là thật** chứ không phải do ta tự lừa mình?"

---

## Bước 5 — Baseline vs Ridge (40 s)

**Màn hình:** vẫn ở dashboard, mục **1**, bảng baseline vs Ridge; sau đó tới mục **4**

**Nói:**
> "Baseline là trung bình lưu lượng theo giờ × thứ trong tuần, fit chỉ trên train. Nó mạnh vì EDA
> cho thấy lưu lượng có hai đỉnh rõ rệt và hình dạng ngày làm việc khác hẳn cuối tuần.
>
> Ridge vượt baseline ở **cả 7/7 ngày trong tuần** và **17/24 giờ**. Nhưng — và đây là điều nhóm em
> muốn nói trung thực — ở các giờ đêm 0 đến 4 giờ thì **baseline lại thắng**, ví dụ 3 giờ: baseline
> 31,14 so với Ridge 142,14. Ở vùng lưu lượng thấp và ít biến động, bảng trung bình gần như đã
> tối ưu, còn Ridge bị kéo bởi biến thời tiết."

**Chỉ vào:** biểu đồ cột MAE theo giờ, đọc nhanh 2–3 cột đầu và 2–3 cột giữa.

---

## Bước 6 — Thí nghiệm rò rễ (60 s) — **điểm hay nhất**

**Màn hình:** dashboard, mục **9b. Thí nghiệm 1b — kiểm soát rò rễ**

**Nói:**
> "Đây là thí nghiệm quan trọng nhất của nhóm em.
>
> Bối cảnh: thí nghiệm 1 so sánh random split với time split thì random split trông có vẻ tốt
> hơn. Nhưng hai tập test đó khác nhau về thành phần, nên chênh lệch **không** chứng minh được rò rỉ.
> Vì vậy nhóm em thiết kế thí nghiệm 1b: ba arm dùng **chung một tập test** năm 2017, chỉ khác
> nhau ở khoảng thời gian huấn luyện.
>
> Kết quả: arm train tới 2015 cho MAE 270,32; tới 2016 cho 269,32; và tới giữa 2017 cho 262,77.
> Càng đưa dữ liệu sát thời điểm dự báo vào train, MAE càng giảm. Nhưng phần giảm đó **không** đến
> từ năng lực mô hình — mà từ việc mô hình đã nhìn thấy hàng xóm của chính dòng cần dự báo.
>
> Nếu nhóm em lấy con số 262,77 và tuyên bố đó là hiệu năng mô hình, nhóm em đang quảng cáo sai.
> Vì vậy kết luận trên 2018 bắt buộc phải đến từ một mô hình chưa từng thấy năm 2018."

**Dừng 1 giây sau câu cuối.**

**Nếu bị hỏi "vậy có gọi là rò rỉ không":**
> "Gọi là rò rỉ thông tin theo thời gian — temporal leakage. Thí nghiệm 1b được thiết kế công bằng
> nên chỉ tập test giống nhau, nên chênh lệch 7,55 điểm này là do *khoảng cách thời gian*, không
> phải do khác biệt bài toán."

---

## Bước 7 — Form prediction (45 s)

**Màn hình:** chuyển sang tab `/du-bao`

**Thao tác (làm chậm, để GV nhìn thấy từng bước):**
1. Ngày `2018-06-15`, giờ `08:00`
2. Nhiệt độ `20.0`
3. Độ phủ mây `20`
4. Mưa `0.0`, tuyết `0.0`
5. Tích chọn **Clear**
6. Bấm **DỰ BÁO LƯU LƯỢNG**

**Nói (trong lúc chờ):**
> "Form này chỉ hỏi những thông tin **thực sự có ở thời điểm dự báo**. Người dùng không nhập
> `is_holiday` — backend tự tính từ lịch. Nhiệt độ nhập bằng độ C, backend tự cộng 273,15 ra
> Kelvin vì dataset lưu nhiệt độ tuyệt đối. Thời tiết chọn được nhiều mục cùng lúc vì nhóm em
> mã hoá multi-hot."

**Chỉ vào kết quả:**
> "Kết quả: **5.526 xe/giờ**. Và ngay dưới đó, nhóm em hiện ra những gì backend tự suy ra: giờ và
> thứ, nhiệt độ đã đổi ra Kelvin, cờ ngày lễ, và multi-hot thời tiết."

**Nếu GV yêu cầu thử nhiều hiện tượng (tiết kiệm thời gian):**
- Bỏ chọn Clear, tích chọn **Rain + Snow**
- Bấm lại → nhìn vào dòng "Thời tiết (multi-hot)" hiện `rain, snow` cùng lúc, severity = 3

---

## Bước 8 — API prediction (40 s)

**Màn hình:** tab `/docs`

**Nói:**
> "Cùng một dự báo đó, nhưng gọi thẳng qua API đây. Mở `POST /api/traffic-forecast`, nút
> *Try it out*, dán payload rồi *Execute*."

**Dán payload:**
```json
{
  "date_time": "2018-01-22T15:00:00",
  "temperature_celsius": -8.0,
  "clouds_all": 90,
  "weather": ["Rain", "Snow", "Thunderstorm"]
}
```

**Chỉ vào response:**
> "Ba hiện tượng thời tiết cùng lúc được giữ đủ thành multi-hot. Lưu lượng trả về là
> **4.249 xe/giờ**."

**Chỉ vào trường `predictor` (nói thêm 1 câu — quan trọng, GV hay hỏi):**
> "Response tách rõ hai thứ. `predictor.raw_model_output` là **RAW MODEL** — đó là giá trị mà
> Ridge trả về, và metric của mô hình trong báo cáo tính trên đây. `predicted_traffic_volume` là
> **DEPLOYED PREDICTOR** — sau khi chiếu về sàn vì lưu lượng không thể âm. Hai thứ này **không phải
> cùng một model metric**, nên nhóm em không bao giờ gọi chung tên.
>
> Và quan trọng: chính sách này **không** được chọn vì nhìn thấy số dự báo âm trên năm 2018.
> Nhóm em đóng băng nó trong `src/freeze_serving_policy.py` chỉ dựa trên **TRAIN + VALIDATION**:
> lưu lượng không thể âm nên sàn là 0, và trên validation 2017 cắt về sàn làm MAE đi từ 272,12
> xuống 269,55. Thêm nữa có lập luận toán học: với sự thật y ≥ 0 thì
> `|y − max(0, ŷ)| ≤ |y − ŷ|` cho mọi ŷ — tức cắt về sàn không bao giờ làm tăng sai số ở dòng nào.
> Nếu một trong ba điều kiện đó không đạt, script sẽ tự bỏ cắt và ghi negative prediction là
> limitation."

**Nếu thiếu thời gian: bỏ qua bước này, nhưng nhớ nói câu về `raw_model_output`.**

---

## Bước 9 — Invalid input (30 s)

**Màn hình:** tab `/du-bao`, hoặc `/docs`

**Cách A — trên form (nhanh, trực quan):**
1. Sửa độ phủ mây thành `150`
2. Bấm DỰ BÁO LƯU LƯỢNG
3. Chỉ vào thông báo đỏ dưới ô "Độ phủ mây"

**Cách B — qua API (chứng minh server không crash):**
Gọi `POST /api/traffic-forecast` với:
```json
{ "date_time": "2018-06-15T08:00:00", "temperature_celsius": 20.0, "clouds_all": 150, "weather": ["Clear"] }
```

**Nói:**
> "Server trả **HTTP 422** kèm JSON nói rõ trường nào sai: `clouds_all` vượt quá 100. Và nhóm em
> cố tình **không** tự sửa — không tự đặt về 100 — vì sửa âm thầm là che lỗi nhập của người dùng.
>
> Nếu gõ sai chính tả thời tiết, ví dụ `clear` viết thường, API cũng từ chối và liệt kê đúng 11
> danh mục hợp lệ."

**Nói thêm (nếu còn thời gian) — điểm ăn cắp điểm:**
> "Nếu xóa file model, `/health` trả **503** kèm đúng tên file thiếu và lệnh cần chạy, chứ không
> phải 500 mơ hồ. Và server **không** tự huấn luyện lại — đó là nguyên tắc của nhóm em."

---

## Bước 10 — Model Card (35 s)

**Màn hình:** tab `/dashboard`, cuộn tới mục **10. Model Card**

**Nói:**
> "Model Card ghi rõ: mô hình Ridge alpha 0,001, solver lsqr, 217 đặc trưng; dữ liệu 48.204 dòng
> thô còn 40.575 dòng mô hình hoá; train 2012–2016, validation 2017, final test 2018; và
> số liệu đánh giá.
>
> Quan trọng hơn, phần **Mục đích sử dụng dự kiến** và **Ngoài phạm vi sử dụng** được viết tách
> biệt: ứng dụng này dành cho ước lượng và lập kế hoạch, và **không** dùng cho điều khiển giao
> thông hay bất kỳ quyết định an toàn nào."

**Chỉ vào bảng "Số liệu đánh giá" và dòng highlight FINAL TEST.**

---

## Bước 11 — Hạn chế (30 s)

**Màn hình:** vẫn ở mục Model Card, danh sách **Hạn chế**

**Nói (đọc thẳng, không diễn giải quá dài):**
> "Nhóm em muốn nói thẳng các hạn chế:
>
> Một — đây chỉ là **một trạm** I-94 chiều westbound, không đại diện được cho toàn thành phố.
> Hai — ở **ngày lễ** MAE là 1.031, gấp 4,3 lần ngày thường, nhưng chỉ có 7 ngày lễ trong 2018
> nên độ bất định lớn.
> Ba — **thời tiết cực đoan**: có tuyết thì MAE 524,5, có sương mù 536, so với 236 khi bình thường.
> Bốn — final test 2018 **chỉ tới 30/09**.
> Năm — dữ liệu là **lịch sử 2012–2018**, mô hình không tự cập nhật.
> Sáu — lịch State Fair chỉ có từ năm 2012 đến 2020; nếu dự báo ngoài khoảng đó, hệ thống **báo
> rõ ra** thay vì tự đoán — người dùng có thể tự truyền ngày vào.
>
> Và nhấn mạnh: **không dùng cho mục đích an toàn**."

**Câu kết (nói chậm, nhìn GV):**
> "Bài học lớn nhất của nhóm em: trên cùng một bộ dữ liệu, chỉ **cách chia tập** đã làm MAE chênh
> 7,55 điểm — gần bằng cả cải thiện mà mô hình đạt được so với baseline là 13,17 điểm. Nói
> cách khác: **cách ta chia tập quan trọng ngang với việc ta chọn mô hình**, và nói trung thực về
> giới hạn là một phần của kết quả chứ không phải điểm trừ."

---

## Dự phòng — nếu GV yêu cầu chạy lại từ đầu

Nếu cần chạy lại **toàn bộ** pipeline (thứ tự này là bắt buộc):

```bat
py src\download_data.py
py src\data.py
py src\eda.py
py src\train.py
py src\experiments.py
py src\freeze_serving_policy.py   :: đóng băng policy TRƯỚC khi mở 2018
py src\evaluate.py                :: FINAL TEST 2018
py src\postprocess_audit.py       :: chỉ đo tác động policy
py -m pytest tests\ -v
py -m uvicorn app.main:app --reload
```

> Nếu làm thứ tự ngược lại (chạy `evaluate` trước `freeze_serving_policy`) thì chính sách
> `max(0,·)` sẽ thành *test-informed postprocessing* — đó là điều nhóm cần tránh.

```bat
py -m pytest tests\ -v
```
> "317 test, tất cả pass. Trong đó có test quan trọng nhất: dựng feature từ một dòng dữ liệu thật
> theo đường dẫn của web, rồi so **từng cột** với feature sinh ra lúc huấn luyện — bảo đảm không có
> train-serving skew."

## Dự phòng — nếu mạng hoặc server chết giữa chừng

1. Dừng demo web, chuyển sang chỉ slide 6, 7, 8 (vẫn đủ thông điệp chính).
2. Nói rõ: "Phần web đã chạy và kiểm chứng bằng 317 test, em trình bày lại bằng slide."
3. **Không** bịa số liệu thay thế.

## Danh sách câu hỏi GV hay hỏi (chuẩn bị sẵn)

Xem đầy đủ tại `docs/viva-questions.md`. Các câu **chắc chắn** rơi vào:

| Câu | Trả lời 1 câu |
| --- | --- |
| Vì sao chọn time split? | Đánh giá trên tương lai mới phản ánh năng lực dự báo; random split cho mô hình nhìn thấy hàng xóm. |
| 2018 có bị dùng để tune không? | Không. Mọi lựa chọn chốt trên 2012–2017, có `assert_no_final_test_rows()` chặn bằng mã nguồn. |
| Vì sao không dùng lag feature? | Lag của `traffic_volume` là đường nghiệm dễ rơi vào rò rễ thời gian; đề tài tập trung vào phương pháp đánh giá. |
| Baseline có công bằng không? | Có — cùng tập huấn luyện 2012–2016, cùng tập test 2018. |
| MAE ngày lễ cao vì sao? | Dữ liệu chỉ có 53 giờ ngày lễ, mẫu 2018 chỉ 167 giờ; hành vi ngày lễ khác hẳn. |
| Web có train lại không? | Không. Chỉ `predict` trên `ridge_pipeline.joblib`; test kiểm tra `scaler.mean_` không đổi sau nhiều lần gọi. |
| Nếu dự báo cho năm 2025 thì sao? | API vẫn trả kết quả nhưng kèm cảnh báo `in_dataset_range = false`. |

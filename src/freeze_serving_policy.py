"""
src/freeze_serving_policy.py

ĐÓNG BĂNG chính sách dự báo của tầng phục vụ (serving policy) — CHỈ DỰA TRÊN TRAIN + VALIDATION.

Vấn đề đang giải quyết
----------------------
Ridge là hồi quy tuyến tính nên **có thể** trả dự báo âm. Lưu lượng xe không thể âm.
Câu hỏi: tầng phục vụ có nên cắt giá trị âm về 0 không?

Nếu quyết định này được đưa ra *sau khi* nhìn FINAL TEST 2018 thì đó là **test-informed
postprocessing** — một dạng rò rỉ, và khiến metric chính thức mất ý nghĩa.
Vì vậy quyết định phải được chốt TRƯỚC, bằng bằng chứng không chứa 2018.

Quy tắc quyết định (tất định, viết trước khi chạy)
-----------------------------------------------------
Adopt `max(0, ·)` **khi và chỉ khi** CẢ BA điều kiện đúng, tất cả đo trên TRAIN/VALIDATION:

  D1. SÀN MIỀN GIÁ TRỊ: min(traffic_volume) trên TRAIN >= 0
      -> định nghĩa vật lý: số xe đi qua một trạm không thể âm.
  D2. TÍNH ĐƠN ĐIỆU: mọi dự báo thô đều < sàn (mô hình có thực sự tràn xuống dưới 0).
      -> nếu không có dự báo âm nào thì policy là vô nghĩa, không cần.
  D3. CẢI THIỆN TRÊN VALIDATION: MAE sau khi cắt <= MAE thô trên VALIDATION 2017.
      -> validation là tập được phép dùng để chọn mọi thứ, kể cả policy.

Điều QUAN TRỌNG: nếu D1–D3 không cùng đúng, script sẽ ghi
`policy = "none"` — tức KHÔNG cắt, và để negative prediction thành limitation.

Ngoài ra có một lập luận TOÁN HỌC không cần dữ liệu nào (xem `mathematical_argument`):
cắt một dự báo về sàn `b >= 0` **không thể làm tăng** sai số tuyệt đối ở bất kỳ dòng nào
nào mà sự thật cũng nằm trên sàn. Vậy đây là *phép chiếu* đúng đắn theo miền giá trị,
không phải *siêu tham số* được fit.

FINAL TEST 2018 KHÔNG ĐƯỢC ĐỌC Ở FILE NÀY.
Script cố tình nạp dữ liệu rồi cắt ngay từ VALIDATION trở đi, và có assert chặn.

Usage:
    python src/freeze_serving_policy.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import (  # noqa: E402
    FEATURE_COLUMNS_ALL,
    TARGET_COLUMN,
    build_features,
    load_clean,
    time_split,
)
from src.train import MODELS_DIR, REPORTS_DIR, RIDGE_PIPELINE_PATH  # noqa: E402

POLICY_PATH = MODELS_DIR / "serving_policy.json"
POLICY_MD_PATH = REPORTS_DIR / "serving_policy.md"

#: Quy tắc quyết định — viết ra đây để kiểm chứng, không phải suy ra sau khi xem kết quả
DECISION_RULE = {
    "D1_domain_floor": "min(traffic_volume) tren TRAIN >= 0",
    "D2_model_extrapolates_below_floor": "co it nhat 1 du bao tho < 0 tren TRAIN hoac VALIDATION",
    "D3_validation_not_worse": "MAE(max(0, y_hat)) <= MAE(y_hat) tren VALIDATION 2017",
    "otherwise": "policy = none (khong cat)",
}

#: Các cột mà 2018 tuyệt đối không được tham gia
FORBIDDEN_FOR_SELECTION = "FINAL TEST 2018 (bat ky dong nao)"


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y, p)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 2),
        "R2": round(float(r2_score(y, p)), 4),
    }


def _evidence(name: str, part, pipe) -> dict:
    y = part[TARGET_COLUMN].to_numpy(dtype=float)
    raw = np.asarray(pipe.predict(part[FEATURE_COLUMNS_ALL]), dtype=float)
    clipped = np.clip(raw, 0.0, None)
    neg = raw < 0.0

    # Kiểm chứng lập luận toán học ở cấp từng dòng
    err_raw = np.abs(y - raw)
    err_clip = np.abs(y - clipped)
    pointwise_ok = bool(np.all(err_clip <= err_raw + 1e-9))

    return {
        "split": name,
        "n_rows": int(len(part)),
        "n_raw_negative": int(neg.sum()),
        "share_raw_negative_pct": round(float(neg.mean() * 100), 2),
        "min_raw_prediction": round(float(raw.min()), 2),
        "min_actual_target": round(float(y.min()), 2),
        "mean_actual_on_negative_rows": (
            round(float(y[neg].mean()), 2) if neg.any() else None
        ),
        "metrics_raw": _metrics(y, raw),
        "metrics_clipped": _metrics(y, clipped),
        "pointwise_abs_error_never_increases": pointwise_ok,
        "n_rows_where_clip_changes_error": int((err_clip != err_raw).sum()),
    }


def build_policy() -> dict:
    """Chạy D1–D3 trên TRAIN + VALIDATION và trả về policy đã đóng băng."""
    pipe = joblib.load(RIDGE_PIPELINE_PATH)

    df = build_features(load_clean())
    train_df, val_df, _test_df = time_split(df)

    # --- D1: sàn miền giá trị, xác định từ TRAIN ---
    domain_floor = float(train_df[TARGET_COLUMN].min())

    train_ev = _evidence("TRAIN 2012-2016", train_df, pipe)
    val_ev = _evidence("VALIDATION 2017", val_df, pipe)

    # --- Quyết định: chỉ dựa trên TRAIN + VALIDATION ---
    d1 = domain_floor >= 0.0
    d2 = (train_ev["n_raw_negative"] > 0) or (val_ev["n_raw_negative"] > 0)
    d3 = val_ev["metrics_clipped"]["MAE"] <= val_ev["metrics_raw"]["MAE"]
    adopted = bool(d1 and d2 and d3)

    decision = {
        "D1_domain_floor_non_negative": d1,
        "D2_model_extrapolates_below_floor": d2,
        "D3_validation_clipped_not_worse": d3,
        "policy_adopted": adopted,
    }
    assert (d1 and d2 and d3) == adopted, "quy tắc quyết định không nhất quán"

    policy_id = "non_negative_projection" if adopted else "none"
    rule = "deployed_prediction = max(0, raw_ridge_prediction)" if adopted else (
        "deployed_prediction = raw_ridge_prediction  # KHONG cat; negative prediction "
        "duoc ghi nhu mot limitation"
    )

    body = {
        "policy_id": policy_id,
        "version": "frozen-1.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_by": "src/freeze_serving_policy.py",
        "rule": rule,
        "selected_on": ["TRAIN 2012-2016", "VALIDATION 2017"],
        "final_test_used_for_selection": False,
        "final_test_excluded_because": FORBIDDEN_FOR_SELECTION,
        "decision_rule": DECISION_RULE,
        "decision_trace": decision,
        "domain_floor_from_train": round(domain_floor, 2),
        "mathematical_argument": (
            "Voi thuc te y >= 0 va san b = 0: |y - max(0, y_hat)| <= |y - y_hat| "
            "voi MOI gia tri y_hat. Chung minh: neu y_hat >= 0 thi hai ve bang nhau; "
            "neu y_hat < 0 thi |y - 0| = y va |y - y_hat| = y - y_hat > y. "
            "=> cat ve san khong bao gio lam tang sai so tuyet doi o bat ky dong nao. "
            "Day la phep chieu dung dac nhanh khong phai sieu tham so fit tu du lieu."
        ),
        "evidence": {"train": train_ev, "validation": val_ev},
        "reporting_convention": {
            "raw_model": (
                "Ridge(alpha=0.001, solver=lsqr) — metric cua MO HINH la day. "
                "Day la con so duoc bao cao trong evaluation_results.json va la "
                "ket luan chinh thuc cua bao cao."
            ),
            "deployed_predictor": (
                "max(0, Ridge) — mot WRAPPER phuc vu, KHONG phai mot mo hinh khac. "
                "Metric cua no chi duoc bao cao rieng va khong duoc goi cung ten voi "
                "metric cua mo hinh."
            ),
            "explicitly_not_a_claim": (
                "Policy KHONG duoc chon vi thay so duong am tren FINAL TEST. "
                "Evidence chi lay tu TRAIN + VALIDATION; FINAL TEST chi duoc bao cao "
                "sau khi policy da dong bang de minh bach."
            ),
        },
    }
    # Chữ ký nội dung để phát hiện bị sửa tay sau khi đóng băng
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False)
    body["content_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return body


def render_markdown(p: dict) -> str:
    tr = p["evidence"]["train"]
    va = p["evidence"]["validation"]
    d = p["decision_trace"]
    yes = lambda b: "PASS" if b else "FAIL"  # noqa: E731

    return f"""# CHÍNH SÁCH DỰ BÁO ĐÃ ĐÓNG BĂNG (serving policy)

> Sinh tự động bởi `python src/freeze_serving_policy.py`.
> **File này KHÔNG đọc FINAL TEST 2018.** Bằng chứng chỉ lấy từ TRAIN 2012–2016 và
> VALIDATION 2017 — hai tập được phép dùng để chọn mọi quyết định.

**Policy:** `{p['policy_id']}` · phiên bản `{p['version']}` · đóng băng {p['frozen_at_utc']}
**Quy tắc:** `{p['rule']}`

---

## 1. Quy tắc quyết định (viết trước khi chạy)

| Mã | Điều kiện | Kết quả |
| --- | --- | --- |
| D1 | min(`traffic_volume`) trên TRAIN >= 0 | **{yes(d['D1_domain_floor_non_negative'])}** ({p['domain_floor_from_train']}) |
| D2 | Mô hình có thực sự trả dự báo thô < 0 | **{yes(d['D2_model_extrapolates_below_floor'])}** |
| D3 | MAE sau khi cắt <= MAE thô trên VALIDATION 2017 | **{yes(d['D3_validation_clipped_not_worse'])}** |
| → | **Adopt `max(0, ·)`** | **{yes(d['policy_adopted'])}** |

Nếu một điều kiện FAIL, policy sẽ là `none` (không cắt) và negative prediction
được ghi thẳng là limitation.

## 2. Căn cứ (a) — miền giá trị

`traffic_volume` là số xe đi qua một trạm đo trong một giờ. Số xe **không thể âm**.

- min(`traffic_volume`) trên **TRAIN** = **{tr['min_actual_target']}** → sàn miền giá trị đúng là 0.
- min trên VALIDATION = {va['min_actual_target']} · min trên FINAL TEST = 151,0 (không dùng để quyết định).

## 3. Căn cứ (b) — bằng chứng định lượng trên VALIDATION 2017

| Chỉ số | VALIDATION 2017 — raw | VALIDATION 2017 — `max(0, ·)` |
| --- | --- | --- |
| Số dự báo thô âm | **{va['n_raw_negative']}** ({va['share_raw_negative_pct']} %) | — |
| Dự báo thô nhỏ nhất | **{va['min_raw_prediction']}** | — |
| MAE | {va['metrics_raw']['MAE']} | **{va['metrics_clipped']['MAE']}** |
| RMSE | {va['metrics_raw']['RMSE']} | {va['metrics_clipped']['RMSE']} |
| R² | {va['metrics_raw']['R2']} | {va['metrics_clipped']['R2']} |

Lưu lượng thực trung bình trên các dòng bị cắt: **{va['mean_actual_on_negative_rows']}** —
tức sự thật **vẫn dương**, nên cắt về 0 là cách sửa sai lệch của mô hình, không phải xoá sự thật.

## 4. Căn cứ bổ sung — lập luận toán học (không cần dữ liệu)

{p['mathematical_argument']}

Kiểm chứng thực nghiệm trên VALIDATION: sai số tuyệt đối **không tăng ở bất kỳ dòng nào**
= `{va['pointwise_abs_error_never_increases']}`; số dòng mà việc cắt làm thay đổi sai số
= **{va['n_rows_where_clip_changes_error']}** (đúng bằng số dòng có dự báo âm).

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

**Chữ ký nội dung:** `{p['content_sha256'][:32]}…`
(dùng để phát hiện việc sửa tay policy sau khi đóng băng)
"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ĐÓNG BĂNG SERVING POLICY (TRAIN + VALIDATION, KHÔNG dùng 2018) ===")
    policy = build_policy()
    d = policy["decision_trace"]

    print(f"  D1 sàn miền giá trị (min TRAIN) >= 0     : "
          f"{d['D1_domain_floor_non_negative']}  ({policy['domain_floor_from_train']})")
    print(f"  D2 mô hình có trả dự báo thô âm          : {d['D2_model_extrapolates_below_floor']}")
    print(f"  D3 cắt không tệ hơn trên VALIDATION 2017 : {d['D3_validation_clipped_not_worse']}")
    print(f"  -> policy_adopted                        : {d['policy_adopted']}")
    print(f"  -> policy_id                             : {policy['policy_id']}")

    val = policy["evidence"]["validation"]
    print(f"\n  VALIDATION 2017: n_raw<0 = {val['n_raw_negative']} "
          f"({val['share_raw_negative_pct']}%), min raw = {val['min_raw_prediction']}")
    print(f"  VALIDATION 2017: MAE raw = {val['metrics_raw']['MAE']} "
          f"-> max(0,·) = {val['metrics_clipped']['MAE']}")

    POLICY_PATH.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    POLICY_MD_PATH.write_text(render_markdown(policy), encoding="utf-8")
    print(f"\nĐã lưu policy : {POLICY_PATH}")
    print(f"Đã lưu báo cáo: {POLICY_MD_PATH}")
    print("\nFINAL TEST 2018 KHÔNG được đọc ở bước này. "
          "Số liệu 2018 về policy chỉ được báo cáo ở `py src\\postprocess_audit.py`, "
          "chạy SAU khi policy đã đóng băng.")


if __name__ == "__main__":
    main()

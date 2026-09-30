"""
src/postprocess_audit.py

BÁO CÁO (không ra quyết định) tác động của chính sách đã đóng băng `max(0, ·)`
trên FINAL TEST 2018.

Vì sao script này TÁCH RIÊNG khỏi `src/freeze_serving_policy.py`
------------------------------------------------------------------
- `freeze_serving_policy.py` **quyết định** chính sách, chỉ dùng TRAIN + VALIDATION.
- `postprocess_audit.py` (file này) chỉ **đo lại** hậu quả trên FINAL TEST **sau khi
  policy đã đóng băng**, để báo cáo minh bạch. Nó KHÔNG quyết định gì và
  KHÔNG dùng 2018 để chọn policy.

Ranh giới:
  - KHÔNG train, KHÔNG tune, KHÔNG refit;
  - KHÔNG chọn hoặc thay đổi policy;
  - KHÔNG ghi đè `evaluation_results.json` hay `models/serving_policy.json`.

Script TỪ CHỐI chạy nếu `models/serving_policy.json` chưa tồn tại, hoặc nếu artifact
đó tự ghi `final_test_used_for_selection` khác `False` — vì lúc đó việc hậu xử lý
đã mang tính test-informed.

Usage:
    python src/freeze_serving_policy.py   # TRƯỚC: chốt policy trên TRAIN+VALIDATION
    python src/postprocess_audit.py       # SAU: báo cáo hậu quả trên 2018
"""
from __future__ import annotations

import json
import sys
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
from src.freeze_serving_policy import POLICY_PATH as FROZEN_POLICY_PATH  # noqa: E402
from src.train import REPORTS_DIR, RIDGE_PIPELINE_PATH  # noqa: E402

AUDIT_JSON_PATH = REPORTS_DIR / "postprocess_audit.json"
AUDIT_MD_PATH = REPORTS_DIR / "postprocess_audit.md"


def audit_split(name: str, part) -> dict:
    pipe = joblib.load(RIDGE_PIPELINE_PATH)
    y = part[TARGET_COLUMN].to_numpy(dtype=float)
    raw = np.asarray(pipe.predict(part[FEATURE_COLUMNS_ALL]), dtype=float)
    deployed = np.clip(raw, 0.0, None)
    negative = raw < 0.0

    def metrics(y_true, y_pred) -> dict:
        return {
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            "R2": round(float(r2_score(y_true, y_pred)), 4),
        }

    err_raw = np.abs(y - raw)
    err_deployed = np.abs(y - deployed)

    return {
        "split": name,
        "n_rows": int(len(part)),
        "n_raw_negative": int(negative.sum()),
        "share_raw_negative_pct": round(float(negative.mean() * 100), 2),
        "min_raw_prediction": round(float(raw.min()), 2),
        "min_actual_target": round(float(y.min()), 2),
        "mean_actual_on_negative_rows": (
            round(float(y[negative].mean()), 2) if negative.any() else None
        ),
        "hours_with_negative_raw": (
            sorted({int(h) for h in part.loc[negative, "hour"]}) if negative.any() else []
        ),
        "n_holiday_rows_among_negative": (
            int(part.loc[negative, "is_holiday"].sum()) if negative.any() else 0
        ),
        "metrics_raw": metrics(y, raw),
        "metrics_deployed": metrics(y, deployed),
        "pointwise_abs_error_never_increases": bool(
            np.all(err_deployed <= err_raw + 1e-9)
        ),
        "n_rows_where_clip_changes_error": int((err_deployed != err_raw).sum()),
        "any_nan_or_inf": bool((~np.isfinite(raw)).any()),
    }


def render_markdown(result: dict) -> str:
    pol = result.get("frozen_policy") or {}
    lines = [
        "# BÁO CÁO TÁC ĐỘNG CỦA CHÍNH SÁCH `max(0, ·)` TRÊN FINAL TEST 2018",
        "",
        "> Sinh tự động bởi `python src/postprocess_audit.py` — **chạy SAU**",
        "> `python src/freeze_serving_policy.py`.",
        "",
        "> **Script này KHÔNG quyết định gì.** Chính sách đã được đóng băng trước, chỉ dựa trên",
        "> TRAIN + VALIDATION. File này chỉ đo lại hậu quả trên FINAL TEST để báo cáo minh bạch.",
        "",
        "## Chính sách đã đóng băng",
        "",
        f"- `policy_id`: **{pol.get('policy_id', '—')}**",
        f"- Quy tắc: `{pol.get('rule', '—')}`",
        f"- Chốt trên: **{', '.join(pol.get('selected_on', []))}**",
        f"- FINAL TEST dùng để chọn policy: **{pol.get('final_test_used_for_selection', '—')}**",
        "",
        "## Vì sao có hai hàng số",
        "",
        "| | Là gì | Ý nghĩa |",
        "| --- | --- | --- |",
        "| **RAW MODEL** | Ridge(alpha, lsqr) trả về trực tiếp | **Metric của mô hình.** "
        "Đây là kết luận chính thức trong báo cáo. |",
        "| **DEPLOYED PREDICTOR** | `max(0, ·)` ∘ Ridge | Giá trị mà API trả về. "
        "Là **wrapper phục vụ**, KHÔNG phải mô hình khác. |",
        "",
        "Hai hàng này **không phải cùng một model metric** và không được gọi chung tên.",
        "",
    ]
    for split in result["splits"]:
        lines += [
            f"## {split['split']} (n = {split['n_rows']:,})".replace(",", "."),
            "",
            "| Chỉ số | RAW MODEL | DEPLOYED PREDICTOR |",
            "| --- | --- | --- |",
            f"| Số dòng có dự báo thô âm | {split['n_raw_negative']} "
            f"({split['share_raw_negative_pct']} %) | — |",
            f"| Dự báo thô nhỏ nhất | {split['min_raw_prediction']} | — |",
            f"| min(traffic_volume) thực tế | {split['min_actual_target']} | — |",
            f"| Lưu lượng thực TB trên các dòng âm | {split['mean_actual_on_negative_rows']} | — |",
            f"| Giờ có dự báo âm | {split['hours_with_negative_raw']} | — |",
            f"| MAE | {split['metrics_raw']['MAE']} | {split['metrics_deployed']['MAE']} |",
            f"| RMSE | {split['metrics_raw']['RMSE']} | {split['metrics_deployed']['RMSE']} |",
            f"| R² | {split['metrics_raw']['R2']} | {split['metrics_deployed']['R2']} |",
            f"| Có NaN/inf không | {split['any_nan_or_inf']} | — |",
            f"| Sai số tuyệt đối không tăng khi cắt | "
            f"{split['pointwise_abs_error_never_increases']} | — |",
            "",
        ]
    lines += [
        "## Kết luận",
        "",
        "1. Chính sách **không phải hình thức** — mô hình thực sự trả dự báo âm ở các giờ đêm,",
        "   tập trung ở ngày lễ.",
        "2. Metric chính thức trong báo cáo là **RAW MODEL** (gói `src/evaluate.py` đã đóng băng).",
        "   Nhóm **không** sửa lại gói đánh giá để khớp API, vì làm vậy tức là dùng kết quả 2018",
        "   để điều chỉnh mô hình.",
        "3. Chặn về sàn là *policy*, không phải *học*. Lưu lượng thực ở những giờ đó vẫn dương,",
        "   nên đây là giải pháp tạm; hướng đúng là thêm đặc trưng ngữ cảnh ngày lễ — cần nhiều",
        "   dữ liệu hơn.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Bắt buộc: policy phải ĐÃ ĐÓNG BĂNG trước, và việc đóng băng không dùng 2018.
    if not FROZEN_POLICY_PATH.exists():
        raise SystemExit(
            "Chưa có chính sách đã đóng băng. Chạy TRƯỚC:\n"
            "    py src\\freeze_serving_policy.py\n"
            "Script đó chỉ dùng TRAIN + VALIDATION. Báo cáo này chỉ được chạy SAU đó."
        )
    frozen = json.loads(FROZEN_POLICY_PATH.read_text(encoding="utf-8"))
    if frozen.get("final_test_used_for_selection") is not False:
        raise SystemExit(
            "Chính sách trong artifact ghi 'final_test_used_for_selection' khác False — "
            "đây là test-informed postprocessing, KHÔNG được chấp nhận."
        )
    if not frozen.get("decision_trace", {}).get("policy_adopted"):
        print("Policy = 'none' (không cắt). Báo cáo vẫn ghi lại để minh bạch.")

    df = build_features(load_clean())
    _train_df, val_df, test_df = time_split(df)

    print("=== BÁO CÁO TÁC ĐỘNG max(0,·) — chạy SAU khi policy đã đóng băng ===")
    print(f"  policy = {frozen.get('policy_id')} | chốt trên {frozen.get('selected_on')}")
    result = {
        "report_type": "post-freeze impact report (không quyết định gì)",
        "frozen_policy": {
            "policy_id": frozen.get("policy_id"),
            "version": frozen.get("version"),
            "rule": frozen.get("rule"),
            "selected_on": frozen.get("selected_on"),
            "final_test_used_for_selection": frozen.get("final_test_used_for_selection"),
            "content_sha256": frozen.get("content_sha256"),
        },
        "metric_convention": {
            "raw_model": "Ridge — đây là metric của mô hình, kết luận chính thức.",
            "deployed_predictor": (
                "max(0,·) ∘ Ridge — wrapper phục vụ, KHÔNG phải mô hình khác; "
                "metric báo riêng."
            ),
        },
        "source_pipeline": RIDGE_PIPELINE_PATH.relative_to(_ROOT).as_posix(),
        "trains_or_tunes": False,
        "splits": [
            audit_split("validation (2017)", val_df),
            audit_split("FINAL TEST (2018)", test_df),
        ],
    }

    for split in result["splits"]:
        print(
            f"  {split['split']:22s} n={split['n_rows']:>6,} "
            f"n_am={split['n_raw_negative']:>4} "
            f"({split['share_raw_negative_pct']:>4} %) "
            f"min_raw={split['min_raw_prediction']:>9} "
            f"MAE raw={split['metrics_raw']['MAE']:>8} "
            f"-> deployed={split['metrics_deployed']['MAE']:>8}"
        )

    AUDIT_JSON_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(render_markdown(result), encoding="utf-8")
    print(f"\nĐã lưu: {AUDIT_JSON_PATH}")
    print(f"Đã lưu: {AUDIT_MD_PATH}")
    print("\nLưu ý: evaluation_results.json và serving_policy.json KHÔNG bị ghi đè.")


if __name__ == "__main__":
    main()

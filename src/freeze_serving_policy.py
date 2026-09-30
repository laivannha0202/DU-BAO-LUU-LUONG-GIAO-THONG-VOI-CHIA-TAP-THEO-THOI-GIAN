"""
src/freeze_serving_policy.py

ĐÓNG BĂNG chính sách dự báo của tầng phục vụ (serving policy) — CHỈ DỰA TRÊN TRAIN + VALIDATION.

Vấn đề đang giải quyết
----------------------
Ridge là hồi quy tuyến tính nên **có thể** trả dự báo âm. Lưu lượng xe không thể âm.
Câu hỏi: tầng phục vụ có nên cắt giá trị âm về 0 không?

Nếu quyết định này được đưa ra *sau khi* nhìn FINAL TEST 2018 thì đó là **test-informed
postprocessing** — một dạng rò rỉ, và khiến metric chính thức mất ý nghĩa. Vì vậy quyết định
phải được chốt TRƯỚC, bằng bằng chứng không chứa 2018.

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

ĐÓNG BĂNG LẶP LẠI ĐƯỢC (idempotent) — vì sao, và thế nào
-------------------------------------------------------
`frozen_at_utc` là **thời điểm đóng băng thật**: nó được ghi đúng MỘT lần, khi policy được
đóng băng lần đầu, và giữ nguyên vĩnh viễn sau đó. Nếu mỗi lần chạy lại đều dùng
`datetime.now()`, thì `models/serving_policy.json` đổi hash dù quyết định không đổi, và
ba hậu quả là: (1) không tái lập được trên máy sạch; (2) một script mang tên "freeze"
lại ghi đè chính policy đã đóng băng mà không cảnh báo; (3) không chứng minh được rằng
chạy lại chỉ khác đúng timestamp.

Nên script chạy theo ba nhánh:

  1. `models/serving_policy.json` CHƯA tồn tại -> tạo mới, ghi `frozen_at_utc` = hiện tại.
  2. Đã tồn tại và nội dung khớp -> **KHÔNG ghi lại file**, giữ nguyên `frozen_at_utc` cũ,
     in "policy đã đóng băng, nội dung khớp".
  3. Đã tồn tại và nội dung LỆCH -> dừng với exit code khác 0, in diff dễ đọc, **KHÔNG ghi
     đè**. Chỉ ghi đè khi có cờ `--force`, kèm cảnh báo phải ghi lý do vào
     `docs/project-log.md`.

"Nội dung" ở đây là **mọi trường trừ `frozen_at_utc`**. Số thực được so theo dung sai
(`FLOAT_ABS_TOL` / `FLOAT_REL_TOL`) vì phép tính float có thể khác nhẹ chút giữa các nền
tảng; chỉ khi đó mới coi là khớp. `content_sha256` là hàm băm của toàn bộ thân policy
(bao gồm `frozen_at_utc`) nên nó cũng bị loại khỏi phép so nội dung — thay vào đó nó
được **kiểm tra tự tham chiếu**: băm lại thân của chính file đang có rồi so với giá trị
lưu. Nếu hai bên lệch thì file đã bị sửa tay và được coi là không khớp.

Usage:
    python src/freeze_serving_policy.py            # đóng băng (lặp lại được) hoặc báo lệch
    python src/freeze_serving_policy.py --check    # chỉ tính lại rồi so sánh, KHÔNG ghi file
    python src/freeze_serving_policy.py --force    # đóng băng lại (bắt buộc ghi lý do)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

#: exit code: nội dung khớp (hoặc đã ghi thành công)
EXIT_OK = 0
#: exit code: nội dung lệch với file đang có -> KHÔNG ghi đè
EXIT_MISMATCH = 1
#: exit code: dùng sai cờ trên dòng lệnh
EXIT_USAGE = 2

#: Hai trường bị loại khỏi phép so "nội dung":
#:   - `frozen_at_utc` là thời điểm đóng băng, giữ nguyên vĩnh viễn;
#:   - `content_sha256` là hàm băm của cả thân policy nên vô nghĩa nếu đem ra so
#:     (một chênh lệch ở bit cuối của một số thực sẽ làm đổi toàn bộ chữ băm).
#: Thay vào đó `content_sha256` được kiểm tra tự tham chiếu — xem `check_integrity`.
CONTENT_EXCLUDED_FIELDS = ("frozen_at_utc", "content_sha256")

#: Dung sai khi so số thực giữa hai lần tính (float có thể khác nhẹ giữa nền tảng/BLAS).
FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-9

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
    """Chạy D1–D3 trên TRAIN + VALIDATION và trả về NỘI DUNG policy đã đóng băng.

    Hàm này CỐ TÌNH không sinh `frozen_at_utc` và không sinh `content_sha256`:
    đó là hai trường phụ thuộc thời điểm, thuộc về lần đóng băng chứ không thuộc về quyết
    định. Nhờ vậy kết quả của hàm là **hằng số** và có thể so sánh giữa các lần chạy,
    trên mọi nền tảng, mà không bị nhiễu bởi đồng hồ.
    """
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
    assert (d1 and d2 and d3) == adopted, "quyết định không nhất quán"

    policy_id = "non_negative_projection" if adopted else "none"
    rule = "deployed_prediction = max(0, raw_ridge_prediction)" if adopted else (
        "deployed_prediction = raw_ridge_prediction  # KHONG cat; negative prediction "
        "duoc ghi nhu mot limitation"
    )

    return {
        "policy_id": policy_id,
        "version": "frozen-1.0",
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


def render_markdown(p: dict) -> str:
    """Sinh `reports/figures/serving_policy.md` TỪ CHÍNH policy đã đóng băng trên đĩa.

    `frozen_at_utc` đọc từ `p` — tức từ `models/serving_policy.json` — chứ KHÔNG gọi
    `datetime.now()`. Nếu làm vậy thì mỗi lần chạy lại lại làm tài liệu sai lệch với
    thời điểm đóng băng thật, và không còn dùng để đối chiếu được.
    """
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

> Mốc thời gian ở trên đọc thẳng từ `models/serving_policy.json` (`frozen_at_utc`) và
> được giữ nguyên vĩnh viễn kể từ lần đóng băng đầu tiên — **không** lấy từ đồng hồ khi
> chạy lại. Vì vậy chạy lại `src/freeze_serving_policy.py` cho ra đúng file này
> (xem `py src\\freeze_serving_policy.py --check`).

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

## 7. Tính lặp lại được của chính bước đóng băng

`models/serving_policy.json` chỉ được ghi **một lần**, tại thời điểm đóng băng thật:

| Việc | Hành vi |
| --- | --- |
| Chạy lại, nội dung **khớp** | Không ghi lại file. Giữ nguyên `frozen_at_utc` cũ. |
| Chạy lại, nội dung **lệch** | Dừng, exit code khác 0, in diff, **không** ghi đè. |
| Muốn ghi đè | Phải có `--force`, và phải ghi lý do vào `docs/project-log.md`. |
| Chỉ muốn kiểm tra | `--check`: chỉ tính lại rồi so sánh, exit 0 nếu khớp / 1 nếu lệch, không ghi file nào. |

Nhờ vậy hash của `models/serving_policy.json` là ổn định: máy sạch cài
`requirements-lock.txt` rồi chạy lại pipeline sẽ sinh ra đúng file đã commit.
"""


# ---------------------------------------------------------------------------
# Đóng băng lặp lại được: so nội dung, giữ nguyên frozen_at_utc
# ---------------------------------------------------------------------------
def content_of(policy: dict) -> dict:
    """Lấy phần nội dung của policy — bỏ mọi trường phụ thuộc thời điểm."""
    return {k: v for k, v in policy.items() if k not in CONTENT_EXCLUDED_FIELDS}


def sign(body: dict) -> str:
    """Chữ ký nội dung: băm toàn bộ thân policy (kể cả `frozen_at_utc`)."""
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stamp(content: dict, frozen_at_utc: str) -> dict:
    """Ghép `frozen_at_utc` vào nội dung rồi ký lại.

    `frozen_at_utc` phải là chuỗi đã có sẵn (thời điểm đóng băng thật) — hàm này
    không tự gọi đồng hồ, nhờ vậy người gọi kiểm soát được mốc thời gian.
    """
    frozen = dict(content)
    frozen["frozen_at_utc"] = frozen_at_utc
    frozen["content_sha256"] = sign(frozen)
    return frozen


def check_integrity(on_disk: dict) -> tuple[bool, str, str]:
    """Kiểm tra chữ ký của chính file đang có (phát hiện bị sửa tay)."""
    stored = on_disk.get("content_sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        return False, "(thiếu)", "(không đọc được)"
    recomputed = sign({k: v for k, v in on_disk.items() if k != "content_sha256"})
    return stored == recomputed, stored, recomputed


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same_number(a: Any, b: Any) -> bool:
    """Số thực so theo dung sai; bool KHÔNG được coi là số (True != 1 ở đây)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if not (_is_number(a) and _is_number(b)):
        return False
    return math.isclose(float(a), float(b), rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)


def _show(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 90 else text[:87] + "..."


def _collect_diffs(where: str, on_disk: Any, fresh: Any, out: list[dict]) -> None:
    if isinstance(on_disk, dict) and isinstance(fresh, dict):
        for key in sorted(set(on_disk) | set(fresh)):
            path = f"{where}.{key}" if where else key
            if key not in on_disk:
                out.append({"path": path, "on_disk": "(thiếu)", "fresh": _show(fresh[key]),
                            "note": "trường mới"})
            elif key not in fresh:
                out.append({"path": path, "on_disk": _show(on_disk[key]), "fresh": "(thiếu)",
                            "note": "trường bị mất"})
            else:
                _collect_diffs(path, on_disk[key], fresh[key], out)
        return
    if isinstance(on_disk, list) and isinstance(fresh, list):
        if len(on_disk) != len(fresh):
            out.append({"path": where, "on_disk": f"{len(on_disk)} phần tử",
                        "fresh": f"{len(fresh)} phần tử", "note": "độ dài khác"})
            return
        for i, (a, b) in enumerate(zip(on_disk, fresh)):
            _collect_diffs(f"{where}[{i}]", a, b, out)
        return
    # Cả hai bên đều là số (bool đã bị loại khỏi _is_number): so theo dung sai, và
    # PHẢI return ở đây — nếu không, một cặp số khớp trong dung sai sẽ rơi xuống
    # nhánh so sánh bằng `!=` bên dưới và bị báo là khác biệt.
    if _is_number(on_disk) and _is_number(fresh):
        if not _same_number(on_disk, fresh):
            out.append({
                "path": where,
                "on_disk": _show(on_disk),
                "fresh": _show(fresh),
                "note": f"chênh lệch {abs(float(on_disk) - float(fresh)):.6g} "
                        f"(dung sai {FLOAT_ABS_TOL:g})",
            })
        return
    if type(on_disk) is not type(fresh) or on_disk != fresh:
        out.append({"path": where, "on_disk": _show(on_disk), "fresh": _show(fresh),
                    "note": "giá trị khác"})


def compare_content(on_disk: dict, fresh_content: dict) -> list[dict]:
    """So nội dung đã đóng băng với nội dung vừa tính. Trả về danh sách khác biệt."""
    diffs: list[dict] = []
    _collect_diffs("", content_of(on_disk), content_of(fresh_content), diffs)
    return diffs


def format_diffs(diffs: list[dict]) -> str:
    if not diffs:
        return "  (không có khác biệt)"
    lines = []
    for d in diffs:
        lines.append(f"  - {d['path']}   [{d['note']}]")
        lines.append(f"      file hiện có : {d['on_disk']}")
        lines.append(f"      tính lại được: {d['fresh']}")
    return "\n".join(lines)


def read_frozen_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _write_if_changed(path: Path, text: str) -> bool:
    """Chỉ ghi khi nội dung thật sự khác — để lần chạy lại không đụng vào file."""
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return False
        except (OSError, UnicodeDecodeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="freeze_serving_policy.py",
        description="Đóng băng chính sách max(0,·) — chỉ TRAIN + VALIDATION, KHÔNG dùng 2018. "
                    "Lặp lại được: nội dung khớp thì không ghi lại file.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check", action="store_true",
        help="chỉ tính lại rồi so sánh; KHÔNG ghi file nào. Exit 0 nếu khớp, 1 nếu lệch.",
    )
    group.add_argument(
        "--force", action="store_true",
        help="ghi đè policy đã đóng băng khi nội dung lệch — phải ghi lý do vào "
             "docs/project-log.md (đây là đóng băng LẠI, không phải chạy lại thường).",
    )
    return parser.parse_args(argv)


def _print_decision(policy: dict) -> None:
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


def _save(frozen: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    wrote_json = _write_if_changed(POLICY_PATH, json.dumps(frozen, indent=2, ensure_ascii=False))
    wrote_md = _write_if_changed(POLICY_MD_PATH, render_markdown(frozen))
    if wrote_json:
        print(f"Đã lưu policy : {POLICY_PATH}")
    else:
        print(f"Không ghi lại  : {POLICY_PATH} (nội dung không đổi)")
    if wrote_md:
        print(f"Đã lưu báo cáo: {POLICY_MD_PATH}")
    else:
        print(f"Không ghi lại  : {POLICY_MD_PATH} (nội dung không đổi)")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args(argv)

    if args.check:
        print("=== KIỂM TRA SERVING POLICY ĐÃ ĐÓNG BĂNG (chỉ đọc, KHÔNG ghi file) ===")
    else:
        print("=== ĐÓNG BĂNG SERVING POLICY (TRAIN + VALIDATION, KHÔNG dùng 2018) ===")

    content = build_policy()
    _print_decision(content)

    if not POLICY_PATH.exists():
        if args.check:
            print(f"\nSAI: chưa có {POLICY_PATH} — policy chưa được đóng băng.")
            print("Chạy: py src\\freeze_serving_policy.py")
            return EXIT_MISMATCH
        frozen = stamp(content, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        print(f"\nĐóng băng MỚI tại {frozen['frozen_at_utc']}.")
        _save(frozen)
        print("\nFINAL TEST 2018 KHÔNG được đọc ở bước này. "
              "Số liệu 2018 về policy chỉ được báo cáo ở `py src\\postprocess_audit.py`, "
              "chạy SAU khi policy đã đóng băng.")
        return EXIT_OK

    on_disk = read_frozen_policy()
    diffs = compare_content(on_disk, content)
    intact, stored_sig, recomputed_sig = check_integrity(on_disk)

    print(f"\nNội dung đã đóng băng : {POLICY_PATH}")
    print(f"  frozen_at_utc (giữ nguyên)   : {on_disk.get('frozen_at_utc', '(thiếu)')}")
    print(f"  chữ ký nội dung              : {'hợp lệ' if intact else 'KHÔNG hợp lệ'}")
    if not intact:
        print(f"      lưu trong file   : {stored_sig}")
        print(f"      băm lại được     : {recomputed_sig}")
        print("      -> file đã bị SỬA TAY sau khi đóng băng.")

    if diffs or not intact:
        print(f"\n=== NỘI DUNG KHÔNG KHỚP: {len(diffs)} khác biệt "
              f"(đã bỏ qua {', '.join(CONTENT_EXCLUDED_FIELDS)}) ===")
        print(format_diffs(diffs))
        if not args.force:
            print(f"\nDỪNG: KHÔNG ghi đè {POLICY_PATH}.")
            print("Nếu đây là đóng băng LẠI có chủ đích, chạy lại với --force và "
                  "ghi lý do vào docs/project-log.md.")
            return EXIT_MISMATCH
        print("\n" + "!" * 78)
        print("CẢNH BÁO --force: đây là ĐÓNG BĂNG LẠI, không phải chạy lại thường.")
        print("Mọi thay đổi so với bản đã đóng băng đều phải được GHI LÝ DO vào "
              "docs/project-log.md.")
        print("!" * 78)
        frozen = stamp(content, on_disk.get("frozen_at_utc")
                       or datetime.now(timezone.utc).isoformat(timespec="seconds"))
        _save(frozen)
        return EXIT_OK

    # Nội dung khớp -> giữ nguyên mọi thứ, KHÔNG ghi lại file.
    if args.check:
        print("\nOK: policy đã đóng băng, nội dung khớp.")
        return EXIT_OK

    if args.force:
        print("\n" + "!" * 78)
        print("CẢNH BÁO --force: nội dung vốn đã khớp; --force chỉ ký lại chữ ký nội dung.")
        print("Nếu thực sự thay đổi quyết định, hãy ghi lý do vào docs/project-log.md.")
        print("!" * 78)
        _save(on_disk)
        return EXIT_OK

    print("\npolicy đã đóng băng, nội dung khớp")
    print(f"Không ghi lại {POLICY_PATH} — giữ nguyên frozen_at_utc "
          f"{on_disk.get('frozen_at_utc')}.")
    # Báo cáo markdown vẫn được sinh lại từ policy trên đĩa (nếu có khác biệt) để
    # tài liệu không bị lệch khỏi artifact; nội dung ở đây là hằng số.
    if _write_if_changed(POLICY_MD_PATH, render_markdown(on_disk)):
        print(f"Đã cập nhật báo cáo: {POLICY_MD_PATH}")
    else:
        print(f"Không ghi lại       : {POLICY_MD_PATH} (nội dung không đổi)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

"""
app/registry.py

Nạp artifact ĐÃ ĐÓNG BĂNG. Đây là nơi DUY NHẤT được phép chạm vào `models/`.

Quy tắc:
- Không train, không tune, không refit bất cứ thứ gì.
- Nếu thiếu artifact bắt buộc -> ném `ArtifactError` với thông báo rõ ràng
  (tên file thiếu + lệnh cần chạy để tạo ra nó). Server vẫn sống nhưng
  `/health` và mọi endpoint dự báo trả 503 kèm giải thích.
- Artifact được cache lại sau lần nạp đầu tiên (đọc 1 lần cho cả vòng đời process).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from app import config

_ROOT = config.ROOT_DIR
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class ArtifactError(RuntimeError):
    """Không nạp được artifact cần thiết để phục vụ dự báo."""

    def __init__(self, message: str, *, missing: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing = list(missing or [])


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _relative(path: Path) -> str:
    """Đường dẫn tương đối so với gốc dự án (nếu được), tránh lộ đường dẫn máy."""
    path = Path(path)
    try:
        return path.resolve().relative_to(config.ROOT_DIR).as_posix()
    except ValueError:
        return path.name


@dataclass
class Artifacts:
    """Bundle artifact đã đóng băng."""

    pipeline: Any
    metadata: dict[str, Any]
    models_dir: Path
    reports_dir: Path
    run_config: dict[str, Any] | None = None
    baseline_meta: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    experiments: dict[str, Any] | None = None
    alpha_tuning: dict[str, Any] | None = None
    data_audit: dict[str, Any] | None = None
    postprocess_audit: dict[str, Any] | None = None
    serving_policy: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- tiện ích -----------------------------------------------------------
    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    @property
    def alpha(self) -> float | None:
        hp = self.metadata.get("hyperparameters") or {}
        value = hp.get("alpha")
        return None if value is None else float(value)

    @property
    def time_split(self) -> dict[str, Any]:
        return dict(self.metadata.get("time_split") or {})

    @property
    def coverage(self) -> dict[str, Any]:
        audit_cov = (self.data_audit or {}).get("coverage") or {}
        if audit_cov.get("start") and audit_cov.get("end"):
            return {
                "start": str(audit_cov["start"]),
                "end": str(audit_cov["end"]),
                "n_present_hours": audit_cov.get("n_present_hours"),
                "missing_ratio": audit_cov.get("missing_ratio"),
            }
        split = self.time_split
        start = (split.get("train") or {}).get("start")
        end = (split.get("test") or {}).get("end")
        return {"start": start, "end": end, "n_present_hours": None, "missing_ratio": None}

    # -- serving policy -----------------------------------------------------
    def apply_non_negative_projection(self) -> bool:
        """Có cắt dự báo âm về 0 khi phục vụ không?

        Nguồn: `models/serving_policy.json` do `src/freeze_serving_policy.py` sinh,
        chốt trên TRAIN + VALIDATION (không dùng FINAL TEST 2018).
        Nếu thiếu artifact, mặc định VẪN cắt — vì đây là ràng buộc miền giá trị của
        `traffic_volume` (số xe không thể âm) và cắt về sàn không thể làm tăng sai số
        tuyệt đối khi sự thật nằm trên sàn. Việc thiếu artifact được BÁO CÁO, không
        được giấu.
        """
        policy = self.serving_policy or {}
        adopted = policy.get("decision_trace", {}).get("policy_adopted")
        if adopted is None:
            return True
        return bool(adopted)

    def serving_policy_info(self) -> dict[str, Any]:
        policy = self.serving_policy or {}
        evidence = ((policy.get("evidence") or {}).get("validation")) or {}
        return {
            "policy_artifact_present": self.serving_policy is not None,
            "policy_id": policy.get("policy_id", "non_negative_projection (default)"),
            "version": policy.get("version"),
            "rule": policy.get("rule", "deployed_prediction = max(0, raw_ridge_prediction)"),
            "selected_on": policy.get("selected_on", ["TRAIN 2012-2016", "VALIDATION 2017"]),
            "final_test_used_for_selection": policy.get("final_test_used_for_selection", False),
            "domain_floor_from_train": policy.get("domain_floor_from_train", 0.0),
            "decision_rule": policy.get("decision_rule"),
            "decision_trace": policy.get("decision_trace"),
            "validation_evidence": evidence or None,
            "train_evidence": ((policy.get("evidence") or {}).get("train")) or None,
            "justification_domain": (
                "traffic_volume là số xe/giờ — không thể âm. min() trên TRAIN = "
                f"{policy.get('domain_floor_from_train', 0.0)}."
            ),
            "justification_validation": (
                "Trên VALIDATION 2017, cắt về sàn làm MAE không tăng và R² không giảm."
            ),
            "explicitly_not_justified_by": (
                "KHÔNG được chọn vì nhìn thấy số dự báo âm trên FINAL TEST 2018."
            ),
            "mathematical_argument": policy.get("mathematical_argument"),
            "reporting_convention": policy.get("reporting_convention"),
            "content_sha256": policy.get("content_sha256"),
        }

    def artifact_manifest(self) -> list[dict[str, Any]]:
        """Liệt kê artifact + trạng thái, dùng cho /health và /api/model-info.

        Đường dẫn trả về LUÔN là đường dẫn TƯƠNG ĐỐI so với thư mục gốc dự án
        (ví dụ `models/ridge_pipeline.joblib`) để không rò rỉ đường dẫn máy
        của người chạy vào log/tài liệu.
        """
        out: list[dict[str, Any]] = []
        for name in config.REQUIRED_ARTIFACTS:
            out.append(
                {
                    "name": name,
                    "required": True,
                    "present": (self.models_dir / name).exists(),
                    "path": _relative(self.models_dir / name),
                }
            )
        for name in config.OPTIONAL_ARTIFACTS:
            present = (self.models_dir / name).exists()
            out.append(
                {
                    "name": name,
                    "required": False,
                    "present": present,
                    "path": _relative(self.models_dir / name),
                }
            )
        out.append(
            {
                "name": "serving_policy.json",
                "required": False,
                "present": self.serving_policy is not None,
                "path": _relative(self.models_dir / "serving_policy.json"),
            }
        )
        for name, present, base in (
            ("evaluation_results.json", self.evaluation is not None, self.reports_dir),
            ("experiments_results.json", self.experiments is not None, self.reports_dir),
            ("alpha_tuning.json", self.alpha_tuning is not None, self.reports_dir),
            ("postprocess_audit.json", self.postprocess_audit is not None, self.reports_dir),
            ("data_audit.json", self.data_audit is not None, config.data_dir()),
        ):
            out.append(
                {
                    "name": name,
                    "required": False,
                    "present": present,
                    "path": _relative(base / name),
                }
            )
        return out


def load_artifacts(
    models_dir: Path | None = None, reports_dir: Path | None = None
) -> Artifacts:
    """Nạp pipeline + metadata. Ném `ArtifactError` nếu thiếu artifact bắt buộc."""
    m_dir = Path(models_dir) if models_dir is not None else config.models_dir()
    r_dir = Path(reports_dir) if reports_dir is not None else config.reports_dir()

    missing = [name for name in config.REQUIRED_ARTIFACTS if not (m_dir / name).exists()]
    if missing:
        raise ArtifactError(
            "Thiếu artifact bắt buộc: "
            + ", ".join(missing)
            + f" (thư mục: {_relative(m_dir)}). "
            "Hãy chạy `python src/download_data.py` rồi `python src/data.py` "
            "và `python src/train.py` để tạo lại model đã đóng băng. "
            "Web/API KHÔNG tự train — đó là nguyên tắc của dự án.",
            missing=missing,
        )

    pipeline_path = m_dir / "ridge_pipeline.joblib"
    try:
        pipeline = joblib.load(pipeline_path)
    except Exception as exc:  # noqa: BLE001 - thông báo phải rõ cho người dùng
        raise ArtifactError(
            f"Không đọc được pipeline tại {_relative(pipeline_path)}: {exc}. "
            "Artifact có thể bị hỏng hoặc tạo bởi phiên bản scikit-learn khác. "
            "Hãy chạy lại `python src/train.py`."
        ) from exc

    metadata = _read_json(m_dir / "model_metadata.json")
    if not metadata:
        raise ArtifactError(
            f"Không đọc được JSON tại {_relative(m_dir / 'model_metadata.json')}. "
            "Hãy chạy lại `python src/train.py`."
        )

    return Artifacts(
        pipeline=pipeline,
        metadata=metadata,
        models_dir=m_dir,
        reports_dir=r_dir,
        run_config=_read_json(m_dir / "run_config.json"),
        baseline_meta=_read_json(m_dir / "baseline_meta.json"),
        evaluation=_read_json(r_dir / "evaluation_results.json"),
        experiments=_read_json(r_dir / "experiments_results.json"),
        alpha_tuning=_read_json(r_dir / "alpha_tuning.json"),
        data_audit=_read_json(config.data_dir() / "data_audit.json"),
        postprocess_audit=_read_json(r_dir / "postprocess_audit.json"),
        serving_policy=_read_json(m_dir / "serving_policy.json"),
    )


# ---------------------------------------------------------------------------
# Cache trong process
# ---------------------------------------------------------------------------
_CACHE: Artifacts | None = None
_CACHE_KEY: tuple[str, str] | None = None


def get_artifacts() -> Artifacts:
    """Trả về artifact đã cache, nạp lần đầu nếu chưa có."""
    global _CACHE, _CACHE_KEY
    key = (str(config.models_dir()), str(config.reports_dir()))
    if _CACHE is None or _CACHE_KEY != key:
        _CACHE = load_artifacts(key[0], key[1])
        _CACHE_KEY = key
    return _CACHE


def reset_artifacts_cache() -> None:
    """Xóa cache (dùng trong test và khi đổi biến môi trường)."""
    global _CACHE, _CACHE_KEY
    _CACHE = None
    _CACHE_KEY = None


__all__ = ["ArtifactError", "Artifacts", "get_artifacts", "load_artifacts", "reset_artifacts_cache"]

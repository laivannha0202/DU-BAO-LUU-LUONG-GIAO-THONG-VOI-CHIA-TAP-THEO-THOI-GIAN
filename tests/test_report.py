"""
tests/test_report.py

Kiểm chứng tài liệu khớp artifact — chống "bịa số liệu".

Báo cáo, slide và README đều là văn bản do con người gõ tay, nên chúng dễ lệch khỏi
artifact sau nhiều lần chạy lại pipeline. Test này khóa các con số cốt lõi lại:
đọc artifact, rồi kiểm tra chúng xuất hiện đúng như vậy trong tài liệu.

Nếu một lần chạy `src/train.py` / `src/evaluate.py` cho ra số khác, các test này sẽ đỏ —
đó chính là mục đích: buộc phải cập nhật tài liệu thay vì để nó sai âm thầm.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

from conftest import requires_artifacts  # noqa: E402

FINAL_REPORT = ROOT / "reports" / "final_report.md"
SLIDES = ROOT / "docs" / "slides-outline.md"
DEMO = ROOT / "docs" / "demo-script.md"
VIVA = ROOT / "docs" / "viva-questions.md"
README = ROOT / "README.md"


def vi(value: float, digits: int = 2) -> str:
    """Định dạng số kiểu Việt Nam (dấu phẩy thập phân, dấu chấm phân tách nghìn)."""
    if digits == 0:
        return f"{int(round(value)):,}".replace(",", ".")
    s = f"{value:,.{digits}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def require_text(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"Chưa có {path.name}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Báo cáo phải chứa đúng số của FINAL TEST
# ---------------------------------------------------------------------------
@requires_artifacts
def test_final_report_matches_final_test_metrics(artifacts):
    text = require_text(FINAL_REPORT)
    ridge = artifacts.evaluation["final_test"]["ridge"]
    base = artifacts.evaluation["final_test"]["baseline"]
    for value in (ridge["MAE"], ridge["RMSE"], base["MAE"], base["RMSE"]):
        assert vi(value) in text, f"thiếu MAE/RMSE {vi(value)} trong final_report.md"
    for value in (ridge["R2"], base["R2"]):
        assert vi(value, 4) in text, f"thiếu R² {vi(value, 4)} trong final_report.md"
    # n = 6.533 và các mốc thời gian của time split
    assert vi(artifacts.time_split["test"]["n_rows"], 0) in text


@requires_artifacts
def test_final_report_matches_time_split(artifacts):
    text = require_text(FINAL_REPORT)
    for name in ("train", "validation", "test"):
        n = artifacts.time_split[name]["n_rows"]
        assert vi(n, 0) in text, f"thiếu số dòng {name} = {vi(n, 0)}"
        assert artifacts.time_split[name]["start"][:10] in text


@requires_artifacts
def test_final_report_matches_hyperparameter(artifacts):
    text = require_text(FINAL_REPORT)
    assert "alpha" in text
    assert vi(artifacts.alpha, 3) in text or f"{artifacts.alpha}" in text


@requires_artifacts
def test_final_report_never_claims_unverified_2018_single_run():
    """Báo cáo không được viết sai rằng 2018 'chỉ từng chạy đúng 1 lần'."""
    text = require_text(FINAL_REPORT)
    for bad in ["chỉ từng chạy đúng 1 lần", "chỉ từng chạy đúng một lần",
                "chạy đúng 1 lần duy nhất và không bao giờ"]:
        assert bad not in text, f"báo cáo chứa phát biểu không trung thực: {bad!r}"
    # và phải có phát biểu trung thực tương ứng
    assert "không tham gia" in text.lower() and "tuning" in text.lower()


@requires_artifacts
def test_final_report_contains_postprocess_policy(artifacts):
    text = require_text(FINAL_REPORT)
    assert "max(0, raw_prediction)" in text or "max(0, ·)" in text
    assert "clipped_to_zero" in text
    assert "freeze_serving_policy" in text, "báo cáo phải nêu script đóng băng policy"


# ---------------------------------------------------------------------------
# 2. Hạn chế bắt buộc phải có trong tài liệu
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "phrase",
    [
        "ATR 301",
        "westbound",
        "2018-09-30",
        "State Fair",
        "safety-critical",
    ],
)
def test_required_limitations_appear_in_report(phrase):
    text = require_text(FINAL_REPORT)
    assert phrase in text, f"thiếu hạn chế bắt buộc: {phrase!r}"


def test_report_separates_mandatory_team_decisions_and_limits():
    text = require_text(FINAL_REPORT)
    for tag in ("[BẮT BUỘC]", "[QUYẾT ĐỊNH]", "[HẠN CHẾ]"):
        assert tag in text, f"thiếu phân loại nội dung {tag}"


def test_report_has_enough_content_for_15_to_25_pages():
    """Báo cáo phải đủ dung lượng để thành 15–25 trang khi xuất DOCX/PDF."""
    text = require_text(FINAL_REPORT)
    words = len(text.split())
    lines = text.count("\n")
    assert words >= 4_000, f"báo cáo quá ngắn: {words} từ"
    assert lines >= 300, f"báo cáo quá ngắn: {lines} dòng"
    # 1 trang A4 ~ 500 từ -> 15 trang ≈ 7.500 từ; 25 trang ≈ 12.500 từ
    assert words >= 6_000, f"cần ít nhất ~15 trang, hiện chỉ {words} từ"
    assert words <= 22_000, f"quá dài, có thể vượt 25 trang: {words} từ"


# ---------------------------------------------------------------------------
# 3. Tài liệu trong docs/
# ---------------------------------------------------------------------------
def test_slides_outline_exists_with_10_to_12_slides():
    text = require_text(SLIDES)
    slides = re.findall(r"^##\s+Slide\s+(\d+)", text, flags=re.MULTILINE)
    assert len(slides) >= 10, f"chỉ có {len(slides)} slide"
    assert len(slides) <= 12, f"có {len(slides)} slide, quá nhiều"
    assert [int(s) for s in slides] == list(range(1, len(slides) + 1))


def test_slides_reference_real_figures():
    text = require_text(SLIDES)
    figures = list((config.reports_dir()).glob("*.png"))
    if not figures:
        pytest.skip("Chưa có hình trong reports/figures")
    referenced = [f.name for f in figures if f.name in text]
    assert referenced, "slide không tham chiếu hình nào sinh từ mã nguồn"


def test_demo_script_covers_all_eleven_steps():
    text = require_text(DEMO)
    for keyword in ["Giới thiệu", "Dataset", "Time split", "Dashboard", "Baseline",
                    "Ridge", "rò rỉ", "Form", "API", "invalid", "Model Card",
                    "Hạn chế"]:
        assert keyword.lower() in text.lower(), f"kịch bản demo thiếu mục: {keyword}"
    # phải có mốc thời gian để bảo đảm đủ 5-7 phút
    assert re.search(r"\b\d+\s*(?:s|giây)\b", text), "kịch bản demo thiếu mốc thời gian"


def test_viva_has_at_least_30_questions_with_answers():
    text = require_text(VIVA)
    questions = re.findall(r"^###\s+Câu\s+(\d+)", text, flags=re.MULTILINE)
    assert len(questions) >= 30, f"chỉ có {len(questions)} câu hỏi"
    assert [int(q) for q in questions] == list(range(1, len(questions) + 1))
    answers = re.findall(r"\*\*Đáp án", text)
    assert len(answers) >= len(questions), (
        f"chỉ {len(answers)} đáp án cho {len(questions)} câu hỏi"
    )


@requires_artifacts
def test_viva_and_slides_use_artifact_numbers(artifacts):
    for path in (SLIDES, DEMO, VIVA, README):
        text = require_text(path)
        ridge = artifacts.evaluation["final_test"]["ridge"]
        base = artifacts.evaluation["final_test"]["baseline"]
        assert vi(ridge["MAE"]) in text, f"{path.name} thiếu MAE Ridge {vi(ridge['MAE'])}"
        assert vi(base["MAE"]) in text, f"{path.name} thiếu MAE baseline {vi(base['MAE'])}"


# ---------------------------------------------------------------------------
# 4. README phải hướng dẫn chạy được trên Windows và không lộ đường dẫn cá nhân
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "command",
    [
        "py -m pip install -r requirements.txt",
        "py src\\download_data.py",
        "py src\\data.py",
        "py src\\eda.py",
        "py src\\train.py",
        "py src\\experiments.py",
        "py src\\evaluate.py",
        "py -m pytest tests\\ -v",
        "py -m uvicorn app.main:app --reload",
        "http://localhost:8000",
    ],
)
def test_readme_contains_windows_command(command):
    text = require_text(README)
    assert command in text, f"README thiếu lệnh: {command}"


@pytest.mark.parametrize(
    "path",
    [README, FINAL_REPORT, SLIDES, DEMO, VIVA,
     ROOT / "app" / "main.py", ROOT / "app" / "serving.py", ROOT / "app" / "config.py",
     ROOT / "app" / "registry.py", ROOT / "app" / "schemas.py"],
)
def test_no_personal_machine_path_in_source_or_docs(path):
    """Không được có đường dẫn cá nhân (C:\\Users\\...) trong mã nguồn hoặc tài liệu."""
    if not path.exists():
        pytest.skip(f"chưa có {path}")
    text = path.read_text(encoding="utf-8")
    assert "C:\\Users" not in text, f"{path} chứa đường dẫn cá nhân"
    assert "C:/Users" not in text, f"{path} chứa đường dẫn cá nhân"
    assert "Downloads\\traffic-forecast" not in text, f"{path} chứa đường dẫn workspace cụ thể"


def test_requirements_cover_app_dependencies():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ["fastapi", "uvicorn", "pydantic", "httpx", "joblib", "pytest",
                    "scikit-learn", "pandas", "numpy", "matplotlib"]:
        assert package in text, f"requirements.txt thiếu {package}"


# ---------------------------------------------------------------------------
# 8. Không có ký tự điều khiển lọt vào mã nguồn / tài liệu
# ---------------------------------------------------------------------------
def _control_chars(text: str) -> set[str]:
    return {c for c in text if (ord(c) < 32 and c not in "\n\t") or ord(c) == 127}


@pytest.mark.parametrize(
    "rel",
    [
        "src/export_docs.py", "src/freeze_serving_policy.py", "src/postprocess_audit.py",
        "app/serving.py", "app/registry.py", "app/main.py", "app/schemas.py",
        "app/static/js/dashboard.js", "app/static/js/predict.js",
        "README.md", "reports/final_report.md", "docs/demo-script.md",
        "docs/slides-outline.md", "docs/viva-questions.md",
    ],
)
def test_no_control_characters_in_source_or_docs(rel):
    """Ký tự điều khiển (\\x01, \\x08, ...) là bug tiếp Việt/hiển thị rất khó phát hiện."""
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"chưa có {rel}")
    bad = _control_chars(path.read_text(encoding="utf-8"))
    assert not bad, f"{rel} chứa ký tự điều khiển: {sorted(hex(ord(c)) for c in bad)}"


@pytest.mark.parametrize(
    "rel", ["README.md", "reports/final_report.md", "docs/demo-script.md",
            "docs/slides-outline.md", "docs/viva-questions.md"]
)
def test_no_mojibake_replacement_character(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"chưa có {rel}")
    assert "\ufffd" not in path.read_text(encoding="utf-8"), f"{rel} chứa ký tự hỏng (mojibake)"


# ---------------------------------------------------------------------------
# 9. Tài liệu phát hành
# ---------------------------------------------------------------------------
def test_export_script_exists():
    assert (ROOT / "src" / "export_docs.py").exists(), "thiếu src/export_docs.py"
    assert (ROOT / "requirements-export.txt").exists(), "thiếu requirements-export.txt"


def test_readme_documents_how_to_export_release_files():
    text = require_text(README)
    assert "export_docs.py" in text
    for token in ("DOCX", "PDF", "PPTX"):
        assert token in text, f"README chưa hướng dẫn xuất {token}"
    assert "requirements-export.txt" in text


def test_report_mentions_export_and_actual_page_count():
    text = require_text(FINAL_REPORT)
    assert "export_docs.py" in text
    assert "release/" in text


@pytest.mark.parametrize("name", ["final_report.docx", "final_report.pdf", "slides.pptx"])
def test_release_artifact_exists_and_is_non_trivial(name):
    """Nếu đã sinh thì phải là file thật, dung lượng hợp lý — không phải file rỗng."""
    path = ROOT / "release" / name
    if not path.exists():
        pytest.skip(f"chưa sinh {name}; chạy: py src\\export_docs.py all")
    size = path.stat().st_size
    assert size > 20_000, f"{name} có vẻ rỗng hoặc hỏng ({size:,} bytes)"
    head = path.read_bytes()[:8]
    if name.endswith(".docx"):
        assert head.startswith(b"PK"), "DOCX phải là file ZIP (PK)"
    if name.endswith(".pptx"):
        assert head.startswith(b"PK"), "PPTX phải là file ZIP (PK)"
    if name.endswith(".pdf"):
        assert head.startswith(b"%PDF"), "PDF phải bắt đầu bằng %PDF"


# ---------------------------------------------------------------------------
# 5. Cảnh báo: nếu artifact đổi, tài liệu phải được cập nhật
# ---------------------------------------------------------------------------
@requires_artifacts
def test_documented_hyperparameter_matches_artifact(artifacts):
    """README và báo cáo phải nói đúng alpha đang dùng (chấp nhận cả . và ,)."""
    variants = {
        f"alpha = {artifacts.alpha}",
        f"alpha = {vi(artifacts.alpha, 3)}",
        f"alpha={artifacts.alpha}",
        f"alpha={vi(artifacts.alpha, 3)}",
        f"`alpha` = {artifacts.alpha}",
    }
    for path in (README, FINAL_REPORT):
        text = require_text(path)
        assert any(v in text for v in variants), (
            f"{path.name} chưa nêu đúng alpha = {artifacts.alpha}"
        )

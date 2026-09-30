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
import subprocess
import sys
import unicodedata
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
PROJECT_LOG = ROOT / "docs" / "project-log.md"
LOCK_FILE = ROOT / "requirements-lock.txt"
ENVIRONMENT = ROOT / "models" / "environment.json"


def code_blocks(text: str) -> list[str]:
    """Tách các khối code fence thành cặp mở/đóng đúng (không dùng regex tham lam)."""
    blocks: list[str] = []
    current: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("```"):
            if inside:
                blocks.append("\n".join(current))
                current, inside = [], False
            else:
                inside = True
            continue
        if inside:
            current.append(line)
    return blocks


def vi(value: float, digits: int = 2) -> str:
    """Định dạng số kiểu Việt Nam (dấu phẩy thập phân, dấu chấm phân tách nghìn)."""
    if digits == 0:
        return f"{int(round(value)):,}".replace(",", ".")
    s = f"{value:,.{digits}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def require_text(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"Chưa có {path.name}")
    # Chuẩn hoá NFC: tránh so khớp thất bại chỉ vì một tệp dùng dạng dựng sẵn (NFC)
    # còn tệp khác dùng dạng phân rã (NFD) — hai dạng trông giống nhau khi đọc.
    return unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))


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


def _release(name: str) -> Path:
    path = ROOT / "release" / name
    if not path.exists():
        pytest.skip(f"chưa sinh {name}; chạy: py src\\export_docs.py all")
    return path


def test_release_docx_opens_and_has_no_raw_markdown():
    """DOCX phải mở được, nhúng hình, và không còn Markdown thô (**, backtick, [a](b))."""
    import zipfile

    path = _release("final_report.docx")
    with zipfile.ZipFile(path) as zf:
        assert zf.testzip() is None, "DOCX hỏng: không đọc được toàn bộ entry"
        names = zf.namelist()
        assert "word/document.xml" in names, "DOCX thiếu word/document.xml"
        images = [n for n in names if n.startswith("word/media/")]
        xml = zf.read("word/document.xml").decode("utf-8")
    assert images, "DOCX không nhúng hình nào"
    assert xml.count("<w:tbl>") > 5, "DOCX không có bảng nào"
    text = re.sub(r"<[^>]+>", "", xml)
    assert "**" not in text, "DOCX còn Markdown thô '**'"
    assert "`" not in text, "DOCX còn backtick thô"
    assert not re.search(r"\[[^\]]+\]\([^)]+\)", text), "DOCX còn Markdown link thô"
    assert "\ufffd" not in text, "DOCX có ký tự hỏng (mojibake)"
    assert "Claude" not in text, "DOCX còn nhắc công cụ AI không dùng"


def test_release_pdf_is_valid_and_within_page_target():
    """PDF phải hợp lệ, 15–25 trang, và nhúng font hỗ trợ tiếng Việt."""
    path = _release("final_report.pdf")
    raw = path.read_bytes()
    assert raw.startswith(b"%PDF"), "PDF không bắt đầu bằng %PDF"
    assert raw.rstrip().endswith(b"%%EOF"), "PDF bị cắt cụt (thiếu %%EOF)"
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    assert 15 <= pages <= 25, f"PDF có {pages} trang, ngoài mục tiêu 15–25"
    # font TrueType nhung => khả năng hiển thị đúng dấu tiếng Việt
    assert raw.count(b"/FontFile2") + raw.count(b"/FontFile") >= 2, (
        "PDF không nhúng font — dấu tiếng Việt có thể lỗi"
    )


def test_release_pptx_has_10_to_12_content_slides_and_no_raw_markdown():
    """PPTX: 10–12 slide nội dung, không sót Markdown thô trên thân slide."""
    import zipfile

    path = _release("slides.pptx")
    with zipfile.ZipFile(path) as zf:
        assert zf.testzip() is None, "PPTX hỏng"
        names = zf.namelist()
        slide_xmls = [n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        media = [n for n in names if n.startswith("ppt/media/")]
        all_text = "".join(
            re.sub(r"<[^>]+>", "", zf.read(n).decode("utf-8")) for n in slide_xmls
        )
    # slide 1 là bìa; các slide còn lại là slide nội dung
    content_slides = len(slide_xmls) - 1
    assert 10 <= content_slides <= 12, (
        f"PPTX có {content_slides} slide nội dung, ngoài mục tiêu 10–12"
    )
    assert media, "PPTX không nhúng hình nào"
    assert "**" not in all_text, "PPTX còn Markdown thô '**' trên thân slide"
    assert "`" not in all_text, "PPTX còn backtick thô trên thân slide"
    assert not re.search(r"\[[^\]]+\]\([^)]+\)", all_text), "PPTX còn Markdown link thô"
    assert "\ufffd" not in all_text, "PPTX có ký tự hỏng (mojibake)"


# ---------------------------------------------------------------------------
# 10. Số test trong tài liệu phải khớp số test thật (tự sinh, không gõ tay)
# ---------------------------------------------------------------------------
def vi_count(n: int) -> str:
    """Định dạng số nguyên kiểu Việt Nam (1.000 có dấu chấm phân tách nghìn)."""
    return f"{n:,}".replace(",", ".")


@pytest.fixture(scope="session")
def real_test_count() -> int:
    """Số test pytest thật sự thu thập được, đọc từ `pytest --collect-only`."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )
    match = re.search(r"(\d+)\s+tests?\s+collected", proc.stdout)
    if not match:
        pytest.fail(f"không đọc được số test từ pytest --collect-only:\n{proc.stdout[-2000:]}")
    return int(match.group(1))


DOC_FILES_WITH_TEST_COUNT = [README, FINAL_REPORT, SLIDES, DEMO, VIVA]


@pytest.mark.parametrize("path", DOC_FILES_WITH_TEST_COUNT, ids=lambda p: p.name)
def test_documented_test_count_matches_real_collection(path, real_test_count):
    """Tài liệu không được nói sai số test.

    Trước đây các file khác nhau ghi 238 / 197 / 265 / 294 cho cùng một con số. Bây giờ
    con số được lấy từ `pytest --collect-only` — thêm hay bớt một test mà quên sửa tài liệu
    là test đỏ, không thể lọt.
    """
    text = require_text(path)
    expected = vi_count(real_test_count)
    raw = str(real_test_count)
    assert (expected in text) or (raw in text), (
        f"{path.name} không nói số test thật ({expected}). "
        f"Chạy `py -m pytest tests\\ -v` rồi cập nhật con số trong tài liệu."
    )
    # và không được còn sót con số của các lần đếm cũ
    for stale in ("238 test", "197 test", "265 test"):
        assert stale not in text, f"{path.name} còn sót số test cũ: {stale!r}"


# ---------------------------------------------------------------------------
# 11. Mọi tài liệu phải mô tả đúng thứ tự pipeline chống test-informed policy
# ---------------------------------------------------------------------------
# Thứ tự bắt buộc: policy được đóng băng TRƯỚC khi FINAL TEST 2018 được mở ra.
PIPELINE_ORDER = [
    "download_data", "data", "eda", "train", "experiments",
    "freeze_serving_policy", "evaluate", "postprocess_audit",
]

PIPELINE_DOCS = [README, FINAL_REPORT, DEMO, VIVA]


@pytest.mark.parametrize("path", PIPELINE_DOCS, ids=lambda p: p.name)
def test_pipeline_order_freezes_policy_before_final_test(path):
    """Tài liệu không được kể sai thứ tự chạy.

    Nếu tài liệu mô tả `evaluate` chạy trước `freeze_serving_policy`, nó đang kể sai
    câu chuyện học thuật: chính sách hậu xử lý sẽ thành test-informed postprocessing.
    """
    text = require_text(path)
    freeze_at = text.find("freeze_serving_policy")
    evaluate_at = text.find("evaluate.py")
    assert freeze_at != -1, f"{path.name} không nhắc tới freeze_serving_policy"
    assert evaluate_at != -1, f"{path.name} không nhắc tới evaluate.py"

    # Trong khối hướng dẫn chạy, freeze_serving_policy phải xuất hiện trước evaluate.py.
    ordered_blocks = [
        b for b in code_blocks(text)
        if "freeze_serving_policy" in b and "evaluate.py" in b
    ]
    assert ordered_blocks, (
        f"{path.name} không có khối lệnh chạy nào nêu cả freeze_serving_policy và evaluate.py"
    )
    for block in ordered_blocks:
        assert block.find("freeze_serving_policy") < block.find("evaluate.py"), (
            f"{path.name}: trong khối lệnh, freeze_serving_policy phải đứng TRƯỚC evaluate.py"
        )
        # postprocess_audit chỉ đo, nên phải sau cả hai
        if "postprocess_audit" in block:
            assert block.find("postprocess_audit") > block.find("evaluate.py"), (
                f"{path.name}: postprocess_audit phải chạy SAU evaluate.py"
            )

    # Và phải nói rõ FINAL TEST không dùng để quyết định policy.
    lowered = text.lower()
    assert "final test" in lowered, f"{path.name} không nhắc FINAL TEST"


@pytest.mark.parametrize("path", PIPELINE_DOCS, ids=lambda p: p.name)
def test_report_never_claims_unverified_2018_single_run_in_docs(path):
    """Không tài liệu nào được hứa "2018 chỉ chạy đúng một lần".

    Được phép *trích câu hỏi* của GV kèm câu trả lời phủ định — đó là cách trả lời
    trung thực, không phải một tuyên bố. Bị cấm là tuyên bố khẳng định 2018 chỉ chạy
    đúng một lần.
    """
    text = require_text(path)
    claims = [
        "chỉ từng chạy đúng 1 lần", "chỉ từng chạy đúng một lần",
        "chỉ đánh giá đúng một lần", "chỉ chạy đúng 1 lần",
        "chỉ chạy đúng một lần",
    ]
    for bad in claims:
        start = 0
        while (idx := text.find(bad, start)) != -1:
            # ngữ cảnh xung quanh: câu hỏi được trích + câu trả lời phủ định -> được phép
            context = text[max(0, idx - 120): idx + 260]
            start = idx + len(bad)
            if "không khẳng định" in context or "kh\u0302ng khẳng định" in context:
                continue
            if "có chắc" in context or "hỏi" in context:
                continue
            raise AssertionError(
                f"{path.name} chứa phát biểu không trung thực: {bad!r} — "
                f"ngữ cảnh: ...{text[max(0, idx - 60):idx + 60]}..."
            )


# ---------------------------------------------------------------------------
# 12. Môi trường phải tái lập được (requirements lock + environment.json)
# ---------------------------------------------------------------------------
def test_reproducible_environment_artifacts_exist_and_are_real():
    """Phải có requirements-lock.txt và models/environment.json với phiên bản thật."""
    assert LOCK_FILE.exists(), (
        "thiếu requirements-lock.txt — chạy `py -m pip freeze > requirements-lock.txt`"
    )
    lock_lines = [
        ln.strip() for ln in LOCK_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    pinned = [ln for ln in lock_lines if "==" in ln]
    assert len(pinned) >= 10, f"requirements-lock.txt chỉ có {len(pinned)} gói được ghim phiên bản"
    assert not any(ln.startswith("-e") for ln in lock_lines), (
        "requirements-lock.txt chứa cài đặt cục bộ (-e) — không tái lập được trên máy khác"
    )
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "fastapi",
                    "uvicorn", "pydantic", "matplotlib", "pytest"):
        assert any(ln.lower().startswith(package + "==") for ln in pinned), (
            f"requirements-lock.txt chưa ghim phiên bản của {package}"
        )

    assert ENVIRONMENT.exists(), (
        "thiếu models/environment.json — môi trường đã sinh artifact phải được ghi lại"
    )
    env = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["python"]), f"python version lạ: {env['python']!r}"
    for key in ("numpy", "pandas", "scikit-learn", "joblib", "fastapi",
                "uvicorn", "pydantic", "matplotlib", "pytest"):
        value = env["packages"][key]
        assert re.fullmatch(r"\d+(\.\d+)*", value), f"{key} version lạ: {value!r}"
    assert env["seed"] == 42, f"seed phải là 42, đang là {env['seed']!r}"


def test_environment_json_matches_installed_packages():
    """models/environment.json phải mô tả môi trường đang chạy, không phải số bịa."""
    from importlib.metadata import version

    env = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    for key in ("numpy", "pandas", "scikit-learn", "joblib", "fastapi", "pydantic", "pytest"):
        installed = version(key)
        recorded = env["packages"][key]
        assert installed == recorded, (
            f"{key}: môi trường đang cài {installed} nhưng environment.json ghi {recorded}. "
            f"Cập nhật lại file này (và requirements-lock.txt) sau khi đổi môi trường."
        )


def test_requirements_lock_matches_environment_json():
    """Hai nguồn môi trường không được mâu thuẫn nhau."""
    env = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    lock = {}
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, _, ver = line.partition("==")
            lock[name.lower().replace("_", "-")] = ver
    for key, recorded in env["packages"].items():
        pinned = lock.get(key.lower().replace("_", "-"))
        assert pinned == recorded, (
            f"{key}: requirements-lock.txt ghim {pinned} nhưng environment.json ghi {recorded}"
        )


def test_readme_explains_both_install_modes():
    """README phải nói rõ khác biệt giữa cài tái lập tuyệt đối và cài theo khoảng version."""
    text = require_text(README)
    assert "requirements-lock.txt" in text
    assert "py -m pip install -r requirements-lock.txt" in text
    assert "py -m pip install -r requirements.txt" in text
    assert "project-log.md" in text, "README phải link tới docs/project-log.md"


def test_project_log_has_required_schema():
    """Nhật ký dự án phải đủ cột: tuần, người, giờ, công việc, kết quả, vấn đề."""
    text = require_text(PROJECT_LOG)
    for column in ("Tuần", "Người thực hiện", "Giờ ước lượng",
                   "Công việc", "Kết quả", "Vấn đề"):
        assert column in text, f"nhật ký dự án thiếu cột: {column}"
    assert "CẦN NGƯỜI DÙNG ĐIỀN TRƯỚC KHI NỘP" in text, (
        "nhật ký dự án phải ghi rõ chỗ nào người dùng phải điền"
    )
    assert text.count("[NGƯỜI THỰC HIỆN]") >= 1
    assert text.count("[GIỜ THỰC TẾ]") >= 1


# ---------------------------------------------------------------------------
# 13. Công cụ AI phải được khai báo thật, không để placeholder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [README, FINAL_REPORT], ids=lambda p: p.name)
def test_ai_disclosure_is_declared(path):
    """Phải khai báo công cụ AI thật đã dùng, không để placeholder kỹ thuật."""
    text = require_text(path)
    assert "[CÔNG CỤ AI" not in text, f"{path.name} còn placeholder công cụ AI"
    for tool in ("ChatGPT", "Pi Agent"):
        assert tool in text, f"{path.name} thiếu khai báo công cụ AI: {tool}"
    assert "Cách kiểm chứng" in text or "kiểm chứng" in text


# ---------------------------------------------------------------------------
# 14. Chính tả: "rò rỉ" (KHÔNG phải "rò rễ")
# ---------------------------------------------------------------------------
#: Chỉ quét các file văn bản; file sinh tự động cũng phải sạch vì chúng được commit.
SPELLING_SCAN_GLOBS = [
    "README.md",
    "reports/final_report.md",
    "reports/project_brief.md",
    "data/README.md",
    "data/data_dictionary.md",
    "docs/*.md",
    "src/*.py",
    "app/*.py",
    "app/templates/*.html",
    "app/static/js/*.js",
    "reports/figures/*.md",
    "reports/figures/*.json",
]

#: Cụm viết sai. Chỉ dùng đúng cụm này — không khửng hồ mọi từ có vần "ễ".
MISSPELLED_TERM = "rò rễ"


def _spelling_scan_paths() -> list[Path]:
    found: list[Path] = []
    for pattern in SPELLING_SCAN_GLOBS:
        found.extend(sorted(ROOT.glob(pattern)))
    return [p for p in found if p.is_file()]


def test_no_misspelled_leakage_term_anywhere_in_repo():
    """'rò rỉ' phải viết đúng; 'rò rễ' là lỗi chính tả và bị cấm toàn repo.

    Lỗi này rất dễ tái xuất hiện khi sao chép đoạn văn bản, nên cần một test chặn.
    """
    offenders: list[str] = []
    for path in _spelling_scan_paths():
        text = path.read_text(encoding="utf-8")
        if MISSPELLED_TERM in text:
            lines = [
                str(i + 1) for i, ln in enumerate(text.splitlines())
                if MISSPELLED_TERM in ln
            ]
            offenders.append(f"{path.relative_to(ROOT).as_posix()}: dòng {', '.join(lines)}")
    assert not offenders, (
        f"Tìm thấy '{MISSPELLED_TERM}' (sai; phải viết 'rò rỉ') tại:\n  " + "\n  ".join(offenders)
    )


def test_spelling_scan_actually_finds_files():
    """Bảo đảm glob của test trên không rỗng — nếu rỗng thì test trên là 'xanh giả'."""
    paths = _spelling_scan_paths()
    assert len(paths) >= 12, f"Glob quét chỉ tìm thấy {len(paths)} file — kiểm tra lại danh sách"
    names = {p.name for p in paths}
    assert "final_report.md" in names
    assert "experiments.py" in names


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

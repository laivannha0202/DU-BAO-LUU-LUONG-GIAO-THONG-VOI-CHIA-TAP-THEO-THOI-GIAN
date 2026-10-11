"""
src/check_submission.py

DANH SÁCH KIỂM TRA TRƯỚC KHI NỘP — script CHỈ BÁO CÁO, không tự điền gì.

Chạy:
    python src/check_submission.py

Nó KHÔNG nằm trong pytest (không phải test) và không sửa file nào. Việc của nó là nhắc
con người những ô còn trống phải điền tay trước khi nộp:

  - các placeholder còn sót ([TÊN THÀNH VIÊN, [NGƯỜI THỰC HIỆN, [NGÀY, [GIỜ THỰC TẾ,
    [PHÂN CÔNG, "CẦN NGƯỜI DÙNG CUNG CẤP) trong README.md, docs/, reports/final_report.md;
  - số tác giả khác nhau trong `git shortlog`;
  - số trang PDF và số slide của bản phát hành.

Thoát mã 1 nếu còn placeholder (để có thể dùng trong CI/pre-commit), ngược lại 0.
"""
from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Các cụm placeholder mà con người PHẢI điền. Script không bao giờ tự điền.
PLACEHOLDERS = [
    "[TÊN THÀNH VIÊN",
    "[NGƯỜI THỰC HIỆN",
    "[NGÀY",
    "[GIỜ THỰC TẾ",
    "[PHÂN CÔNG",
    "CẦN NGƯỜI DÙNG CUNG CẤP",
]

#: File cần quét placeholder.
SCAN_FILES = [
    ROOT / "README.md",
    ROOT / "reports" / "final_report.md",
]
SCAN_DIRS = [ROOT / "docs"]


def scan_placeholders() -> list[str]:
    """Trả về danh sách '<file>: dòng' còn chứa placeholder."""
    hits: list[str] = []
    files: list[Path] = [p for p in SCAN_FILES if p.exists()]
    for d in SCAN_DIRS:
        if d.exists():
            files.extend(sorted(d.rglob("*.md")))
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for ph in PLACEHOLDERS:
                if ph in line:
                    hits.append(f"{path.relative_to(ROOT).as_posix()}:{i}: {line.strip()[:120]}")
                    break
    return hits


def git_authors() -> list[str]:
    """Số tác giả khác nhau trong `git shortlog -sne` (dấu hiệu phân công Git)."""
    try:
        proc = subprocess.run(
            ["git", "shortlog", "-sne", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        return [f"(không chạy được git shortlog: {e})"]
    if proc.returncode != 0:
        return [f"(git shortlog lỗi: {proc.stderr.strip()[:120]})"]
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def pdf_page_count() -> str:
    """Đếm số trang PDF bằng cách đếm object /Type /Page (không cần thư viện ngoài)."""
    path = ROOT / "release" / "final_report.pdf"
    if not path.exists():
        return "(chưa có release/final_report.pdf — chạy: py src\\export_docs.py pdf)"
    raw = path.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    return f"{pages} trang (mục tiêu 15–25 trang A4)"


def pptx_slide_count() -> str:
    """Đếm số slide trong PPTX (tổng và số slide nội dung)."""
    path = ROOT / "release" / "slides.pptx"
    if not path.exists():
        return "(chưa có release/slides.pptx — chạy: py src\\export_docs.py pptx)"
    with zipfile.ZipFile(path) as zf:
        slides = [n for n in zf.namelist()
                  if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
    return f"{len(slides)} slide tổng (1 bìa + {max(0, len(slides) - 1)} nội dung; mục tiêu 10–12)"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 74)
    print("KIỂM TRA TRƯỚC KHI NỘP (chỉ báo cáo — script KHÔNG tự điền gì)")
    print("=" * 74)

    print("\n1) Placeholder còn sót (con người phải điền):")
    hits = scan_placeholders()
    if hits:
        for h in hits:
            print("   [THIẾU] " + h)
    else:
        print("   (không còn placeholder)")

    print("\n2) Tác giả Git khác nhau trong `git shortlog -sne HEAD`:")
    authors = git_authors()
    for a in authors:
        print("   " + a)
    print(f"   -> {len([a for a in authors if not a.startswith('(')])} tác giả. "
          "Nếu đề yêu cầu mỗi thành viên có commit riêng, phần đó do CHÍNH thành viên đó commit.")

    print("\n3) Bản phát hành:")
    print("   PDF  : " + pdf_page_count())
    print("   Slide: " + pptx_slide_count())

    print("\n" + "-" * 74)
    if hits:
        print(f"KẾT LUẬN: CÒN {len(hits)} chỗ placeholder — PHẢI điền tay trước khi nộp.")
        print("(Script này KHÔNG bịa tên, số giờ, ngày hay phân công.)")
        return 1
    print("KẾT LUẬN: không còn placeholder trong các file được quét.")
    print("Nhắc lại: vẫn cần con người tự kiểm tra lịch sử Git và khai báo công cụ AI đầy đủ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

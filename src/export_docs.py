"""
src/export_docs.py

Xuất tài liệu phát hành: **DOCX**, **PDF**, **PPTX** từ các file Markdown trong dự án.

Vì sao cần script này
---------------------
Báo cáo và slide được soạn bằng Markdown vì dễ kiểm chứng bằng test
(`tests/test_report.py` đối chiếu số liệu với artifact). Khi nộp, cần bản DOCX/PDF/PPTX.
Viết tay bằng Word/PowerPoint sẽ **tách** Markdown khỏi artifact và dễ sai số — nên
script này **sinh ra từ chính các file Markdown đó**, không ai chép số tay.

Nguồn:
    reports/final_report.md   ->  final_report.docx / final_report.pdf
    docs/slides-outline.md    ->  slides.pptx

Công cụ (cài riêng bằng `py -m pip install -r requirements-export.txt`):
    DOCX  : pandoc (qua `pypandoc-binary`)
    PDF   : reportlab (tự dịch Markdown -> flowables, font Arial để hiện dấu tiếng Việt)
    PPTX  : python-pptx

Nếu thiếu công cụ nào, script **báo rõ và bỏ qua** phần đó — KHÔNG tạo file rỗng và
KHÔNG báo là đã xuất xong.

Usage:
    py src\\export_docs.py docx
    py src\\export_docs.py pdf
    py src\\export_docs.py pptx
    py src\\export_docs.py all
    py src\\export_docs.py          # mặc định: all
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPORTS_DIR = _ROOT / "reports"
DOCS_DIR = _ROOT / "docs"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORT_MD = REPORTS_DIR / "final_report.md"
SLIDES_MD = DOCS_DIR / "slides-outline.md"
VIVA_MD = DOCS_DIR / "viva-questions.md"
DEMO_MD = DOCS_DIR / "demo-script.md"

OUT_DIR = _ROOT / "release"

#: Regex gạch nghiêng Markdown (*text*) — dựng bằng chr(92) để tránh lỗi escape
_ITALIC_RE = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")

#: Phụ lục ảnh ở cuối PDF (báo cáo đã nhúng ảnh tại đúng chỗ rồi)
WITH_FIGURE_APPENDIX = False


# ===========================================================================
# Tiện ích Markdown dùng chung
# ===========================================================================
def slugify(text: str) -> str:
    text = re.sub(r"[*_`#>]", "", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "-", text.strip()).lower()[:80]


def parse_blocks(md: str) -> list[tuple[str, object]]:
    """Markdown -> danh sách (loại, payload). Đủ dùng cho báo cáo học thuật."""
    lines = md.splitlines()
    blocks: list[tuple[str, object]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", (lang, "\n".join(buf))))
            continue

        # table
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            header = _split_row(stripped)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i].strip()))
                i += 1
            blocks.append(("table", (header, rows)))
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            blocks.append(("heading", (len(m.group(1)), m.group(2).strip())))
            i += 1
            continue

        # horizontal rule
        if re.match(r"^(-{3,}|\*{3,})$", stripped):
            blocks.append(("hr", None))
            i += 1
            continue

        # blockquote (gộp nhiều dòng)
        if stripped.startswith(">"):
            buf = []
            while i < n and (lines[i].strip().startswith(">") or
                             (buf and lines[i].strip() == "")):
                if lines[i].strip().startswith(">"):
                    buf.append(lines[i].strip().lstrip(">").strip())
                elif buf and buf[-1]:
                    buf[-1] += " " + lines[i].strip()
                i += 1
            blocks.append(("quote", " ".join(x for x in buf if x)))
            continue

        # list
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            items: list[tuple[int, str, str]] = []
            while i < n and re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]):
                m2 = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", lines[i])
                indent = len(m2.group(1)) // 2
                checked = ""
                body = m2.group(3)
                if body.startswith("[ ] "):
                    checked, body = "todo", body[4:]
                elif body.startswith("[x] ") or body.startswith("[X] "):
                    checked, body = "done", body[4:]
                items.append((indent, checked, body))
                i += 1
            blocks.append(("list", items))
            continue

        # paragraph (gộp dòng trống)
        if stripped:
            buf = [stripped]
            i += 1
            while i < n and lines[i].strip() and not re.match(
                r"^\s*(#{1,6}\s|\||>|```|\s*([-*+]|\d+\.)\s)", lines[i]
            ):
                buf.append(lines[i].strip())
                i += 1
            blocks.append(("para", " ".join(buf)))
            continue

        i += 1
    return blocks


def _split_row(row: str) -> list[str]:
    cells = row.strip().strip("|").split("|")
    return [c.strip() for c in cells]


# ===========================================================================
# DOCX — qua pandoc
# ===========================================================================
def export_docx(verbose: bool = True) -> Path | None:
    try:
        import pypandoc
    except ImportError:
        print("  [DOCX] BO QUA — thieu pypandoc. Cai: py -m pip install -r requirements-export.txt")
        return None

    if not REPORT_MD.exists():
        print(f"  [DOCX] BO QUA — khong tim thay {REPORT_MD.name}")
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "final_report.docx"
    # resource_path = thư mục gốc để ảnh reports/figures/*.png được nhúng
    extra = ["--resource-path=" + str(_ROOT), "--standalone", "--toc", "--toc-depth=2"]
    if (ROOT_STYLE := _ROOT / "reference.docx").exists():
        extra.append(f"--reference-doc={ROOT_STYLE}")
    pypandoc.convert_file(
        str(REPORT_MD), "docx", outputfile=str(out),
        extra_args=extra, format="gfm",
    )
    if verbose:
        print(f"  [DOCX] OK  {out.relative_to(_ROOT)}  ({out.stat().st_size:,} bytes)")
    return out


# ===========================================================================
# PDF — qua reportlab, tự dịch Markdown
# ===========================================================================
def _pdf_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        ("Body", r"C:\Windows\Fonts\arial.ttf"),
        ("Body-Bold", r"C:\Windows\Fonts\arialbd.ttf"),
        ("Body-Italic", r"C:\Windows\Fonts\ariali.ttf"),
        ("Body-BoldItalic", r"C:\Windows\Fonts\arialbi.ttf"),
        ("Mono", r"C:\Windows\Fonts\consola.ttf"),
    ]
    installed = {}
    for name, path in candidates:
        p = Path(path)
        if p.exists():
            try:
                pdfmetrics.registerFont(TTFont(name, str(p)))
                installed[name] = name
            except Exception:  # noqa: BLE001
                pass
    return installed


def _inline(text: str, styles: dict, base: str = "Body", size: float = 9.5):
    """Markdown inline -> list các đoạn reportlab. Giữ bản in đơn giản nhưng đủ dấu."""
    from reportlab.lib import colors

    out = []
    # `code`
    parts = re.split(r"(`[^`]+`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append((part[1:-1], "Mono", size - 0.5, colors.Color(0.25, 0.25, 0.28)))
        else:
            # **bold**
            for sub in re.split(r"(\*\*[^*]+\*\*)", part):
                if not sub:
                    continue
                if sub.startswith("**") and sub.endswith("**") and len(sub) > 4:
                    out.append((sub[2:-2], "Body-Bold", size, colors.black))
                else:
                    out.append((sub, base, size, colors.black))
    return out


def export_pdf(verbose: bool = True) -> Path | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_JUSTIFY
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak,
                                        PageTemplate, Paragraph, Spacer, Table, TableStyle)
    except ImportError:
        print("  [PDF] BO QUA — thieu reportlab. Cai: py -m pip install -r requirements-export.txt")
        return None

    if not REPORT_MD.exists():
        print(f"  [PDF] BO QUA — khong tim thay {REPORT_MD.name}")
        return None

    _pdf_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "final_report.pdf"

    body_font = "Body"
    styles = {
        "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=15, leading=19,
                             spaceBefore=12, spaceAfter=7, textColor=colors.HexColor("#1f4e79")),
        "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=12, leading=15.5,
                             spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1f4e79")),
        "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=10.5, leading=14,
                             spaceBefore=8, spaceAfter=4),
        "h4": ParagraphStyle("h4", fontName="Body-Bold", fontSize=9.8, leading=13,
                             spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#44484d")),
        "body": ParagraphStyle("body", fontName=body_font, fontSize=9.2, leading=12.9,
                               alignment=TA_JUSTIFY, spaceAfter=4),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=7.8, leading=10.2,
                               backColor=colors.HexColor("#f5f6f8"), borderPadding=5,
                               leftIndent=5, rightIndent=5, spaceAfter=6),
        "quote": ParagraphStyle("quote", fontName=body_font, fontSize=9, leading=12.6,
                                leftIndent=9, textColor=colors.HexColor("#3a3f45"),
                                borderColor=colors.HexColor("#1f4e79"), borderWidth=0,
                                spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName=body_font, fontSize=7.5, leading=9.6),
        "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8, leading=10.4),
    }

    story: list = []
    blocks = parse_blocks(REPORT_MD.read_text(encoding="utf-8"))

    def strip_md_links_only(text: str) -> str:
        """Chi bo lien ket `[text](url)`, giu lai `**dam**` va `` `code` ``."""
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\[([^\]]*)\]", r"\1", text)
        text = re.sub(r"^\s*[-*]\s+\[[ xX]\]\s*", "", text, flags=re.MULTILINE)
        return text

    def strip_md(t: str) -> str:
        """Bo TOAN BO cu phap Markdown con sot (dung cho code block)."""
        t = strip_md_links_only(t)
        t = t.replace("**", "").replace("`", "")
        t = _ITALIC_RE.sub(r"\1", t)
        return t.strip()

    def esc(t: str) -> str:
        return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    def strip_md_links_only(text: str) -> str:
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\[([^\]]*)\]", r"\1", text)
        text = re.sub(r"^\s*[-*]\s+\[[ xX]\]\s*", "", text, flags=re.MULTILINE)
        return text

    def rich(text: str, style: ParagraphStyle, base_bold: bool = False) -> Paragraph:
        """Paragraph có render **đậm** và `code` thật sự (không để lộ ký hiệu Markdown)."""
        runs = []
        for seg in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", strip_md_links_only(text)):
            if not seg:
                continue
            if seg.startswith("**") and seg.endswith("**") and len(seg) > 4:
                runs.append((seg[2:-2], "Body-Bold", style.fontSize, colors.black))
            elif seg.startswith("`") and seg.endswith("`") and len(seg) > 2:
                runs.append((seg[1:-1], "Mono", style.fontSize - 0.7,
                             colors.Color(0.25, 0.25, 0.28)))
            else:
                runs.append((seg.replace("~~", ""),
                             "Body-Bold" if base_bold else "Body",
                             style.fontSize, colors.black))
        return Paragraph("".join(
            f'<font name="{fn}" size="{sz}" color="{col.hexval()}">{esc(txt)}</font>'
            for txt, fn, sz, col in runs
        ), style)

    def para(text: str, style: ParagraphStyle) -> Paragraph:
        # bỏ liên kết + gạch ngang trước khi render để không lộ ký hiệu Markdown
        cleaned = strip_md_links_only(text).replace("~~", "")
        return Paragraph("".join(
            f'<font name="{fn}" size="{sz}" color="{col.hexval()}">{esc(t)}</font>'
            for t, fn, sz, col in _inline(cleaned, styles)
        ), style)

    for kind, payload in blocks:
        if kind == "heading":
            level, text = payload
            style = styles["h1"] if level == 1 else styles["h2"] if level == 2 else \
                styles["h3"] if level == 3 else styles["h4"]
            # Chỉ ngắt trang ở các mục lớn thật sự — ngắt ở MỌI H1 sẽ lãng phí
            # nhiều trang trắng và làm báo cáo dài hơn mức cần thiết.
            if level == 1 and story and re.match(
                r"^(Phụ lục|Kiểm thử tự động|Model Card)\b", text
            ):
                story.append(PageBreak())
            story.append(para(text, style))
        elif kind == "para":
            story.append(para(payload, styles["body"]))
        elif kind == "quote":
            story.append(para(payload, styles["quote"]))
        elif kind == "code":
            _lang, code = payload
            body = esc(code).replace("\n", "<br/>")
            story.append(Paragraph(body, styles["code"]))
        elif kind == "hr":
            story.append(Spacer(1, 4))
        elif kind == "list":
            items = payload
            for indent, checked, text in items:
                mark = {"done": "[x] ", "todo": "[ ] "}.get(checked, "")
                bullet = "• " if indent == 0 else "– "
                story.append(para(bullet + mark + text, styles["body"]))
        elif kind == "table":
            header, rows = payload
            ncol = max([len(header)] + [len(r) for r in rows]) or 1
            data = [[rich(h, styles["cell"], base_bold=True) for h in header]]
            for r in rows:
                r = list(r) + [""] * (ncol - len(r))
                data.append([rich(cell, styles["cell"]) for cell in r])
            avail = A4[0] - 30 * mm
            t = Table(data, colWidths=[avail / ncol] * ncol, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9ced4")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
            ]))
            story += [t, Spacer(1, 6)]

    # --- nhúng hình minh hoạ từ reports/figures ---
    pngs = sorted(FIGURES_DIR.glob("*.png")) if WITH_FIGURE_APPENDIX else []
    if pngs:
        story += [PageBreak(), para("PHỤ LỤC — HÌNH MINH HOẠ", styles["h1"])]
        for png in pngs:
            if png.stat().st_size < 8000:
                continue
            story.append(para(png.stem, styles["h3"]))
            try:
                img = Image(str(png), width=avail, height=avail * 0.58)
                img.hAlign = "CENTER"
                story += [img, Spacer(1, 8)]
            except Exception:  # noqa: BLE001
                pass

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(colors.HexColor("#767d87"))
        canvas.drawString(15 * mm, 11 * mm,
                           "Dự báo lưu lượng giao thông I-94 (westbound) — ATR 301 · Project 20 Bài 7")
        canvas.drawRightString(A4[0] - 15 * mm, 11 * mm, f"Trang {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#d9dde2"))
        canvas.line(15 * mm, 14 * mm, A4[0] - 15 * mm, 14 * mm)
        canvas.restoreState()

    doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                          topMargin=15 * mm, bottomMargin=18 * mm,
                          title="Báo cáo — Dự báo lưu lượng giao thông I-94 (ATR 301)",
                          author="Project 20")
    frame = Frame(15 * mm, 18 * mm, A4[0] - 30 * mm, A4[1] - 33 * mm, id="main")
    doc.addPageTemplates([PageTemplate(id="body", frames=[frame], onPage=_footer)])
    doc.build(story)
    if verbose:
        import re as _re
        n_pages = (_re.findall(rb"/Count\s+(\d+)", out.read_bytes()) or [b"?"])[0]
        n_pages = int(n_pages) if n_pages.isdigit() else "?"
        print(f"  [PDF]  OK  {out.relative_to(_ROOT)}  "
              f"({n_pages} trang A4, {out.stat().st_size:,} bytes)")
    return out


# ===========================================================================
# PPTX — qua python-pptx, dựng từ docs/slides-outline.md
# ===========================================================================
def export_pptx(verbose: bool = True) -> Path | None:
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Emu, Inches, Pt
    except ImportError:
        print("  [PPTX] BO QUA — thieu python-pptx. Cai: py -m pip install -r requirements-export.txt")
        return None

    if not SLIDES_MD.exists():
        print(f"  [PPTX] BO QUA — khong tim thay {SLIDES_MD.name}")
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "slides.pptx"

    md = SLIDES_MD.read_text(encoding="utf-8")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    NAVY = RGBColor(0x1F, 0x4E, 0x79)
    INK = RGBColor(0x1A, 0x1D, 0x21)
    SOFT = RGBColor(0x4A, 0x50, 0x58)
    FAINT = RGBColor(0x76, 0x7D, 0x87)
    WARN = RGBColor(0x8A, 0x5A, 0x00)

    blank = prs.slide_layouts[6]

    def add_slide():
        return prs.slides.add_slide(blank)

    def txbox(slide, left, top, width, height):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = box.text_frame
        tf.word_wrap = True
        return tf

    # Tách inline: **đậm** và `code`. Nhánh đầu cho phép `code` NẰM TRONG **đậm**
    # (kiểu **CHUNG hàm `src.features.build_features`**) — nếu không, dấu backtick
    # sẽ lọt thẳng ra slide.
    _INLINE_RE = re.compile(r"(\*\*(?:`[^`]+`|[^*`])+\*\*|`[^`]+`)")

    def add_text(tf, text, size, bold=False, color=INK, space_after=6, first=False):
        """Thêm một đoạn, hỗ trợ **đậm** và `code`."""
        par = tf.paragraphs[0] if first else tf.add_paragraph()
        for part in _INLINE_RE.split(text):
            if not part:
                continue
            r = par.add_run()
            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                # `code` lồng trong **đậm**: bỏ backtick, giữ chữ đậm
                r.text = part[2:-2].replace("`", "")
                is_bold, col, sz, font = True, color, size, "Calibri"
            elif part.startswith("`") and part.endswith("`") and len(part) > 2:
                r.text = part[1:-1]
                is_bold, col, sz, font = False, RGBColor(0x33, 0x37, 0x3C), size - 1, "Consolas"
            else:
                r.text = part
                is_bold, col, sz, font = bold, color, size, "Calibri"
            r.font.size = Pt(sz)
            r.font.bold = is_bold
            r.font.color.rgb = col
            r.font.name = font
        par.space_after = Pt(space_after)
        return par

    # ---- Slide tiêu đề ----
    s = add_slide()
    tf = txbox(s, 0.9, 2.3, 11.5, 2.4)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = "Dự báo lưu lượng giao thông I-94 (chiều westbound) — trạm ATR 301"
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = "Calibri"
    p2 = tf.add_paragraph()
    r = p2.add_run(); r.text = "Project 20 — Bài 7: Rò rỉ dữ liệu, chia tập đúng và đánh giá trung thực"
    r.font.size = Pt(16); r.font.color.rgb = SOFT; r.font.name = "Calibri"
    p3 = tf.add_paragraph()
    r = p3.add_run(); r.text = "Dữ liệu: Metro Interstate Traffic Volume (UCI, CC BY 4.0)"
    r.font.size = Pt(13); r.font.color.rgb = FAINT; r.font.name = "Calibri"

    # ---- Các slide theo outline ----
    sections = re.split(r"^##\s+", md, flags=re.MULTILINE)[1:]
    n_slide = 0
    for sec in sections:
        lines = sec.splitlines()
        if not re.match(r"^Slide\s+\d+", lines[0].strip()):
            continue  # phan "Phu luc" khong phai slide danh so
        title = re.sub(r"^Slide\s+\d+\s*[—\-–:]\s*", "", lines[0].strip())
        n_slide += 1
        s = add_slide()

        bar = s.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = NAVY
        bar.line.fill.background()

        tf = txbox(s, 0.55, 0.28, 12.2, 0.95)
        add_text(tf, f"{n_slide}. {title}", 24, bold=True, color=NAVY, space_after=0, first=True)

        tf = txbox(s, 0.55, 1.35, 7.0, 5.4)
        first = True
        # Cac khoi "ghi chu" (phan biet ra phien ban day du) — bo ca khoi, khong chi
        # dong dau: cac dong tiep theo cua mot doan **Noi:** cung la ghi chu, neu
        # chi bo dong dau thi phan con lai lot len slide.
        note_prefixes = ("**Hình", "**Nói", "**Câu", "**Dự phòng",
                         "**Danh mục", "**Công cụ")
        for block in re.split(r"\n[ \t]*\n", "\n".join(lines[1:])):
            block_lines = [ln for ln in block.splitlines() if ln.strip()]
            if not block_lines:
                continue
            head = block_lines[0].strip()
            if head.startswith(">") or head.startswith(note_prefixes):
                continue
            for st in block_lines:
                st = st.strip()
                if not st or st.startswith("|"):
                    continue
                m = re.match(r"^([-*])\s+(.*)$", st)
                if m:
                    st = "• " + m.group(2)
                else:
                    st = re.sub(r"^\d+\.\s+", "", st)
                add_text(tf, st, 13, space_after=5, first=first)
                first = False
        if first:
            add_text(tf, "(xem tài liệu bản đầy đủ)", 12, color=FAINT, first=True)

        # hình minh hoạ thật nếu slide có nhắc tới file trong reports/figures
        ref = " ".join(lines)
        for cname in dict.fromkeys(re.findall(r"([a-z0-9_]+\.png)", ref)):
            f = FIGURES_DIR / cname
            if f.exists() and f.stat().st_size > 8000:
                s.shapes.add_picture(str(f), Inches(7.8), Inches(1.45), width=Inches(5.0))
                cap = txbox(s, 7.8, 5.45, 5.0, 0.5)
                add_text(cap, cname, 9, color=FAINT, space_after=0, first=True)
                break

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    if verbose:
        print(f"  [PPTX] OK  {out.relative_to(_ROOT)}  "
              f"({len(prs.slides._sldIdLst)} slide, {out.stat().st_size:,} bytes)")
    return out


# ===========================================================================
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    targets = sys.argv[1:] or ["all"]
    if "all" in targets:
        targets = ["docx", "pdf", "pptx"]

    print("=== XUẤT TÀI LIỆU PHÁT HÀNH ===")
    made: list[Path] = []
    for t in targets:
        if t == "docx":
            r = export_docx()
        elif t == "pdf":
            r = export_pdf()
        elif t == "pptx":
            r = export_pptx()
        else:
            print(f"  Khong ro lenh '{t}'. Dung: docx | pdf | pptx | all")
            continue
        if r:
            made.append(r)

    if made:
        print(f"\nĐã tạo {len(made)} tệp trong {OUT_DIR.relative_to(_ROOT)}:")
        for m in made:
            print(f"  - {m.name}  ({m.stat().st_size:,} bytes)")
        print("\nTài liệu được SINH TỰ ĐỘNG từ Markdown; số liệu lấy từ artifact, không chép tay.")
    else:
        print("\nKhông tạo được tệp nào. Xem thông báo BO QUA ở trên để biết thiếu gì.")


if __name__ == "__main__":
    main()

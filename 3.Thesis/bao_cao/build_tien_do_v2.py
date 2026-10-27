#!/usr/bin/env python3
"""Báo cáo tiến độ theo bố cục luận văn mẫu và số liệu đang chạy.

Bìa theo section đầu của luận văn Nguyễn Vĩnh Quang (A4, lề 3/1,5/2/2, logo).
Phần chữ theo section nội dung của luận văn đó: khổ 21,59×27,94 cm,
lề 3,5/2,5/3/3 cm, Times New Roman 13, dãn dòng 1,5, chương 14 pt.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "1.Documents" / "Bao_cao_tien_do_v2.docx"
LOGO = Path(__file__).resolve().parent / "logo_tdmu.png"
FIG = Path(__file__).resolve().parent / "figures"
RESULTS = ROOT / "3.Thesis" / "models" / "yolo12_classroom" / "results.csv"
FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"

NAVY = "#1B3A4B"
TEAL = "#1F6F6A"
SAND = "#8C6A3D"
RED = "#8C3A3A"
GRAY = "#5C6770"
LINE = "#D5DDE3"

TITLE_LINES = (
    "ỨNG DỤNG THỊ GIÁC MÁY TÍNH HỖ TRỢ ĐIỂM DANH",
    "VÀ THEO DÕI HÀNH VI HỌC SINH",
    "TẠI TRƯỜNG TIỂU HỌC VÀ TRUNG HỌC CƠ SỞ",
)


def set_run_font(run, size=13, bold=False, italic=False, name="Times New Roman"):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(attr), name)


def shade_black(style):
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.name = "Times New Roman"
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(attr), "Times New Roman")


def configure(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(13)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    pf = normal.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.first_line_indent = Cm(0)
    shade_black(normal)
    for style_name, size, center, before in (
        ("Heading 1", 14, True, 0),
        ("Heading 2", 13, False, 12),
        ("Heading 3", 13, False, 8),
    ):
        style = doc.styles[style_name]
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.italic = False
        shade_black(style)
        style.paragraph_format.alignment = (
            WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
        )
        style.paragraph_format.first_line_indent = Cm(0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.page_break_before = False


def set_margins(section, *, cover: bool = False) -> None:
    section.orientation = WD_ORIENT.PORTRAIT
    if cover:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(1.5)
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.header_distance = Cm(0.49)
        section.footer_distance = Cm(1.0)
    else:
        # Đúng section nội dung của luận văn mẫu.
        section.page_width = Cm(21.59)
        section.page_height = Cm(27.94)
        section.left_margin = Cm(3.5)
        section.right_margin = Cm(2.5)
        section.top_margin = Cm(3.0)
        section.bottom_margin = Cm(3.0)
        section.header_distance = Cm(1.27)
        section.footer_distance = Cm(1.27)


def _footer_page(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run()
    set_run_font(run, 13)
    for kind, text in (("begin", None), ("instr", " PAGE "), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        run._r.append(el)


def set_page_fmt(section, fmt: str | None, start: int | None) -> None:
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if fmt is None:
        if pg is not None:
            sectPr.remove(pg)
        return
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sectPr.append(pg)
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))


def P(doc, text, *, center=False, bold=False, size=13, indent=True, space_before=None, space_after=None, italic=False):
    para = doc.add_paragraph()
    para.style = doc.styles["Normal"]
    if center:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0) if not indent or center else Cm(1.0)
    if space_before is not None:
        para.paragraph_format.space_before = Pt(space_before)
    if space_after is not None:
        para.paragraph_format.space_after = Pt(space_after)
    run = para.add_run(text)
    set_run_font(run, size, bold=bold, italic=italic)
    return para


def H(doc, text, level=1):
    para = doc.add_heading(text, level=level)
    size = 14 if level == 1 else 13
    for run in para.runs:
        set_run_font(run, size, bold=True)
    return para


def bullet(doc, text):
    para = doc.add_paragraph()
    para.style = doc.styles["Normal"]
    para.paragraph_format.left_indent = Cm(1.25)
    para.paragraph_format.first_line_indent = Cm(-0.5)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)
    run = para.add_run("–  " + text)
    set_run_font(run, 13)
    return para


def caption(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.space_before = Pt(8)
    para.paragraph_format.space_after = Pt(4)
    para.paragraph_format.line_spacing = 1.0
    run = para.add_run(text)
    set_run_font(run, 10, bold=True)
    return para


def note(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(8)
    run = para.add_run(text)
    set_run_font(run, 10, italic=True)
    return para


def formula(doc, text, number):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(6)
    run = para.add_run(text + f"    ({number})")
    set_run_font(run, 13, italic=True)
    return para


def _set_tc_border(cell, **sides):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge, val in sides.items():
        el = tcBorders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            tcBorders.append(el)
        if val:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "8")
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for j, header in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(header)
        set_run_font(run, 12, bold=True)
        _set_tc_border(cell, top=True, bottom=True, left=False, right=False)
    for i, row in enumerate(rows):
        last = i == len(rows) - 1
        for j, value in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(str(value))
            set_run_font(run, 12)
            _set_tc_border(cell, top=False, bottom=last, left=False, right=False)
    return table


def figure(doc, path: Path, width=14.2):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(0)
    run = para.add_run()
    run.add_picture(str(path), width=Cm(width))
    return para


def cover_blank(doc, n: int = 1) -> None:
    for _ in range(n):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.first_line_indent = Cm(0)
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.line_spacing = 1.0


def cover_line(doc, text: str, *, size: int = 13, bold: bool = True, after: float = 0) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(after)
    para.paragraph_format.line_spacing = 1.08 if after else 1.0
    run = para.add_run(text)
    set_run_font(run, size, bold=bold)


def cover_logo(doc) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)
    run = para.add_run()
    run.add_picture(str(LOGO), width=Emu(855544), height=Emu(431250))


def toc_entry(doc, text: str, page: str, *, level: int = 1) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.left_indent = Cm(0 if level == 1 else 0.75)
    para.paragraph_format.space_before = Pt(3 if level == 1 else 0)
    para.paragraph_format.space_after = Pt(0)
    para.paragraph_format.line_spacing = 1.0
    para.paragraph_format.tab_stops.add_tab_stop(
        Cm(15.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
    )
    run = para.add_run(f"{text}\t{page}")
    set_run_font(run, 13, bold=(level == 1))


def sign(doc, text: str, *, bold: bool) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)
    para.paragraph_format.tab_stops.add_tab_stop(Cm(12.5), WD_TAB_ALIGNMENT.CENTER)
    run = para.add_run("\t" + text)
    set_run_font(run, 13, bold=bold)


def write_cover(doc, *, inner: bool) -> None:
    cover_line(doc, "ỦY BAN NHÂN DÂN TỈNH BÌNH DƯƠNG", after=8)
    cover_line(doc, "TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT")
    cover_line(doc, "----------o0o----------")
    cover_logo(doc)
    cover_blank(doc, 1)
    cover_line(doc, "NGUYỄN NGỌC THÁI")
    cover_blank(doc, 1)
    for line in TITLE_LINES:
        cover_line(doc, line, size=16)
    cover_blank(doc, 2)
    cover_line(doc, "CHUYÊN NGÀNH: HỆ THỐNG THÔNG TIN")
    cover_line(doc, "MÃ SỐ: 8 48 01 04")
    if inner:
        cover_blank(doc, 1)
        cover_line(doc, "ĐỀ ÁN TỐT NGHIỆP THẠC SĨ")
        cover_blank(doc, 1)
        cover_line(doc, "NGƯỜI HƯỚNG DẪN KHOA HỌC:")
        cover_line(doc, "BÙI THANH KHIẾT")
        cover_blank(doc, 7)
    else:
        cover_blank(doc, 2)
        cover_line(doc, "ĐỀ ÁN TỐT NGHIỆP THẠC SĨ")
        cover_blank(doc, 11)
    cover_line(doc, "BÌNH DƯƠNG – NĂM 2026", after=8)


def _box(ax, x, y, w, h, text, fc):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.8, edgecolor=NAVY, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=NAVY, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 linewidth=0.9, color=NAVY))


def _load_results():
    rows = []
    with RESULTS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def make_figures() -> dict[str, Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": GRAY,
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
    })

    # Luồng chạy
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    _box(ax, 0.25, 3.35, 2.1, 0.85, "Video hoặc RTSP\n(DDNS, cổng ngoài)", "#E7F0F4")
    _box(ax, 3.05, 3.35, 2.2, 0.85, "Giải mã\nrộng tối đa 1 280 px", "#E7F0F4")
    _box(ax, 5.95, 3.35, 2.3, 0.85, "YOLOv12n lớp học\nimgsz 640, conf 0,15", "#D9E8E6")
    _box(ax, 8.7, 3.35, 2.2, 0.85, "ByteTrack\ngiữ quỹ đạo", "#D9E8E6")
    _arrow(ax, 2.35, 3.77, 3.05, 3.77)
    _arrow(ax, 5.25, 3.77, 5.95, 3.77)
    _arrow(ax, 8.25, 3.77, 8.7, 3.77)
    _box(ax, 0.25, 1.85, 2.5, 0.9, "YOLO11n-pose\nimgsz 320, tối đa 40 người", "#F4EDE2")
    _box(ax, 3.2, 1.85, 2.7, 0.9, "Luật hành vi\n(thời gian giữ, chuẩn hóa vai)", "#F4EDE2")
    _box(ax, 6.4, 1.85, 2.3, 0.9, "Vẽ khung\nsố 1…33 + hành vi", "#E7F0F4")
    _box(ax, 9.05, 1.85, 1.9, 0.9, "JPEG\nWebSocket", "#E7F0F4")
    _arrow(ax, 9.8, 3.35, 1.5, 2.75)
    _arrow(ax, 2.75, 2.3, 3.2, 2.3)
    _arrow(ax, 5.9, 2.3, 6.4, 2.3)
    _arrow(ax, 8.7, 2.3, 9.05, 2.3)
    _box(ax, 0.25, 0.3, 3.3, 0.95, "Luồng ArcFace\nSCRFD + buffalo_l, gallery 33 em", "#F6E4E4")
    _box(ax, 4.15, 0.3, 3.1, 0.95, "Khóa danh tính\n≥28%, khóa cứng ≥45%", "#F6E4E4")
    _box(ax, 7.8, 0.3, 3.1, 0.95, "Excel\nđiểm danh và từng đợt hành vi", "#F6E4E4")
    _arrow(ax, 4.15, 3.35, 1.9, 1.25)
    _arrow(ax, 3.55, 0.77, 4.15, 0.77)
    _arrow(ax, 7.25, 0.77, 7.8, 0.77)
    fig.tight_layout(pad=0.2)
    p_flow = FIG / "luong_chay.png"
    fig.savefig(p_flow, dpi=160)
    plt.close(fig)

    # Luật hành vi
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    _box(ax, 3.3, 4.45, 4.6, 0.7, "17 điểm COCO, bỏ điểm conf < 0,5\nEMA: x = 0,6·mới + 0,4·cũ", "#E7F0F4")
    _box(ax, 0.3, 3.15, 3.3, 0.75, "Thiếu vai hoặc đầu\n→ Không xác định", "#F6E4E4")
    _box(ax, 3.95, 3.15, 3.3, 0.75, "Cổ tay + khuỷu trên vai ≥ 1 s\n→ Giơ tay", "#D9E8E6")
    _box(ax, 7.6, 3.15, 3.3, 0.75, "Cúi + tay trên bàn + tay động\n→ Viết bài", "#D9E8E6")
    _arrow(ax, 4.3, 4.45, 1.9, 3.9)
    _arrow(ax, 5.6, 4.45, 5.6, 3.9)
    _arrow(ax, 6.9, 4.45, 9.2, 3.9)
    _box(ax, 0.3, 1.7, 3.3, 0.8, "Lệch hướng nhìn nền > 30°\nvà giữ ≥ 2 s → Quay đầu", "#D9E8E6")
    _box(ax, 3.95, 1.7, 3.3, 0.8, "Đang viết thì không gắn\ncúi đầu lâu hay gục bàn", "#F4EDE2")
    _box(ax, 7.6, 1.7, 3.3, 0.8, "Nghiêng vai > 25° ≥ 2 s\n→ Nghiêng người (chỉ tư thế)", "#F4EDE2")
    _arrow(ax, 5.6, 3.15, 1.9, 2.5)
    _arrow(ax, 5.6, 3.15, 5.6, 2.5)
    _arrow(ax, 5.6, 3.15, 9.2, 2.5)
    _box(ax, 2.1, 0.25, 3.3, 0.85, "Cúi, không viết, ≥ 5 s\n→ Cúi đầu lâu", "#F6E4E4")
    _box(ax, 5.8, 0.25, 3.4, 0.85, "Đầu thấp và thân nghiêng ≥ 5 s\n→ Gục xuống bàn (không kết luận ngủ)", "#F6E4E4")
    _arrow(ax, 5.6, 1.7, 3.7, 1.1)
    _arrow(ax, 5.6, 1.7, 7.5, 1.1)
    fig.tight_layout(pad=0.15)
    p_bh = FIG / "luat_hanh_vi.png"
    fig.savefig(p_bh, dpi=160)
    plt.close(fig)

    rows = _load_results()
    ep = [r["epoch"] for r in rows]

    def _plot(series, ylabel, path, ylim=None):
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        for key, label, color in series:
            ax.plot(ep, [r[key] for r in rows], color=color, linewidth=1.6, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 50)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, color=LINE, linewidth=0.6)
        ax.legend(frameon=False, fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.2f}".replace(".", ",")))
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    p_map = FIG / "yolo_map.png"
    p_pr = FIG / "yolo_pr.png"
    p_loss = FIG / "yolo_loss.png"
    _plot([
        ("metrics/mAP50(B)", "mAP50", TEAL),
        ("metrics/mAP50-95(B)", "mAP50-95", SAND),
    ], "Độ chính xác trung bình", p_map, (0, 1.05))
    _plot([
        ("metrics/precision(B)", "Precision", NAVY),
        ("metrics/recall(B)", "Recall", RED),
    ], "Precision và recall", p_pr, (0, 1.05))
    _plot([
        ("train/box_loss", "Box loss (huấn luyện)", NAVY),
        ("val/box_loss", "Box loss (kiểm định)", TEAL),
        ("train/cls_loss", "Cls loss (huấn luyện)", SAND),
        ("val/cls_loss", "Cls loss (kiểm định)", RED),
    ], "Giá trị mất mát", p_loss)
    return {"flow": p_flow, "behavior": p_bh, "map": p_map, "pr": p_pr, "loss": p_loss}


def build() -> None:
    figs = make_figures()
    doc = Document()
    configure(doc)
    set_margins(doc.sections[0], cover=True)
    set_page_fmt(doc.sections[0], None, None)
    footer = doc.sections[0].footer
    footer.is_linked_to_previous = False
    footer.paragraphs[0].text = ""
    write_cover(doc, inner=False)
    doc.add_page_break()
    write_cover(doc, inner=True)

    sec_front = doc.add_section()
    set_margins(sec_front)
    set_page_fmt(sec_front, "lowerRoman", 1)
    fp = sec_front.footer.paragraphs[0]
    fp.text = ""
    _footer_page(fp)

    H(doc, "MỤC LỤC", 1)
    for text, page, level in (
        ("LỜI CAM ĐOAN", "ii", 1),
        ("LỜI CẢM ƠN", "iii", 1),
        ("DANH MỤC CÁC THUẬT NGỮ, CHỮ VIẾT TẮT", "iv", 1),
        ("DANH MỤC BẢNG", "v", 1),
        ("DANH MỤC HÌNH", "vi", 1),
        ("CHƯƠNG I: MỞ ĐẦU", "1", 1),
        ("1.1. Lý do lựa chọn đề tài", "1", 2),
        ("1.2. Mục tiêu nghiên cứu", "1", 2),
        ("1.3. Đối tượng và phạm vi nghiên cứu", "2", 2),
        ("1.4. Kết quả, sản phẩm và đóng góp", "2", 2),
        ("1.5. Kết cấu của đề án", "3", 2),
        ("CHƯƠNG II: TỔNG QUAN TÀI LIỆU", "4", 1),
        ("2.1. Thị giác máy tính và các phép đo dùng trong đề án", "4", 2),
        ("2.2. Phát hiện người và theo vết", "4", 2),
        ("2.3. Nhận diện khuôn mặt", "5", 2),
        ("2.4. Các nghiên cứu điểm danh liên quan", "5", 2),
        ("2.5. Tiểu kết chương II", "6", 2),
        ("CHƯƠNG III: PHƯƠNG PHÁP NGHIÊN CỨU", "7", 1),
        ("3.1. Kiến trúc và luồng chạy", "7", 2),
        ("3.2. Dữ liệu khuôn mặt và danh sách lớp", "8", 2),
        ("3.3. Gắn danh tính", "8", 2),
        ("3.4. Luật hành vi", "9", 2),
        ("3.5. Cách đánh giá", "11", 2),
        ("3.6. Thành phần phần mềm", "11", 2),
        ("3.7. Tiểu kết chương III", "11", 2),
        ("CHƯƠNG IV: KẾT QUẢ VÀ THẢO LUẬN", "13", 1),
        ("4.1. Huấn luyện bộ phát hiện người", "13", 2),
        ("4.2. Gallery đang nạp", "15", 2),
        ("4.3. Vận hành", "15", 2),
        ("4.4. Thảo luận", "16", 2),
        ("4.5. Tiểu kết chương IV", "16", 2),
        ("CHƯƠNG V: KẾT LUẬN VÀ ĐỀ XUẤT", "17", 1),
        ("5.1. Kết luận", "17", 2),
        ("5.2. Đề xuất", "17", 2),
        ("TÀI LIỆU THAM KHẢO", "19", 1),
        ("PHỤ LỤC", "21", 1),
        ("Phụ lục 1. Lệnh chạy", "21", 2),
        ("Phụ lục 2. Mẫu nhãn hành vi", "21", 2),
    ):
        toc_entry(doc, text, page, level=level)

    doc.add_page_break()
    H(doc, "LỜI CAM ĐOAN", 1)
    P(doc, "Tôi xin cam đoan đề án thạc sĩ hệ thống thông tin “Ứng dụng thị giác máy tính hỗ trợ điểm danh và theo dõi hành vi học sinh tại trường Tiểu học và Trung học Cơ sở” là do tôi nghiên cứu, tổng hợp và thực hiện.")
    P(doc, "Các số liệu được trích từ mã nguồn, tệp cấu hình và kết quả huấn luyện đang có trong dự án. Những kết quả chưa đo lại trên video đã gán nhãn được ghi rõ là ngưỡng khởi đầu, không được dùng để kết luận về từng học sinh.")
    P(doc, "Các tài liệu tham khảo được trích từ bốn bài đã dùng làm nguồn: bài điểm danh YOLOv5 và ArcFace tại MIET 2022, ByteTrack, SCRFD và MediaPipe, cùng các công trình được các bài đó trích dẫn.")
    P(doc, "Ngoài những tài liệu được trích dẫn, nội dung này chưa được nộp để nhận bằng cấp tại cơ sở đào tạo khác.")

    doc.add_page_break()
    H(doc, "LỜI CẢM ƠN", 1)
    P(doc, "Tôi xin cảm ơn TS. Bùi Thanh Khiết đã hướng dẫn khoa học trong quá trình thực hiện đề án. Tôi cảm ơn Trường Đại học Thủ Dầu Một đã tạo điều kiện về chương trình đào tạo.")
    P(doc, "Tôi cảm ơn nhà trường và phụ huynh đã tạo điều kiện về dữ liệu hình ảnh phục vụ thử nghiệm, với yêu cầu ảnh khuôn mặt học sinh chỉ được dùng trong phạm vi đề án và được bảo vệ theo quy định về dữ liệu cá nhân.")
    P(doc, "Tôi xin chân thành cảm ơn.")
    sign(doc, "Người thực hiện", bold=False)
    sign(doc, "Nguyễn Ngọc Thái", bold=True)

    doc.add_page_break()
    H(doc, "DANH MỤC CÁC THUẬT NGỮ, CHỮ VIẾT TẮT", 1)
    caption(doc, "Bảng 0.1: Các chữ viết tắt dùng trong đề án.")
    add_table(doc, ["Chữ viết tắt", "Diễn giải", "Nghĩa dùng trong đề án"], [
        ["YOLO", "You Only Look Once", "Họ mô hình phát hiện vật thể một lần truyền xuôi"],
        ["ArcFace", "Additive Angular Margin Loss", "Hàm mất mát và mạng đặc trưng khuôn mặt"],
        ["SCRFD", "Sample and Computation Redistribution for Face Detection", "Bộ phát hiện mặt trong buffalo_l"],
        ["IoU", "Intersection over Union", "Độ chồng của hai hộp bao"],
        ["mAP", "mean Average Precision", "Độ chính xác trung bình của bộ phát hiện"],
        ["EMA", "Exponential Moving Average", "Làm mượt tọa độ điểm khớp"],
        ["RTSP", "Real Time Streaming Protocol", "Giao thức luồng hình camera IP"],
        ["DDNS", "Dynamic Domain Name System", "Tên miền trỏ tới IP public thay đổi"],
        ["COCO", "Common Objects in Context", "Bộ 17 điểm khớp thân người"],
        ["CPU", "Central Processing Unit", "Máy đích không dùng GPU"],
    ])
    note(doc, "(Nguồn: quy ước của đề án).")

    doc.add_page_break()
    H(doc, "DANH MỤC BẢNG", 1)
    for text in (
        "Bảng 2.1. So sánh ba hướng nhận diện khuôn mặt được xem xét.",
        "Bảng 2.2. So sánh đề án với các hệ thống điểm danh được khảo sát.",
        "Bảng 3.1. Bộ dữ liệu khuôn mặt đang dùng khi chạy.",
        "Bảng 3.2. Tham số nhận diện và gắn danh tính.",
        "Bảng 3.3. Hành vi đang nhận biết, điều kiện và nhóm quan sát.",
        "Bảng 3.4. Thành phần phần mềm chính.",
        "Bảng 4.1. Kết quả YOLOv12n trên tập kiểm định sau 50 epoch.",
        "Bảng 4.2. Thông số gallery ArcFace đang nạp khi chạy.",
    ):
        P(doc, text, indent=False)

    doc.add_page_break()
    H(doc, "DANH MỤC HÌNH", 1)
    for text in (
        "Hình 3.1. Luồng chạy của một phiên phân tích.",
        "Hình 3.2. Luồng quyết định hành vi sau điểm khớp.",
        "Hình 4.1. mAP50 và mAP50-95 của YOLOv12n theo epoch.",
        "Hình 4.2. Precision và recall của YOLOv12n theo epoch.",
        "Hình 4.3. Box loss và classification loss trên tập huấn luyện và tập kiểm định.",
    ):
        P(doc, text, indent=False)

    sec_body = doc.add_section()
    set_margins(sec_body)
    set_page_fmt(sec_body, "decimal", 1)
    bp = sec_body.footer.paragraphs[0]
    bp.text = ""
    _footer_page(bp)

    H(doc, "CHƯƠNG I: MỞ ĐẦU", 1)
    H(doc, "1.1. Lý do lựa chọn đề tài", 2)
    P(doc, "Điểm danh cả một lớp bằng cách gọi tên dễ sót và tốn thời gian đầu tiết [1]. Thị giác máy tính trên video lớp cần trả lời ba câu: có người hay không, người đó là ai, và tư thế đang thể hiện hành vi nào [2], [4].")
    P(doc, "Camera lớp tạo ra video liên tục, nhưng video tự nó chưa cho biết học sinh nào có mặt và đang giơ tay, viết bài, quay đầu, cúi đầu lâu hay gục xuống bàn. Muốn dùng được video, hệ thống phải phát hiện từng người, giữ cùng một người qua nhiều khung, gắn đúng mã đã đăng ký, rồi đọc tư thế trong một khoảng thời gian đủ dài.")
    P(doc, "Đề án xây dựng đường ống đó trên máy tính chỉ có CPU, hiện kết quả ngay trên giao diện web và lưu biên bản sau buổi học. Hệ thống ghi những gì nhìn thấy trên khung hình. Hệ thống không quy điểm tập trung từ 0 đến 100 và không kết luận học sinh đang ngủ hay đang thẫn thờ.")

    H(doc, "1.2. Mục tiêu nghiên cứu", 2)
    P(doc, "Mục tiêu tổng quát là xây dựng hệ thống thị giác máy tính dùng video lớp học để điểm danh và ghi nhận hành vi quan sát được của từng học sinh, hiển thị trên giao diện web và xuất báo cáo Excel.")
    P(doc, "Mục tiêu cụ thể:")
    bullet(doc, "Huấn luyện bộ phát hiện người YOLOv12 trên ảnh lớp học, với mốc mAP50 từ 0,90 trở lên.")
    bullet(doc, "Nhận diện học sinh bằng ArcFace trên gallery của lớp, giữ danh tính ổn định theo quỹ đạo, ngưỡng chỉnh được trên giao diện.")
    bullet(doc, "Nhận biết giơ tay, viết bài, quay đầu, cúi đầu lâu và gục xuống bàn từ 17 điểm khớp; nghiêng người chỉ được ghi là tư thế; thiếu điểm khớp thì ghi không xác định.")
    bullet(doc, "Xuất báo cáo điểm danh và từng đợt hành vi theo mã học sinh và họ tên trong danh sách lớp.")
    bullet(doc, "Chuẩn bị quy trình đo lại ngưỡng hành vi khi đã có video được giáo viên gán nhãn.")

    H(doc, "1.3. Đối tượng và phạm vi nghiên cứu", 2)
    P(doc, "Đối tượng nghiên cứu là các phương pháp phát hiện, theo vết, nhận diện khuôn mặt và nhận biết hành vi từ video camera lớp học, áp dụng cho một lớp tiểu học hoặc trung học cơ sở.")
    bullet(doc, "Không gian: một lớp, camera cố định phía trước. Nguồn hình là tệp video hoặc luồng RTSP, kể cả camera mở qua DDNS và cổng chuyển tiếp. Ứng dụng không mở webcam của máy tính.")
    bullet(doc, "Dữ liệu nhận diện đang chạy: 33 học sinh, mã HS7_01 đến HS7_33, mỗi em 18 ảnh. Họ tên xuất Excel lấy từ danh_sach_hoc_sinh.csv. Tên trong tệp thử nghiệm là tên giả lập, chưa phải sổ điểm danh của nhà trường.")
    bullet(doc, "Hành vi hiện trên giao diện: giơ tay, viết bài, quay đầu, cúi đầu lâu, gục xuống bàn, nghiêng người và không xác định. Nhắm mắt, nói chuyện, tay bất thường và cúi đầu thoáng qua đang tắt.")
    bullet(doc, "Phần cứng mục tiêu: máy tính cá nhân chỉ có CPU. Phiên bản đang chạy được kiểm tra trên macOS với Python 3.9.")
    bullet(doc, "Chưa thuộc phạm vi: cơ sở dữ liệu nhiều lớp nhiều buổi, và kết nối phần mềm quản lý nhà trường.")

    H(doc, "1.4. Kết quả, sản phẩm và đóng góp", 2)
    P(doc, "Sản phẩm là ứng dụng web. Giáo viên tải video hoặc kết nối camera, theo dõi khung hình đã gắn số thứ tự và hành vi, xem sĩ số cùng các nhóm quan sát, rồi xuất Excel. Bộ công cụ đi kèm dùng để dựng gallery, trích điểm khớp và đo lại luật hành vi khi có nhãn.")
    P(doc, "Đóng góp về phương pháp là đường ống phát hiện người, nhận diện khuôn mặt và ước lượng tư thế chạy song song; luật hành vi chuẩn hóa theo bề rộng vai và theo tư thế nền của từng học sinh; danh tính trên video được đánh số từ 1 theo danh sách lớp trong khi báo cáo vẫn giữ mã gốc.")
    P(doc, "Kết quả định lượng đã có là đường cong huấn luyện YOLOv12n và độ tách gallery ArcFace của đúng bộ dữ liệu đang nạp. Chưa có precision và recall của từng hành vi trên video đã gán nhãn, nên các mốc thời gian trong luật được trình bày là ngưỡng khởi đầu.")

    H(doc, "1.5. Kết cấu của đề án", 2)
    bullet(doc, "Chương I trình bày lý do, mục tiêu, phạm vi và sản phẩm.")
    bullet(doc, "Chương II trình bày cơ sở mô hình, bảng so sánh hướng nhận diện và các nghiên cứu điểm danh liên quan.")
    bullet(doc, "Chương III trình bày luồng chạy, dữ liệu, cách gắn danh tính và luật hành vi.")
    bullet(doc, "Chương IV trình bày biểu đồ huấn luyện, gallery đang dùng và cách hệ thống vận hành.")
    bullet(doc, "Chương V tóm tắt kết quả và các việc cần làm trước khi dùng số liệu để nhận xét học sinh.")

    doc.add_page_break()
    H(doc, "CHƯƠNG II: TỔNG QUAN TÀI LIỆU", 1)
    H(doc, "2.1. Thị giác máy tính và các phép đo dùng trong đề án", 2)
    P(doc, "Học sâu dùng mạng nhiều tầng để học đặc trưng từ điểm ảnh. Mạng tích chập giữ được vị trí tương đối của mắt, mũi và vai, phù hợp với ảnh lớp học. Đề án dùng ba mạng loại này: YOLOv12 để tìm người [2], ArcFace để tạo vectơ khuôn mặt [4], và YOLO11-pose để lấy điểm khớp theo 17 điểm COCO [6].")
    P(doc, "Độ chồng IoU của hai hộp bao A và B được dùng khi huấn luyện bộ phát hiện và khi ghép hộp vào quỹ đạo:")
    formula(doc, "IoU(A, B) = |A ∩ B| / |A ∪ B|", "2.1")
    P(doc, "Trong đó A và B là hai hộp chữ nhật trên cùng một khung hình; tử số là diện tích phần giao, mẫu số là diện tích phần hợp.")
    P(doc, "Độ tương đồng cosine giữa vectơ khuôn mặt q cắt từ video và vectơ gallery g được dùng để gắn tên:")
    formula(doc, "s(q, g) = (q · g) / (||q|| ||g||)", "2.2")
    P(doc, "Vectơ q và g có 512 chiều và đã được chuẩn hóa. Phần mềm nhân s(q, g) với 100 để hiện thành phần trăm khớp.")

    H(doc, "2.2. Phát hiện người và theo vết", 2)
    P(doc, "Họ YOLO dự đoán hộp bao và nhãn trong một lần truyền xuôi, phù hợp với video [2]. Các bước sau của họ này, gồm YOLOv3 và YOLOv4, tiếp tục cân bằng tốc độ và độ chính xác [11], [10]. Bài điểm danh cả phòng của Faruque dùng YOLOv5 để tìm mặt [1]. Đề án không tìm mặt bằng YOLO. Đề án dùng YOLOv12n, khởi tạo từ yolo12n.pt và huấn luyện thêm 50 epoch, batch 16, kích thước ảnh 640, trên ảnh người trong lớp học.")
    P(doc, "Khi suy luận, ngưỡng tin cậy là 0,15 và ngưỡng IoU là 0,45. ByteTrack nối hộp qua các khung, kể cả hộp có độ tin cậy thấp, để giữ quỹ đạo khi học sinh bị che một đoạn ngắn [3]. Một quỹ đạo là một người trong buổi học. Danh tính khuôn mặt được gắn vào quỹ đạo đó, không gắn lại từ đầu ở từng khung.")

    H(doc, "2.3. Nhận diện khuôn mặt", 2)
    P(doc, "ArcFace học vectơ trên mặt siêu cầu và cộng thêm một biên góc vào lớp đúng, để vectơ của những người khác nhau tách xa nhau hơn [4]. Hệ thống dùng bộ buffalo_l của InsightFace. Bộ phát hiện mặt là SCRFD, được thiết kế bằng cách phân bố lại mẫu huấn luyện và khối lượng tính toán giữa thân, cổ và đầu mạng [5]. Mạng nhận diện là ResNet50, xuất vectơ 512 chiều. Một mạng điểm mốc cho góc quay đầu khi mặt đủ lớn.")
    P(doc, "Ba hướng được xem xét là DeepFace [8], FaceNet [7] và ArcFace [4]. Bảng 2.1 so sánh cách học đặc trưng. So sánh này dựa trên mô tả trong các bài nguồn, không dựa trên một bộ kiểm tra chung chạy lại trong đề án.")
    caption(doc, "Bảng 2.1: So sánh ba hướng nhận diện khuôn mặt được xem xét.")
    add_table(doc, ["Mô hình", "Cách học đặc trưng", "Điểm phù hợp với lớp đông"], [
        ["DeepFace", "Căn mặt 3D rồi phân lớp softmax", "Vectơ dài, chậm trên CPU khi sang tập mới"],
        ["FaceNet", "Vectơ 128 chiều, hàm mất mát bộ ba", "Nhẹ, các vectơ cùng trang phục dễ nằm gần nhau"],
        ["ArcFace", "Vectơ 512 chiều, biên góc cộng vào lớp đúng", "Có sẵn bộ phát hiện và điểm mốc trong buffalo_l"],
    ])
    note(doc, "(Nguồn: tổng hợp từ [8], [7], [4], [5]).")

    H(doc, "2.4. Các nghiên cứu điểm danh liên quan", 2)
    P(doc, "Điểm danh bằng khuôn mặt thường gồm tìm mặt, đưa về khung chuẩn và so vectơ với ảnh đã lưu. Faruque và cộng sự điểm danh cả phòng bằng YOLOv5 để tìm mặt và ArcFace để nhận người trên ảnh lớp thật [1]. Sutabri và cộng sự dùng học sâu cho điểm danh sinh viên [12]. Alon và cộng sự dùng YOLOv3 để nhận diện khuôn mặt khi điểm danh [13]. MediaPipe cung cấp khung ghép các bộ nhận thức thành một pipeline, gồm cả điểm mốc khuôn mặt [9].")
    P(doc, "Bảng 2.2 đặt đề án cạnh ba công trình điểm danh đó. Khác biệt chính không nằm ở việc có nhận ra mặt hay không, mà ở chỗ đề án giữ quỹ đạo người, đọc tư thế thân và ghi thời lượng từng hành vi thay vì chỉ đóng một dòng có mặt.")
    caption(doc, "Bảng 2.2: So sánh đề án với các hệ thống điểm danh được khảo sát.")
    add_table(doc, ["Hệ thống", "Phát hiện", "Nhận diện", "Theo vết", "Hành vi trên lớp"], [
        ["Faruque và cộng sự [1]", "YOLOv5, tìm mặt", "ArcFace", "Theo khung hình", "Không ghi"],
        ["Sutabri và cộng sự [12]", "Học sâu", "Nhận diện mặt", "Không mô tả quỹ đạo", "Không ghi"],
        ["Alon và cộng sự [13]", "YOLOv3", "Nhận diện mặt", "Không mô tả quỹ đạo", "Không ghi"],
        ["Đề án này", "YOLOv12n, tìm người", "ArcFace buffalo_l", "ByteTrack", "Giơ tay, viết, quay đầu, cúi lâu, gục bàn"],
    ])
    note(doc, "(Nguồn: [1], [12], [13] và cấu hình đang chạy của đề án).")
    P(doc, "Các công trình điểm danh trên dừng ở danh tính. Đề án tách hai việc: danh tính phục vụ điểm danh; hành vi được liệt kê theo tên và thời lượng để giáo viên xem, không gộp thành một điểm.")

    H(doc, "2.5. Tiểu kết chương II", 2)
    P(doc, "Chương này chọn YOLOv12 cho người, ByteTrack cho quỹ đạo, ArcFace và SCRFD cho danh tính, YOLO11-pose cho 17 điểm khớp. Ba hướng nhận diện được so về cách học vectơ. Ba hệ thống điểm danh được so về phạm vi đầu ra. Chương sau mô tả cách các khối này chạy cùng nhau trong một phiên học.")

    doc.add_page_break()
    H(doc, "CHƯƠNG III: PHƯƠNG PHÁP NGHIÊN CỨU", 1)
    H(doc, "3.1. Kiến trúc và luồng chạy", 2)
    P(doc, "Hệ thống là ứng dụng web trên máy cục bộ, địa chỉ 127.0.0.1 cổng 5000. Một phiên phân tích có luồng đọc khung, luồng phát hiện và tư thế, luồng nhận diện khuôn mặt, và luồng đọc giờ in sẵn trên hình camera. Hình 3.1 thể hiện thứ tự các khối. Nhánh tư thế và nhánh khuôn mặt dùng chung khung hình nhưng không chờ nhau: video vẫn được vẽ khi ArcFace chưa trả tên.")
    figure(doc, figs["flow"], 14.6)
    caption(doc, "Hình 3.1: Luồng chạy của một phiên phân tích.")
    note(doc, "(Nguồn: app.py và core/analysis.py của phiên bản đang chạy).")
    bullet(doc, "Luồng đọc giải mã đúng tốc độ video, thu khung về bề rộng 1 280 pixel và chỉ giữ khung mới nhất.")
    bullet(doc, "Luồng phân tích chạy YOLOv12, ByteTrack và YOLO11-pose, rồi áp luật hành vi. Bước thời gian của bộ đếm là khoảng cách giữa hai lần phân tích.")
    bullet(doc, "Luồng ArcFace nhận diện trên khung và các quỹ đạo của lần phân tích gần nhất.")
    bullet(doc, "Luồng đọc giờ dùng EasyOCR trên dòng ngày giờ camera in sẵn, đọc lại sau mỗi 60 giây video.")
    P(doc, "Khung hiển thị được vẽ đè kết quả phân tích gần nhất và gửi về trình duyệt dạng JPEG. Bảng sĩ số được tính lại mỗi giây. Màu khung theo nhóm quan sát: đỏ là tư thế cần xem, xanh dương là hoạt động quan sát được, vàng là nghiêng người, xám là có tư thế mà không gắn cảnh báo, xanh nhạt là chưa đủ tư thế.")

    H(doc, "3.2. Dữ liệu khuôn mặt và danh sách lớp", 2)
    P(doc, "Khi chạy, phần mềm nạp gallery tại 3.Thesis/models/arcface_classroom-faces-lop07/gallery.npz, đồng bộ từ 2.Data/classroom-faces-lop07 sau khi thư mục ảnh được thay. Mã học sinh giữ thứ tự HS7_01 đến HS7_33. Số hiện trên khung là thứ tự trong danh sách: HS7_01 thành 1, HS7_33 thành 33. Excel vẫn ghi mã gốc.")
    caption(doc, "Bảng 3.1: Bộ dữ liệu khuôn mặt đang dùng khi chạy hệ thống.")
    add_table(doc, ["Hạng mục", "Giá trị đang dùng"], [
        ["Số học sinh", "33"],
        ["Mã", "HS7_01 đến HS7_33"],
        ["Ảnh mỗi học sinh", "18 ảnh, gồm enroll và test"],
        ["Vectơ", "512 chiều, một vectơ đại diện mỗi em"],
        ["Tệp họ tên", "danh_sach_hoc_sinh.csv, cột full_name"],
        ["Số trên khung hình", "1 với HS7_01, 33 với HS7_33"],
        ["Mã ghi trong Excel", "Giữ nguyên HS7_01 … HS7_33"],
        ["Họ tên hiện tại", "Tên giả lập để thử, chưa phải sổ nhà trường"],
    ])
    note(doc, "(Nguồn: students.csv, danh_sach_hoc_sinh.csv và meta.json của gallery).")

    H(doc, "3.3. Gắn danh tính", 2)
    P(doc, "Một khuôn mặt được đưa qua SCRFD, căn theo điểm mốc, rồi so cosine với 33 vectơ gallery. Tên chỉ được gắn khi độ khớp đạt ngưỡng và hơn người đứng thứ hai một biên đủ lớn. Khóa cứng dùng khi độ khớp cao và cách người thứ hai đủ xa, để quỹ đạo không bị đổi tên ở các khung sau.")
    caption(doc, "Bảng 3.2: Tham số nhận diện và gắn danh tính.")
    add_table(doc, ["Tham số", "Giá trị", "Ý nghĩa"], [
        ["Ngưỡng khớp", "0,28 (28%)", "Dưới mức này không gắn tên"],
        ["Biên tối thiểu", "0,012", "Phải hơn người thứ hai ít nhất mức này"],
        ["Khóa mềm", "0,38", "Bắt đầu giữ tên trên quỹ đạo"],
        ["Khóa cứng", "≥ 45% và hơn người sau ≥ 10 điểm", "Không đổi tên ở khung sau"],
        ["Có mặt", "≥ 3 khung nhận ra", "Mới ghi có mặt trong Excel"],
        ["Tin cậy cao", "Khớp tối đa ≥ 45% và ≥ 10 phiếu", "Cột độ tin cậy trong Excel"],
        ["Tin cậy trung bình", "Khớp tối đa ≥ 35%", "Mức còn lại khi đã đủ 3 khung"],
        ["Quỹ đạo chưa nhận ra", "ID kèm số theo dõi", "Không bị gán nhầm vào một mã HS7"],
    ])
    note(doc, "(Nguồn: meta.json và core/face_gallery.py).")

    H(doc, "3.4. Luật hành vi", 2)
    P(doc, "YOLOv12 chưa có trọng số tư thế chính thức, nên điểm khớp lấy từ YOLO11n-pose, đầu vào 320 pixel, tối đa 40 người. Điểm có độ tin cậy dưới 0,5 bị bỏ. Tọa độ còn lại được làm mượt bằng trung bình động hàm mũ, hệ số 0,6 cho giá trị mới. Khoảng cách được chia cho bề rộng vai để cùng một ngưỡng dùng được với em ngồi gần và em ngồi xa.")
    P(doc, "Tư thế nền của từng học sinh chỉ lấy từ những khung ổn định, không cúi và không quay, sau ít nhất 20 mẫu, trong cửa sổ 300 mẫu, lệch hướng nhìn không quá 15° và tốc độ không quá 0,2 bề rộng vai mỗi giây. Mất tín hiệu dưới 1 giây không xóa đợt đang đếm và cũng không cộng thêm thời gian cho nhãn. Hình 3.2 là thứ tự quyết định. Bảng 3.3 là điều kiện đang dùng.")
    figure(doc, figs["behavior"], 14.6)
    caption(doc, "Hình 3.2: Luồng quyết định hành vi sau điểm khớp.")
    note(doc, "(Nguồn: core/behavior.py và BehaviorConfig).")
    caption(doc, "Bảng 3.3: Hành vi đang nhận biết, điều kiện và nhóm quan sát.")
    add_table(doc, ["Hành vi", "Điều kiện khởi đầu", "Nhóm"], [
        ["Giơ tay", "Cổ tay và khuỷu cùng phía cao hơn vai, giữ ≥ 1 s", "Hoạt động"],
        ["Viết bài", "Đầu cúi, cổ tay trong vùng bàn, tay động ≥ 0,15 bề rộng vai/giây trong 1 s", "Hoạt động"],
        ["Quay đầu", "Lệch hướng nhìn nền > 30°, giữ ≥ 2 s", "Hoạt động"],
        ["Cúi đầu lâu", "Đầu thấp hơn nền, không đang viết, giữ ≥ 5 s", "Cần xem"],
        ["Gục xuống bàn", "Đầu thấp và thân nghiêng, không đang viết, giữ ≥ 5 s", "Cần xem"],
        ["Nghiêng người", "Nghiêng vai > 25°, giữ ≥ 2 s", "Chỉ tư thế"],
        ["Không xác định", "Thiếu vai hoặc đầu", "Luôn bật"],
    ])
    note(doc, "(Nguồn: ngưỡng khởi đầu trong core/config.py, chưa đo trên video đã gán nhãn).")
    P(doc, "Viết bài chặn hai nhãn cúi đầu lâu và gục xuống bàn, vì em cúi xuống để viết không được tính là tư thế cần xem. Gục xuống bàn đòi hỏi đồng thời vị trí đầu và độ nghiêng thân. Nhãn này không có nghĩa học sinh đang ngủ. Nhãn thẫn thờ không được dùng.")

    H(doc, "3.5. Cách đánh giá", 2)
    P(doc, "Bộ phát hiện được đánh giá bằng precision, recall, mAP50 và mAP50-95 trên tập kiểm định của chính lần huấn luyện, ghi trong results.csv. Gallery được mô tả bằng độ tương đồng cosine giữa hai học sinh khác nhau: trung bình và lớn nhất. Hai chỉ số này cho biết các vectơ trong lớp có tách nhau hay không, chưa phải tỷ lệ điểm danh đúng trên video.")
    P(doc, "Hành vi sẽ được đo khi có tệp nhãn CSV với các cột student_id, behavior, start, end và note. Công cụ tools/eval_behavior.py đọc nhật ký điểm khớp và đối chiếu với nhãn đó. Trước khi có nhãn, đề án không báo precision hay recall của giơ tay, viết bài, quay đầu, cúi đầu lâu và gục xuống bàn.")

    H(doc, "3.6. Thành phần phần mềm", 2)
    caption(doc, "Bảng 3.4: Thành phần phần mềm chính.")
    add_table(doc, ["Thành phần", "Vai trò"], [
        ["3.Thesis/app.py", "Máy chủ, phiên phân tích, WebSocket, xuất báo cáo"],
        ["core/analysis.py", "Ghép phát hiện, tư thế, danh tính và luật hành vi"],
        ["core/behavior.py", "Hậu xử lý điểm khớp thành nhãn có thời gian giữ"],
        ["core/face_gallery.py", "Nạp gallery, danh sách lớp và số thứ tự 1…33"],
        ["core/behavior_log.py", "Excel điểm danh và từng đợt hành vi"],
        ["tools/eval_arcface.py", "Đo gallery khi cần hạng 1, FAR và FRR"],
        ["tools/eval_behavior.py", "Đo luật hành vi khi đã có nhãn"],
        ["2.Data/cameras.json", "Camera đã lưu; mật khẩu bị che trên giao diện"],
    ])
    note(doc, "(Nguồn: cây mã nguồn phiên bản đang chạy).")

    H(doc, "3.7. Tiểu kết chương III", 2)
    P(doc, "Phiên phân tích tách luồng hình, luồng tư thế và luồng khuôn mặt. Danh tính khóa theo quỹ đạo với ngưỡng lấy từ meta.json. Hành vi là hậu xử lý có thời gian giữ, viết bài được ưu tiên hơn cúi đầu và gục bàn. Chương sau trình bày những gì đã đo được với cấu hình này.")

    doc.add_page_break()
    H(doc, "CHƯƠNG IV: KẾT QUẢ VÀ THẢO LUẬN", 1)
    H(doc, "4.1. Huấn luyện bộ phát hiện người", 2)
    P(doc, "YOLOv12n được huấn luyện 50 epoch, batch 16, kích thước ảnh 640, khởi tạo từ yolo12n.pt, trên máy có GPU của môi trường Kaggle. Bộ trọng số best.pt là bản đang nạp khi chạy ứng dụng trên CPU. Hình 4.1, Hình 4.2 và Hình 4.3 lấy nguyên chuỗi 50 epoch trong results.csv, không làm tròn trung gian.")
    figure(doc, figs["map"], 13.2)
    caption(doc, "Hình 4.1: mAP50 và mAP50-95 của YOLOv12n theo epoch.")
    note(doc, "(Nguồn: 3.Thesis/models/yolo12_classroom/results.csv).")
    figure(doc, figs["pr"], 13.2)
    caption(doc, "Hình 4.2: Precision và recall của YOLOv12n theo epoch.")
    note(doc, "(Nguồn: 3.Thesis/models/yolo12_classroom/results.csv).")
    figure(doc, figs["loss"], 13.2)
    caption(doc, "Hình 4.3: Box loss và classification loss trên tập huấn luyện và tập kiểm định.")
    note(doc, "(Nguồn: 3.Thesis/models/yolo12_classroom/results.csv).")
    P(doc, "Ba epoch đầu còn thấp: mAP50 ở epoch 3 chỉ là 0,013. Từ khoảng epoch 10 đường cong đi lên và ổn định ở vùng trên 0,98 đối với mAP50. Box loss và classification loss của tập kiểm định giảm cùng tập huấn luyện, không thấy tập kiểm định tăng trở lại ở các epoch cuối. Bảng 4.1 là đúng dòng epoch 50.")
    caption(doc, "Bảng 4.1: Kết quả YOLOv12n trên tập kiểm định sau 50 epoch.")
    add_table(doc, ["Chỉ số", "Epoch 50"], [
        ["Precision", "0,957"],
        ["Recall", "0,962"],
        ["mAP50", "0,988"],
        ["mAP50-95", "0,711"],
        ["Box loss kiểm định", "0,941"],
        ["Classification loss kiểm định", "0,603"],
    ])
    note(doc, "(Nguồn: dòng epoch 50 của results.csv).")
    P(doc, "mAP50 vượt mốc 0,90 đã đặt ở mục tiêu. mAP50-95 bằng 0,711 cho thấy hộp bao chưa thật khít ở ngưỡng IoU cao. Chỉ số này đo trên tập kiểm định của lần huấn luyện, chưa đo trên video của một tiết học mới.")

    H(doc, "4.2. Gallery đang nạp", 2)
    P(doc, "Gallery có 33 vectơ 512 chiều, một vectơ cho mỗi mã HS7. Độ tương đồng trung bình giữa hai học sinh khác nhau là 0,168. Cặp lớn nhất là 0,450, đúng bằng ngưỡng khóa cứng. Trên ảnh đăng ký, hai học sinh này nằm sát nhau. Trên video, mặt nhỏ hơn ảnh đăng ký nên cần đo lại hạng 1 và FAR bằng tools/eval_arcface.py trước khi coi ngưỡng 28% là cấu hình cuối.")
    caption(doc, "Bảng 4.2: Thông số gallery ArcFace đang nạp khi chạy.")
    add_table(doc, ["Hạng mục", "Giá trị"], [
        ["Số học sinh", "33"],
        ["Chiều vectơ", "512"],
        ["Ảnh mỗi em", "18"],
        ["Độ tương đồng khác người, trung bình", "0,168"],
        ["Độ tương đồng khác người, lớn nhất", "0,450"],
        ["Cặp gần nhất", "Cần xem lại trước khi chốt ngưỡng 28%"],
        ["Ngưỡng khớp / khóa mềm", "0,28 / 0,38"],
    ])
    note(doc, "(Nguồn: meta.json của arcface_classroom-faces-lop07).")
    P(doc, "Báo cáo này không lấy lại các bảng hạng 1 của lần dữ liệu trước, vì thư mục ảnh đã đổi. Tên trong danh_sach_hoc_sinh.csv đang là tên giả lập để thử đủ 33 dòng. Khi thay bằng họ tên thật, mã HS7_01 đến HS7_33 phải giữ nguyên để khớp gallery.")

    H(doc, "4.3. Vận hành", 2)
    P(doc, "Ứng dụng khởi động bằng Python của môi trường .venv, nạp YOLOv12 lớp học, YOLO11n-pose và gallery 33 học sinh, rồi mở http://127.0.0.1:5000. Giáo viên có thể tải video, dán đường dẫn RTSP, hoặc nhập tên miền DDNS cùng cổng ngoài. Danh sách camera lưu ở 2.Data/cameras.json, mật khẩu bị che trên giao diện.")
    P(doc, "Khi một học sinh được nhận ra, khung hình hiện số từ 1 đến 33 và hành vi chính nếu có. Thanh bên hiện sĩ số, danh sách có mặt xếp tư thế cần xem lên trước, và danh sách vắng. Xuất Excel tạo biên bản điểm danh và từng đợt hành vi theo mã HS7_xx cùng họ tên trong danh sách. Phiên bản hiện tại không có sheet điểm tập trung.")
    P(doc, "Nhật ký điểm khớp của mỗi phiên được ghi tại logs/keypoints để sau này đo luật hành vi mà không phải chạy lại YOLO. Chưa có tệp nhãn đầy đủ nên chưa có biểu đồ precision, recall của từng hành vi.")

    H(doc, "4.4. Thảo luận", 2)
    P(doc, "Phần đã chạy được là đường ống phát hiện, theo vết, nhận diện, luật hành vi, giao diện và báo cáo. Bộ phát hiện đạt mAP50 0,988 trên tập kiểm định của lần huấn luyện. Gallery mới khớp mã với thư mục ảnh và với danh sách họ tên.")
    P(doc, "Phần chưa đủ để nhận xét học sinh là ngưỡng hành vi và ngưỡng nhận diện trên video thật. Cặp gần nhất trong gallery đạt 0,450. Luật cúi đầu, gục xuống bàn và quay đầu dùng các mốc 5 giây, 2 giây và 30° như điểm khởi đầu. Các mốc này cần được đối chiếu với nhãn của giáo viên trên cùng một đoạn video.")
    P(doc, "Việc không dùng thang điểm 0–100 khiến báo cáo bám vào những gì mô hình nhìn thấy. Giáo viên đọc thời lượng từng hành vi và tự ghi chú. Biểu đồ trong chương này là đường cong huấn luyện của bộ phát hiện, không phải biểu đồ mức tập trung của học sinh.")

    H(doc, "4.5. Tiểu kết chương IV", 2)
    P(doc, "Huấn luyện hội tụ sau 50 epoch với mAP50 0,988 và mAP50-95 0,711. Gallery 33 em có độ tách trung bình 0,168 và một cặp sát ngưỡng khóa cứng. Giao diện và Excel đã chạy theo luồng ở Hình 3.1. Đo hành vi trên video vẫn là việc của bước tiếp theo.")

    doc.add_page_break()
    H(doc, "CHƯƠNG V: KẾT LUẬN VÀ ĐỀ XUẤT", 1)
    H(doc, "5.1. Kết luận", 2)
    P(doc, "Đề án đã có hệ thống theo dõi một lớp học gồm YOLOv12 và ByteTrack để phát hiện, theo vết người, ArcFace để nhận diện 33 học sinh mã HS7_01 đến HS7_33, và YOLO11-pose để đọc tư thế. Hành vi đang ghi là giơ tay, viết bài, quay đầu, cúi đầu lâu và gục xuống bàn. Nghiêng người chỉ được ghi là tư thế. Thiếu điểm khớp thì ghi không xác định.")
    P(doc, "Bộ phát hiện đạt mAP50 0,988 và mAP50-95 0,711 sau 50 epoch. Gallery đang chạy có độ tương đồng khác người trung bình 0,168 và lớn nhất 0,450. Ứng dụng web xuất Excel điểm danh và hành vi, lấy họ tên theo mã trong danh sách lớp, và hiện số thứ tự từ 1 đến 33 trên khung hình.")
    P(doc, "Hệ thống không chấm điểm tập trung. Ngưỡng hành vi và ngưỡng nhận diện trên video chưa được đo bằng nhãn, nên kết quả hiện tại là công cụ quan sát cho giáo viên, chưa phải căn cứ đánh giá học sinh.")

    H(doc, "5.2. Đề xuất", 2)
    bullet(doc, "Thay cột full_name trong danh_sach_hoc_sinh.csv bằng họ tên thật, giữ nguyên mã HS7_01 đến HS7_33.")
    bullet(doc, "Gán nhãn 2 đến 3 đoạn video, mỗi đoạn vài phút, theo mẫu ở phụ lục, rồi chạy tools/eval_behavior.py để đo precision, recall trước khi giữ hoặc đổi các mốc thời gian.")
    bullet(doc, "Chạy tools/eval_arcface.py trên bộ classroom-faces-lop07 hiện tại để có hạng 1, FAR và FRR của đúng gallery đang nạp, rồi xem lại ngưỡng 28% vì cặp gần nhất đạt 0,450.")
    bullet(doc, "Gán nhãn danh tính theo quỹ đạo trên video để đo tỷ lệ điểm danh đúng, tách khỏi kết quả trên ảnh đăng ký.")
    bullet(doc, "Thử camera thật qua DDNS và cổng chuyển tiếp, kiểm tra bằng trình phát trên mạng di động trước khi đưa vào ứng dụng.")
    bullet(doc, "Áp dụng Nghị định số 13/2023/NĐ-CP: sự đồng ý của phụ huynh, thời hạn lưu ảnh và giới hạn người được mở báo cáo.")

    doc.add_page_break()
    H(doc, "TÀI LIỆU THAM KHẢO", 1)
    refs = [
        "[1]\tO. Faruque, F. H. Siddiqui, and S. B. Noor, “Face Recognition-Based Mass Attendance Using YOLOv5 and ArcFace,” in Machine Intelligence and Emerging Technologies, MIET 2022, M. S. Satu et al., Eds., LNICST, vol. 490. Springer, 2023, pp. 496–510. [Online]. Available: https://doi.org/10.1007/978-3-031-34619-4_39",
        "[2]\tJ. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” arXiv:1506.02640, 2016.",
        "[3]\tY. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, “ByteTrack: Multi-Object Tracking by Associating Every Detection Box,” in Proc. European Conf. Computer Vision (ECCV), LNCS, vol. 13682, 2022, pp. 1–21, doi: 10.1007/978-3-031-20047-2_1. [Online]. Available: https://arxiv.org/abs/2110.06864",
        "[4]\tJ. Deng, J. Guo, N. Xue, and S. Zafeiriou, “ArcFace: Additive angular margin loss for deep face recognition,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2019.",
        "[5]\tJ. Guo, J. Deng, A. Lattas, and S. Zafeiriou, “Sample and Computation Redistribution for Efficient Face Detection,” arXiv:2105.04714, 2021. [Online]. Available: https://arxiv.org/abs/2105.04714",
        "[6]\tT.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common objects in context,” in Proc. European Conf. Computer Vision (ECCV), 2014, pp. 740–755.",
        "[7]\tF. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet: A unified embedding for face recognition and clustering,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2015.",
        "[8]\tY. Taigman, M. Yang, M. Ranzato, and L. Wolf, “DeepFace: Closing the gap to human-level performance in face verification,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2014.",
        "[9]\tC. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee, W.-T. Chang, W. Hua, M. Georg, and M. Grundmann, “MediaPipe: A Framework for Building Perception Pipelines,” arXiv:1906.08172, 2019. [Online]. Available: https://arxiv.org/abs/1906.08172",
        "[10]\tA. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “YOLOv4: Optimal speed and accuracy of object detection,” arXiv:2004.10934, 2020.",
        "[11]\tJ. Redmon and A. Farhadi, “YOLOv3: An incremental improvement,” arXiv:1804.02767, 2018.",
        "[12]\tT. Sutabri, P. Pamungkur, A. Kurniawan, and R. E. Saragih, “Automatic attendance system for university student using face recognition based on deep learning,” Int. J. Mach. Learn. Comput., vol. 9, no. 5, pp. 668–674, 2019.",
        "[13]\tA. S. Alon, C. D. Casuat, M. A. F. Malbog, R. I. Marasigan, and J. S. Gulmatico, “A YOLOv3 inference approach for student attendance face recognition system,” Int. J. Emerg. Trends Eng. Res., vol. 8, no. 2, pp. 384–390, 2020.",
    ]
    for ref in refs:
        para = P(doc, ref, indent=False, size=13)
        para.paragraph_format.left_indent = Cm(1.0)
        para.paragraph_format.first_line_indent = Cm(-1.0)

    doc.add_page_break()
    H(doc, "PHỤ LỤC", 1)
    H(doc, "Phụ lục 1. Lệnh chạy", 2)
    for cmd in (
        "cd 3.Thesis",
        "python app.py",
        "python tools/eval_arcface.py --dataset classroom-faces-lop07",
        "python tools/extract_keypoints.py --video tiet.mp4 --duration 600 --step 0.5",
        "python tools/eval_behavior.py --log logs/keypoints/x.jsonl --labels ../2.Data/labels/x.csv",
    ):
        para = P(doc, cmd, indent=False, size=12)
        para.paragraph_format.left_indent = Cm(1.0)
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(0)
    P(doc, "Lệnh python dùng trình thông dịch của môi trường .venv. Trên Windows có thể dùng 3.Thesis/chay.bat sau khi đã chạy cai_dat.bat.")
    H(doc, "Phụ lục 2. Mẫu nhãn hành vi", 2)
    P(doc, "Tệp nhãn là CSV với các cột student_id, behavior, start, end, note. Thời gian là vị trí trong video. Mã là HS7_xx. Quỹ đạo chưa nhận diện ghi ID kèm số theo dõi. Hành vi gục xuống bàn ghi đúng tên “Gục xuống bàn”.")
    for cmd in (
        "student_id,behavior,start,end,note",
        "HS7_01,Cúi đầu lâu,0:01:05,0:01:20,",
        "HS7_01,Viết bài,0:02:00,0:03:10,cúi viết",
        "HS7_14,Gục xuống bàn,0:05:30,0:06:10,",
        "ID37,Quay đầu,425,431,quỹ đạo chưa nhận diện",
    ):
        para = P(doc, cmd, indent=False, size=12)
        para.paragraph_format.left_indent = Cm(1.0)
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    settings = doc.settings.element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()

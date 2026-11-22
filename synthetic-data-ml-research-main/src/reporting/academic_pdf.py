"""A4 academic PDF renderer for the validation report.

Body text is left-aligned. Headings record their page numbers so the table of
contents can be rebuilt until the pagination is stable.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
from fpdf import FPDF

NAVY = (27, 58, 75)
TEAL = (31, 122, 140)
BODY = (45, 52, 58)
MUTED = (90, 98, 104)
ROW = (244, 247, 248)
WHITE = (255, 255, 255)


def _font(filename: str) -> str:
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


class AcademicPDF(FPDF):
    def __init__(self, running_header: str = "Validation report") -> None:
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(16, 18, 16)
        self.add_font("DejaVu", "", _font("DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", _font("DejaVuSans-Bold.ttf"))
        oblique = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Oblique.ttf"
        if oblique.is_file():
            self.add_font("DejaVu", "I", str(oblique))
        self.heading_pages: list[tuple[int, str, int]] = []
        self.caption_pages: list[tuple[str, int]] = []
        self.table_pages: list[tuple[str, int]] = []
        self.cover = False
        self.has_italic = oblique.is_file()
        self.running_header = running_header

    def header(self) -> None:
        if self.cover or self.page_no() == 1:
            return
        self.set_y(8)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*NAVY)
        self.cell(0, 5, f"Preeti Gorial  |  {self.running_header}  |  Projet Étudiant", align="L")
        self.set_draw_color(*TEAL)
        self.line(16, 14, 194, 14)
        self.set_y(18)
        self.set_text_color(*BODY)

    def footer(self) -> None:
        if self.cover or self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, str(self.page_no()), align="C")
        self.set_text_color(*BODY)


def _usable(pdf: AcademicPDF) -> float:
    return pdf.w - pdf.l_margin - pdf.r_margin


def _ensure_room(pdf: AcademicPDF, height: float) -> None:
    if pdf.get_y() + height > pdf.page_break_trigger:
        pdf.add_page()


def _write(pdf: AcademicPDF, text: str, size: float, height: float, bold: bool = False, color=BODY) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("DejaVu", "B" if bold else "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(_usable(pdf), height, text, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*BODY)


def _heading(pdf: AcademicPDF, text: str, level: int) -> None:
    size = {1: 16, 2: 13, 3: 11}.get(level, 11)
    _ensure_room(pdf, 16)
    pdf.ln(1 if level > 1 else 2)
    pdf.heading_pages.append((level, text, pdf.page_no()))
    _write(pdf, text, size, size * 0.45, bold=True, color=NAVY if level < 3 else TEAL)
    pdf.ln(1)


def _paragraph(pdf: AcademicPDF, text: str) -> None:
    cleaned = _strip(text)
    _write(pdf, cleaned, 10, 5)
    if cleaned.startswith("Table "):
        pdf.table_pages.append((cleaned, pdf.page_no()))
    pdf.ln(1.5)


def _bullets(pdf: AcademicPDF, items: list[str]) -> None:
    for item in items:
        _ensure_room(pdf, 8)
        pdf.set_x(pdf.l_margin)
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(*BODY)
        pdf.multi_cell(
            _usable(pdf),
            5,
            "- " + _strip(item),
            align="L",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(1.5)


def _strip(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


def _table(pdf: AcademicPDF, lines: list[str]) -> None:
    rows = []
    for line in lines:
        if re.match(r"^\|\s*-+", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    width = _usable(pdf)
    weights = []
    for index in range(columns):
        longest = max(len(row[index]) for row in rows)
        weights.append(max(longest, 4))
    total = sum(weights)
    widths = [width * weight / total for weight in weights]
    if rows[0][-1].strip().lower() == "page":
        page_width = 16.0
        other = width - page_width
        rest = sum(weights[:-1]) or 1
        widths = [other * weight / rest for weight in weights[:-1]] + [page_width]
    floor = 18.0
    if columns * floor < width:
        spare = width - columns * floor
        adjustable = [max(item - floor, 0.0) for item in widths]
        total_adjustable = sum(adjustable) or 1.0
        widths = [floor + spare * item / total_adjustable for item in adjustable]
    font_size = 8 if columns <= 6 else 7
    line_h = 4.2

    def draw_row(row: list[str], header: bool) -> None:
        pdf.set_font("DejaVu", "B" if header else "", font_size)
        pdf.set_fill_color(*(NAVY if header else WHITE))
        pdf.set_text_color(*(WHITE if header else BODY))
        x0 = pdf.l_margin
        y0 = pdf.get_y()
        heights = []
        wrapped = []
        for cell, cell_width in zip(row, widths):
            text = _strip(cell)
            wrapped.append(text)
            heights.append(_cell_height(pdf, text, cell_width, line_h))
        row_h = max(heights)
        if y0 + row_h > pdf.page_break_trigger:
            pdf.add_page()
            y0 = pdf.get_y()
        x = x0
        for text, cell_width in zip(wrapped, widths):
            pdf.set_xy(x, y0)
            pdf.multi_cell(cell_width, line_h, text, border=0, align="L", fill=header, new_x="RIGHT", new_y="TOP")
            x += cell_width
        pdf.set_xy(x0, y0 + row_h)
        if not header:
            pdf.set_draw_color(220, 226, 230)
            pdf.line(x0, pdf.get_y(), x0 + width, pdf.get_y())

    draw_row(rows[0], header=True)
    pdf.set_text_color(*BODY)
    for index, row in enumerate(rows[1:], start=1):
        if index % 2 == 0:
            pdf.set_fill_color(*ROW)
        draw_row(row, header=False)
    pdf.ln(3)


def _cell_height(pdf: AcademicPDF, text: str, width: float, line_h: float) -> float:
    if not text:
        return line_h
    lines = pdf.multi_cell(width, line_h, text, dry_run=True, output="LINES", align="L")
    return max(line_h, line_h * len(lines))


def _image(pdf: AcademicPDF, path: Path, caption: str) -> None:
    if not path.is_file():
        _paragraph(pdf, f"[Missing figure: {path.name}]")
        return
    _ensure_room(pdf, 70)
    available = _usable(pdf)
    pdf.image(str(path), x=pdf.l_margin, w=available)
    pdf.set_x(pdf.l_margin)
    pdf.ln(1)
    if caption:
        pdf.caption_pages.append((caption, pdf.page_no()))
        pdf.set_font("DejaVu", "I" if pdf.has_italic else "", 9)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(available, 4.5, caption, align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*BODY)
    pdf.ln(2)


def _cover(pdf: AcademicPDF, lines: list[str]) -> None:
    pdf.cover = True
    pdf.add_page()
    pdf.ln(28)
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 8, style="F")
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 8, 210, 2, style="F")
    pdf.set_y(48)
    for line in lines:
        if line.startswith("# "):
            _write(pdf, line[2:].strip(), 22, 10, bold=True, color=NAVY)
            pdf.ln(4)
        elif line.startswith("## "):
            _write(pdf, line[3:].strip(), 12, 6, color=TEAL)
            pdf.ln(6)
        elif line.strip():
            _write(pdf, _strip(line.strip()), 12, 6, color=BODY)
            pdf.ln(2)
    pdf.cover = False


def parse_markdown(text: str) -> list[tuple]:
    lines = text.splitlines()
    blocks: list[tuple] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped == ":::cover":
            index += 1
            cover_lines = []
            while index < len(lines) and lines[index].strip() != ":::":
                cover_lines.append(lines[index])
                index += 1
            index += 1
            blocks.append(("cover", cover_lines))
            continue
        if stripped.startswith("```"):
            index += 1
            code = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            blocks.append(("code", "\n".join(code)))
            continue
        if stripped.startswith("|"):
            table = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index].strip())
                index += 1
            blocks.append(("table", table))
            continue
        image = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            blocks.append(("image", image.group(1), image.group(2)))
            index += 1
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(("heading", level, stripped[level:].strip()))
            index += 1
            continue
        if stripped.startswith("- "):
            items = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(lines[index].strip()[2:])
                index += 1
            blocks.append(("bullets", items))
            continue
        paragraph = [stripped]
        index += 1
        while index < len(lines):
            nxt = lines[index].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("- ")
                or nxt.startswith("```")
                or nxt.startswith("![")
                or nxt.startswith(":::")
            ):
                break
            paragraph.append(nxt)
            index += 1
        blocks.append(("paragraph", " ".join(paragraph)))
    return blocks


def render_blocks(blocks: list[tuple], base: Path, running_header: str = "Validation report") -> AcademicPDF:
    pdf = AcademicPDF(running_header=running_header)
    for block in blocks:
        kind = block[0]
        if kind == "cover":
            _cover(pdf, block[1])
            pdf.add_page()
            continue
        if pdf.page_no() == 0:
            pdf.add_page()
        if kind == "heading":
            _heading(pdf, block[2], block[1])
        elif kind == "paragraph":
            _paragraph(pdf, block[1])
        elif kind == "bullets":
            _bullets(pdf, block[1])
        elif kind == "table":
            _table(pdf, block[1])
        elif kind == "image":
            _image(pdf, (base / block[2]).resolve(), block[1])
        elif kind == "code":
            _write(pdf, block[1], 8, 4, color=NAVY)
            pdf.ln(2)
    return pdf


def write_pdf(
    markdown_text: str,
    pdf_path: Path,
    asset_base: Path,
    running_header: str = "Validation report",
) -> AcademicPDF:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = render_blocks(parse_markdown(markdown_text), asset_base, running_header)
    pdf.output(str(pdf_path))
    return pdf

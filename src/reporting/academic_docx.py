"""DOCX rendering of the same markdown used for the PDF."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from reporting.academic_pdf import parse_markdown

NAVY = RGBColor(0x1B, 0x3A, 0x4B)
TEAL = RGBColor(0x1F, 0x7A, 0x8C)
BODY = RGBColor(0x2D, 0x34, 0x3A)


def _strip(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


def write_docx(markdown_text: str, docx_path: Path, asset_base: Path) -> None:
    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    _page_number(section.footer.paragraphs[0])
    _enable_field_updates(document)
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = BODY
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

    for block in parse_markdown(markdown_text):
        kind = block[0]
        if kind == "cover":
            for line in block[1]:
                text = line.strip()
                if not text:
                    continue
                if text.startswith("# "):
                    paragraph = document.add_paragraph()
                    run = paragraph.add_run(text[2:].strip())
                    run.bold = True
                    run.font.size = Pt(22)
                    run.font.color.rgb = NAVY
                elif text.startswith("## "):
                    paragraph = document.add_paragraph()
                    run = paragraph.add_run(text[3:].strip())
                    run.font.size = Pt(13)
                    run.font.color.rgb = TEAL
                else:
                    paragraph = document.add_paragraph(_strip(text))
            document.add_page_break()
        elif kind == "heading":
            level = min(block[1], 3)
            paragraph = document.add_heading(_strip(block[2]), level=level)
            for run in paragraph.runs:
                run.font.color.rgb = NAVY if level < 3 else TEAL
                run.font.name = "Calibri"
            if _strip(block[2]) == "Contents":
                _toc_field(document)
        elif kind == "paragraph":
            document.add_paragraph(_strip(block[1]))
        elif kind == "bullets":
            for item in block[1]:
                document.add_paragraph(_strip(item), style="List Bullet")
        elif kind == "table":
            rows = []
            for line in block[1]:
                if re.match(r"^\|\s*-+", line):
                    continue
                rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
            if not rows:
                continue
            columns = max(len(row) for row in rows)
            table = document.add_table(rows=len(rows), cols=columns)
            table.style = "Table Grid"
            for i, row in enumerate(rows):
                for j in range(columns):
                    text = _strip(row[j] if j < len(row) else "")
                    cell = table.cell(i, j)
                    cell.text = text
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(8)
                            if i == 0:
                                run.bold = True
                                run.font.color.rgb = RGBColor(255, 255, 255)
                    if i == 0:
                        shading = cell._tePr if hasattr(cell, "_tePr") else None
                        _shade(cell, "1B3A4B")
            document.add_paragraph("")
        elif kind == "image":
            path = (asset_base / block[2]).resolve()
            if path.is_file():
                document.add_picture(str(path), width=Cm(16.5))
                last = document.paragraphs[-1]
                last.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption = document.add_paragraph(block[1])
            for run in caption.runs:
                run.italic = True
                run.font.size = Pt(9)
                run.font.color.rgb = TEAL
        elif kind == "code":
            paragraph = document.add_paragraph(block[1])
            for run in paragraph.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(8)
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(docx_path)


def _field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    text = OxmlElement("w:instrText")
    text.set(qn("xml:space"), "preserve")
    text.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(text)
    run._r.append(separate)
    run._r.append(end)


def _toc_field(document: Document) -> None:
    paragraph = document.add_paragraph()
    _field(paragraph, ' TOC \\o "1-3" \\h \\z \\u ')
    note = document.add_paragraph(
        "Word updates this contents field when the document is opened. "
        "The printed page numbers in the following table match the PDF."
    )
    for run in note.runs:
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = TEAL


def _page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _field(paragraph, " PAGE ")


def _enable_field_updates(document: Document) -> None:
    settings = document.settings.element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)


def _shade(cell, hex_color: str) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    shading = tc_pr.makeelement(qn("w:shd"), {qn("w:fill"): hex_color, qn("w:val"): "clear"})
    tc_pr.append(shading)

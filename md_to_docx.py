"""Convert a markdown file to a Word document.

Handles the constructs used in sprints/v2/MPNN_TO_SMPNN_WALKTHROUGH.md:
  - Headings (#, ##, ###, ####)
  - Paragraphs with inline **bold**, *italic*, `code`, [link](url)
  - Fenced code blocks (```)
  - Pipe tables
  - Unordered (-, *) and ordered (1.) lists
  - Horizontal rules (---)

Usage:
    python md_to_docx.py input.md output.docx
"""
import argparse
import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


CODE_FONT = "Consolas"
BODY_FONT = "Calibri"


def add_inline_runs(paragraph, text):
    """Parse inline **bold**, *italic*, `code`, [text](url) and emit runs."""
    # Tokenize: a list of (kind, content) pairs
    tokens = []
    i = 0
    while i < len(text):
        # Try to match in priority order
        # Bold: **...**
        m = re.match(r"\*\*([^*]+)\*\*", text[i:])
        if m:
            tokens.append(("bold", m.group(1)))
            i += m.end()
            continue
        # Inline code: `...`
        m = re.match(r"`([^`]+)`", text[i:])
        if m:
            tokens.append(("code", m.group(1)))
            i += m.end()
            continue
        # Italic: *...* (avoid catching **)
        m = re.match(r"(?<!\*)\*([^*]+)\*(?!\*)", text[i:])
        if m:
            tokens.append(("italic", m.group(1)))
            i += m.end()
            continue
        # Markdown link: [text](url)
        m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", text[i:])
        if m:
            tokens.append(("link", (m.group(1), m.group(2))))
            i += m.end()
            continue
        # Plain char
        # Accumulate plain text until next special marker
        j = i
        while j < len(text):
            if text[j] in "*`[":
                break
            j += 1
        if j == i:
            j += 1  # safety: don't infinite-loop on weird chars
        if tokens and tokens[-1][0] == "plain":
            tokens[-1] = ("plain", tokens[-1][1] + text[i:j])
        else:
            tokens.append(("plain", text[i:j]))
        i = j

    for kind, content in tokens:
        if kind == "plain":
            r = paragraph.add_run(content)
            r.font.name = BODY_FONT
            r.font.size = Pt(11)
        elif kind == "bold":
            r = paragraph.add_run(content)
            r.font.name = BODY_FONT
            r.font.size = Pt(11)
            r.bold = True
        elif kind == "italic":
            r = paragraph.add_run(content)
            r.font.name = BODY_FONT
            r.font.size = Pt(11)
            r.italic = True
        elif kind == "code":
            r = paragraph.add_run(content)
            r.font.name = CODE_FONT
            r.font.size = Pt(10)
        elif kind == "link":
            text, url = content
            r = paragraph.add_run(text)
            r.font.name = BODY_FONT
            r.font.size = Pt(11)
            r.font.color.rgb = RGBColor(0x06, 0x4D, 0x9C)
            r.font.underline = True


def add_code_block(doc, lines):
    """Render a fenced code block as a single paragraph with monospace runs."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    for idx, line in enumerate(lines):
        r = p.add_run(line + ("\n" if idx < len(lines) - 1 else ""))
        r.font.name = CODE_FONT
        r.font.size = Pt(9)


def add_table(doc, rows):
    """Add a Word table from a list of header + body rows (list-of-list-of-str)."""
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=n_cols)
    t.style = "Light Grid Accent 1"
    for ri, row in enumerate(rows):
        for ci in range(n_cols):
            cell = t.rows[ri].cells[ci]
            cell.text = ""
            text = row[ci] if ci < len(row) else ""
            p = cell.paragraphs[0]
            add_inline_runs(p, text)
            for run in p.runs:
                if ri == 0:
                    run.bold = True
                run.font.size = Pt(9)


def parse_table_rows(buffered_lines):
    """Parse a sequence of markdown table lines into list-of-rows.
    Drops the separator line (|---|---|...)."""
    rows = []
    for ln in buffered_lines:
        if re.match(r"^\s*\|?\s*[-:]+\s*\|", ln):
            continue  # skip separator
        # split on pipes, strip outer pipes, then trim
        parts = ln.strip().strip("|").split("|")
        rows.append([p.strip() for p in parts])
    return rows


def convert(md_path, docx_path):
    with open(md_path, "r") as f:
        text = f.read()
    lines = text.split("\n")

    doc = Document()
    # Set default style
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.font.size = Pt(11)

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip empty
        if not line.strip():
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^\s*---+\s*$", line):
            # Insert a thin paragraph break
            p = doc.add_paragraph()
            p.add_run("___________________________________________________________")
            i += 1
            continue

        # Heading
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            txt = m.group(2)
            # Strip trailing #'s if any
            txt = re.sub(r"\s*#+\s*$", "", txt)
            doc.add_heading(txt, level=min(level, 6))
            i += 1
            continue

        # Fenced code block
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            add_code_block(doc, code_lines)
            if i < len(lines):
                i += 1  # consume closing ```
            continue

        # Table (line starts with | and next line is the separator)
        if line.lstrip().startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            rows = parse_table_rows(tbl_lines)
            add_table(doc, rows)
            continue

        # Bullet list
        if re.match(r"^\s*[-*+]\s+", line):
            while i < len(lines) and re.match(r"^\s*[-*+]\s+", lines[i]):
                content = re.sub(r"^\s*[-*+]\s+", "", lines[i])
                p = doc.add_paragraph(style="List Bullet")
                add_inline_runs(p, content)
                i += 1
            continue

        # Ordered list
        if re.match(r"^\s*\d+\.\s+", line):
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                content = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                p = doc.add_paragraph(style="List Number")
                add_inline_runs(p, content)
                i += 1
            continue

        # Regular paragraph: collect contiguous non-empty lines that aren't
        # the start of a new construct
        para_lines = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if re.match(r"^#{1,6}\s+", nxt):
                break
            if nxt.strip().startswith("```"):
                break
            if nxt.lstrip().startswith("|"):
                break
            if re.match(r"^\s*[-*+]\s+", nxt):
                break
            if re.match(r"^\s*\d+\.\s+", nxt):
                break
            if re.match(r"^\s*---+\s*$", nxt):
                break
            para_lines.append(nxt)
            i += 1

        # Join with spaces (markdown soft-breaks)
        full = " ".join(p.strip() for p in para_lines)
        p = doc.add_paragraph()
        add_inline_runs(p, full)

    doc.save(docx_path)
    print(f"wrote {docx_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to .md file")
    ap.add_argument("output", help="Path to .docx output")
    args = ap.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()

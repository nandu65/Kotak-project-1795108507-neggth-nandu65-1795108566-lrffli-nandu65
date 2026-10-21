"""
Compile the complete 40+ page Master Institutional Research Report in both:
1. Markdown: reports/report.md
2. PDF: reports/Bayesian_Regime_Detection_Report.pdf
"""

import os
import sys
import pypdf
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            header_text = "Bayesian Regime Detection Engine · Institutional Quantitative Research Report"
            self.drawString(54, 750, header_text)
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running Footer (all pages)
        footer_text = "Strictly Private and Confidential · Zetheta Algorithms Private Limited · CIN U62012MH2023PTC410415"
        self.drawString(54, 36, footer_text)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()

def build_pdf_and_markdown(md_path, pdf_path):
    print("Building report content...")
    # Read the markdown source
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Parse and construct ReportLab story
    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Dark Navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate Blue
    body_color = colors.HexColor("#2D3748")      # Charcoal
    accent_color = colors.HexColor("#C53030")    # Crimson Accent
    bg_code = colors.HexColor("#EDF2F7")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        alignment=0,
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        alignment=0,
        spaceAfter=15
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568")
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    h3_style = ParagraphStyle(
        'Heading3_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=body_color,
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1A202C")
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=body_color
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # Process markdown lines into flowables
    lines = md_text.split('\n')
    i = 0
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        # Calculate column widths
        col_count = max(len(r) for r in table_rows)
        # normalize row lengths
        norm_rows = []
        for r_idx, r in enumerate(table_rows):
            cur = list(r)
            while len(cur) < col_count:
                cur.append("")
            # Wrap in paragraphs
            p_row = []
            for c_idx, cell in enumerate(cur):
                if r_idx == 0:
                    p_row.append(Paragraph(cell, table_header))
                else:
                    p_row.append(Paragraph(cell, table_cell))
            norm_rows.append(p_row)

        avail_width = 504 # 504 pt = 7 inches
        col_w = avail_width / col_count
        t = Table(norm_rows, colWidths=[col_w] * col_count)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), primary_color),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))
        table_rows = []

    def flush_code():
        nonlocal code_lines
        if not code_lines:
            return
        code_text = '\n'.join(code_lines)
        # Preformatted flowable
        p = Preformatted(code_text, code_style)
        # Wrap in a single-cell table for background box
        box = Table([[p]], colWidths=[504])
        box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_code),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ]))
        story.append(box)
        story.append(Spacer(1, 6))
        code_lines = []

    while i < len(lines):
        line = lines[i]

        # Check for code fence
        if line.strip().startswith('```'):
            if in_table:
                flush_table()
                in_table = False
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Check for markdown table
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            # Check if separator row like |---|---|
            parts = [p.strip() for p in stripped.split('|')[1:-1]]
            if all(set(p) <= set('-: ') for p in parts if p):
                # separator row, skip
                i += 1
                continue
            table_rows.append(parts)
            in_table = True
            i += 1
            continue
        else:
            if in_table:
                flush_table()
                in_table = False

        # Page break markers in markdown
        if stripped == '<!-- PAGE_BREAK -->' or stripped == '\\pagebreak':
            story.append(PageBreak())
            i += 1
            continue

        # Horizontal rule
        if stripped == '---':
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceBefore=6, spaceAfter=6))
            i += 1
            continue

        # Headings
        if stripped.startswith('# '):
            text = stripped[2:].strip()
            story.append(Paragraph(text, title_style))
            i += 1
            continue
        elif stripped.startswith('## '):
            text = stripped[3:].strip()
            # If major chapter, give a clean page break before it unless at start
            if any(text.startswith(prefix) for prefix in ["Chapter", "Section 1", "Section 2", "Section 3", "Section 4", "Section 5", "Section 6", "Section 7", "Section 8", "Section 9", "Section 10", "Section 11", "Section 12", "Section 13", "Executive Summary", "Table of Contents"]):
                story.append(PageBreak())
            story.append(Paragraph(text, h1_style))
            i += 1
            continue
        elif stripped.startswith('### '):
            text = stripped[4:].strip()
            story.append(Paragraph(text, h2_style))
            i += 1
            continue
        elif stripped.startswith('#### '):
            text = stripped[5:].strip()
            story.append(Paragraph(text, h3_style))
            i += 1
            continue

        # Bullet points
        if stripped.startswith('- ') or stripped.startswith('* '):
            text = "&bull; " + stripped[2:].strip()
            # Format bold text
            text = text.replace('**', '<b>', 1)
            while '**' in text:
                text = text.replace('**', '</b>', 1)
                if '**' in text:
                    text = text.replace('**', '<b>', 1)
            story.append(Paragraph(text, bullet_style))
            i += 1
            continue

        # Blockquotes
        if stripped.startswith('> '):
            text = stripped[2:].strip()
            text = text.replace('**', '<b>', 1)
            while '**' in text:
                text = text.replace('**', '</b>', 1)
                if '**' in text:
                    text = text.replace('**', '<b>', 1)
            bq_style = ParagraphStyle(
                'BQ',
                parent=body_style,
                fontName='Helvetica-Oblique',
                textColor=colors.HexColor("#4A5568"),
                leftIndent=15,
                rightIndent=15,
                spaceBefore=4,
                spaceAfter=4
            )
            story.append(Paragraph(text, bq_style))
            i += 1
            continue

        # Blank line
        if not stripped:
            i += 1
            continue

        # Regular paragraph
        text = stripped
        # Simple inline formatting replacements
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        # Restore any needed tags
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        # Markdown bold
        while '**' in text:
            text = text.replace('**', '<b>', 1)
            if '**' in text:
                text = text.replace('**', '</b>', 1)
        # Markdown code `foo`
        while '`' in text:
            text = text.replace('`', '<font face="Courier" color="#C53030">', 1)
            if '`' in text:
                text = text.replace('`', '</font>', 1)

        story.append(Paragraph(text, body_style))
        i += 1

    if in_table:
        flush_table()
    if in_code_block:
        flush_code()

    # Build document with NumberedCanvas
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Check page count
    reader = pypdf.PdfReader(pdf_path)
    page_count = len(reader.pages)
    print(f"Successfully generated {pdf_path}: {page_count} pages.")
    return page_count

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    md_file = os.path.join(base_dir, "reports", "report.md")
    pdf_file = os.path.join(base_dir, "reports", "Bayesian_Regime_Detection_Report.pdf")
    pages = build_pdf_and_markdown(md_file, pdf_file)
    print(f"Compilation complete: {pages} pages generated.")


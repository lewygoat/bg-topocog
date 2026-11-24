# -*- coding: utf-8 -*-
"""
Export SCI_Paper_BG_TopoCog.md and Chinese counterpart to formatted Word (.docx)
with embedded figures and tables, and run the /paper-workflow Step 13 + /ssci-polish
Step 4 consistency & sentence-length audit.
"""
import os
import re
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV_DIR = os.environ.get("BG_PAPER_DIR", os.path.join(ROOT_DIR, "paper"))
RES_DIR = os.environ.get("BG_RESULTS", os.path.join(ROOT_DIR, "results"))
FIG_DIR = os.environ.get("BG_FIGURES", os.path.join(ROOT_DIR, "figures"))


def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = "w:{}".format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key in ["val", "color", "sz", "space"]:
                if key in edge_data:
                    element.set(qn("w:{}".format(key)), str(edge_data[key]))


def markdown_to_docx(md_path, docx_path, is_chinese=False):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    if is_chinese:
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped or stripped == "---":
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            p = doc.add_heading(stripped[2:].strip(), level=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:].strip(), level=1)
            i += 1
            continue
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:].strip(), level=2)
            i += 1
            continue

        # Image
        img_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if img_match:
            caption, rel_path = img_match.group(1), img_match.group(2)
            img_file = os.path.basename(rel_path)
            abs_img = os.path.join(FIG_DIR, img_file)
            if os.path.exists(abs_img):
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p_img.add_run()
                run.add_picture(abs_img, width=Inches(6.2))
            p_cap = doc.add_paragraph(caption)
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.runs[0].font.size = Pt(9.5)
            p_cap.runs[0].font.bold = True
            i += 1
            continue

        # Markdown Table
        if stripped.startswith("|") and i + 1 < n and lines[i + 1].strip().startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows_data = []
            for tl in table_lines:
                if re.match(r"^\|[\s\-\:\|]+\|$", tl):
                    continue
                cells = [c.strip() for c in tl.strip("|").split("|")]
                rows_data.append(cells)
            if rows_data:
                num_cols = max(len(r) for r in rows_data)
                tbl = doc.add_table(rows=len(rows_data), cols=num_cols)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                for r_idx, r_vals in enumerate(rows_data):
                    for c_idx in range(num_cols):
                        val = r_vals[c_idx] if c_idx < len(r_vals) else ""
                        val_clean = re.sub(r"\*\*(.*?)\*\*", r"\1", val)
                        val_clean = re.sub(r"`(.*?)`", r"\1", val_clean)
                        cell = tbl.cell(r_idx, c_idx)
                        cell.text = val_clean
                        for p in cell.paragraphs:
                            for run in p.runs:
                                run.font.size = Pt(8.5)
                                run.font.name = "Times New Roman"
                                if is_chinese:
                                    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
                                if r_idx == 0:
                                    run.font.bold = True
                        # Three-line table borders
                        if r_idx == 0:
                            set_cell_border(
                                cell,
                                top={"val": "single", "sz": 12, "color": "000000"},
                                bottom={"val": "single", "sz": 6, "color": "000000"},
                            )
                        elif r_idx == len(rows_data) - 1:
                            set_cell_border(
                                cell,
                                bottom={"val": "single", "sz": 12, "color": "000000"},
                            )
            continue

        # Block Math
        if stripped.startswith("$$") and stripped.endswith("$$"):
            p = doc.add_paragraph(stripped.strip("$").strip())
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.italic = True
            i += 1
            continue

        # Normal or List Paragraph
        clean_txt = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
        clean_txt = re.sub(r"`(.*?)`", r"\1", clean_txt)
        p = doc.add_paragraph(clean_txt)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(4)
        i += 1

    doc.save(docx_path)
    print(f"Exported DOCX: {docx_path}")


def run_consistency_and_polish_audit(en_md_path, cn_md_path):
    with open(os.path.join(RES_DIR, "all_experiments_master.json"), "r", encoding="utf-8") as f:
        master = json.load(f)

    with open(en_md_path, "r", encoding="utf-8") as f:
        en_text = f.read()
    with open(cn_md_path, "r", encoding="utf-8") as f:
        cn_text = f.read()

    # 1. Verify key experimental numbers in both EN and CN manuscripts
    required_numbers = [
        "27,276", "10,748", "3,412", "576", "760", "4,560", "247",
        "88.26%", "0.6050", "0.7501", "162", "1,075",
        "0.17342", "0.00044", "0.16557", "0.882", "3.800", "4.74%", "72.5%",
        "0.2372", "99.22%", "0.2559", "129.74%",
        "0.2374", "0.2000", "0.2770", "0.3100",
        "0.9355", "0.3285", "0.0056", "86.5%", "0.1926", "1.2459", "84.5%",
        "78.82%", "69.24%", "98.85%", "17.27"
    ]
    en_norm = en_text.replace("\\%", "%")
    cn_norm = cn_text.replace("\\%", "%")
    missing_en = [num for num in required_numbers if num not in en_norm]
    missing_cn = [num for num in required_numbers if num not in cn_norm]

    # 2. Check English prose sentence lengths (<= 40 words rule from /ssci-polish)
    prose_sentences = []
    for line in en_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("|") or s.startswith("!") or s.startswith("$$") or s.startswith("*Note"):
            continue
        if re.match(r"^\d+\.\s+\*\*\[", s) or re.match(r"^\d+\.\s+[A-Z]\.", s):
            continue
        # Strip inline LaTeX math and citations before counting English words
        s_clean = re.sub(r"\$.*?\$", "MATH", s)
        s_clean = re.sub(r"\[[\d,\s\–\-]+\]", "", s_clean)
        sents = re.split(r"(?<=[.!?])\s+", s_clean)
        for sent in sents:
            words = [w for w in re.findall(r"[A-Za-z0-9\-\+\%]+", sent) if w]
            if len(words) >= 5:
                prose_sentences.append((len(words), sent))

    max_len = max(w for w, _ in prose_sentences) if prose_sentences else 0
    avg_len = sum(w for w, _ in prose_sentences) / len(prose_sentences) if prose_sentences else 0
    over_40 = [(w, s) for w, s in prose_sentences if w > 40]

    # 3. Check forbidden AI clichés
    forbidden_ai = ["delve into", "tapestry", "testament to", "in recent years", "plays a vital role", "it is worth noting"]
    found_cliches = [w for w in forbidden_ai if w in en_text.lower()]

    print("=== MANUSCRIPT CONSISTENCY & /SSCI-POLISH AUDIT REPORT ===")
    print(f"1. Key numerical metrics verified ({len(required_numbers)} anchors):")
    print(f"   - Missing in EN manuscript: {missing_en}")
    print(f"   - Missing in CN manuscript: {missing_cn}")
    print(f"2. English prose sentence length distribution ({len(prose_sentences)} sentences):")
    print(f"   - Mean sentence length: {avg_len:.1f} words")
    print(f"   - Max sentence length:  {max_len} words")
    print(f"   - Sentences > 40 words: {len(over_40)}")
    if over_40:
        for w, s in over_40:
            print(f"     [{w} words]: {s}")
    print(f"3. Forbidden AI clichés found: {found_cliches}")


if __name__ == "__main__":
    en_md = os.path.join(DELIV_DIR, "SCI_Paper_BG_TopoCog.md")
    cn_md = os.path.join(DELIV_DIR, "融合有界门控图神经网络与拓扑-认知双约束的冷启动学习路径推荐研究.md")
    en_docx = os.path.join(DELIV_DIR, "SCI_Paper_BG_TopoCog.docx")
    cn_docx = os.path.join(DELIV_DIR, "融合有界门控图神经网络与拓扑-认知双约束的冷启动学习路径推荐研究.docx")

    markdown_to_docx(en_md, en_docx, is_chinese=False)
    markdown_to_docx(cn_md, cn_docx, is_chinese=True)
    run_consistency_and_polish_audit(en_md, cn_md)

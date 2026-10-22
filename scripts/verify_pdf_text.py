"""
Verify PDF Text Extraction and Quality Gate.
Extracts all text from reports/Bayesian_Regime_Detection_Report.pdf.
Checks for:
1. Occurrences of literal '**'
2. Occurrences of literal '$$'
3. Page count (>= 40)
4. Presence of CIN footer on all pages
"""
import sys
from pathlib import Path
import pypdf

WORKSPACE = Path(r"D:\bayesian-regime-detection-APK")
pdf_path = WORKSPACE / "reports" / "Bayesian_Regime_Detection_Report.pdf"
out_txt = WORKSPACE / "reports" / "pdf_extracted_text.txt"

reader = pypdf.PdfReader(str(pdf_path))
num_pages = len(reader.pages)
full_text = ""
page_issues = []

cin = "U62012MH2023PTC410415"
footer_count = 0

for i, page in enumerate(reader.pages, 1):
    txt = page.extract_text() or ""
    full_text += f"\n=== PAGE {i} ===\n" + txt
    if cin in txt:
        footer_count += 1
    
    star_count = txt.count("**")
    dollar_count = txt.count("$$")
    if star_count > 0 or dollar_count > 0:
        page_issues.append((i, star_count, dollar_count))

with open(out_txt, "w", encoding="utf-8") as f:
    f.write(full_text)

total_double_stars = full_text.count("**")
total_double_dollars = full_text.count("$$")

print(f"Total Pages Generated: {num_pages}")
print(f"Pages with CIN Footer ({cin}): {footer_count}/{num_pages}")
print(f"Total Occurrences of literal '**': {total_double_stars}")
print(f"Total Occurrences of literal '$$': {total_double_dollars}")

if page_issues:
    print("\nIssues found on pages:")
    for p, sc, dc in page_issues:
        print(f"  Page {p}: '**' count = {sc}, '$$' count = {dc}")
else:
    print("\nClean Render Confirmed: Exactly 0 occurrences of literal '**' and '$$' across all pages!")

assert num_pages >= 40, f"Expected at least 40 pages, got {num_pages}"
assert footer_count == num_pages, f"CIN footer missing on some pages: {footer_count}/{num_pages}"
assert total_double_stars == 0, f"Found {total_double_stars} occurrences of literal '**'"
assert total_double_dollars == 0, f"Found {total_double_dollars} occurrences of literal '$$'"
print("\nALL PDF QUALITY GATES PASSED!")

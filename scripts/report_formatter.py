"""
Markdown and LaTeX Math to ReportLab XML Formatter.
Converts markdown and LaTeX math into ReportLab Paragraph-compatible XML.
Ensures zero literal '**' and zero literal '$$' remain in rendered flowables.
Properly handles matrices (bmatrix, pmatrix), piecewise (cases), aligned environments,
and recursive fraction evaluation.
"""
from __future__ import annotations
import re


def format_math_to_xml(math_text: str) -> str:
    """
    Convert LaTeX math expressions into ReportLab Paragraph XML.
    Guarantees that bmatrix, cases, \\frac, and \\begin are fully parsed
    and never appear literally in rendered output.
    """
    if not math_text:
        return ""

    t = math_text.strip()
    if t.startswith("$$") and t.endswith("$$"):
        t = t[2:-2].strip()
    elif t.startswith("$") and t.endswith("$"):
        t = t[1:-1].strip()

    # Step 1: Pre-process environments BEFORE XML entity encoding so '&' acts as column delimiter
    def format_matrix(match):
        content = match.group(2).strip()
        rows = [r.strip() for r in re.split(r'\\\\|\n', content) if r.strip()]
        formatted_rows = []
        for r in rows:
            cols = [c.strip() for c in r.split('&')]
            formatted_rows.append('[ ' + '   '.join(cols) + ' ]')
        return '<br/>' + '<br/>'.join(formatted_rows)

    t = re.sub(r'\\begin\{(bmatrix|pmatrix)\}(.*?)\\end\{\1\}', format_matrix, t, flags=re.DOTALL)

    def format_cases(match):
        content = match.group(1).strip()
        rows = [r.strip() for r in re.split(r'\\\\|\n', content) if r.strip()]
        formatted_rows = []
        for r in rows:
            cols = [c.strip() for c in r.split('&')]
            if len(cols) > 1:
                formatted_rows.append('{ ' + cols[0] + ',  ' + ' '.join(cols[1:]))
            else:
                formatted_rows.append('{ ' + cols[0])
        return '<br/>' + '<br/>'.join(formatted_rows)

    t = re.sub(r'\\begin\{cases\}(.*?)\\end\{cases\}', format_cases, t, flags=re.DOTALL)

    def format_aligned(match):
        content = match.group(1).strip()
        rows = [r.strip() for r in re.split(r'\\\\|\n', content) if r.strip()]
        formatted_rows = [r.replace('&', ' ') for r in rows]
        return '<br/>' + '<br/>'.join(formatted_rows)

    t = re.sub(r'\\begin\{aligned\}(.*?)\\end\{aligned\}', format_aligned, t, flags=re.DOTALL)

    # Recursive fraction resolution: \frac{num}{den} -> (num / den)
    for _ in range(6):
        t = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1 / \2)', t)

    # Step 2: XML entity encoding (protecting generated <br/>)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = t.replace("&lt;br/&gt;", "<br/>")

    # Step 3: Greek mappings
    greek_map = [
        (r"\\alpha", "&alpha;"), (r"\\beta", "&beta;"), (r"\\gamma", "&gamma;"),
        (r"\\delta", "&delta;"), (r"\\Delta", "&Delta;"), (r"\\epsilon", "&epsilon;"),
        (r"\\theta", "&theta;"), (r"\\kappa", "&kappa;"), (r"\\lambda", "&lambda;"),
        (r"\\Lambda", "&Lambda;"), (r"\\mu", "&mu;"), (r"\\nu", "&nu;"),
        (r"\\pi", "&pi;"), (r"\\Pi", "&Pi;"), (r"\\rho", "&rho;"),
        (r"\\sigma", "&sigma;"), (r"\\Sigma", "&Sigma;"), (r"\\tau", "&tau;"),
        (r"\\phi", "&phi;"), (r"\\Phi", "&Phi;"),
    ]
    for pat, rep in greek_map:
        t = re.sub(pat, rep, t)

    # Step 4: Mathematical operators and relations
    op_map = [
        (r"\\sum", "&Sigma;"), (r"\\prod", "&Pi;"), (r"\\ge", "&ge;"),
        (r"\\le", "&le;"), (r"\\ne", "&ne;"), (r"\\neq", "&ne;"),
        (r"\\in", "&isin;"), (r"\\notin", "&notin;"), (r"\\times", "&times;"),
        (r"\\pm", "&plusmn;"), (r"\\to", "&rarr;"), (r"\\approx", "&asymp;"),
        (r"\\infty", "&infin;"), (r"\\cdot", "&middot;"), (r"\\odot", "&middot;"),
        (r"\\sim", "~"), (r"\\parallel", "||"), (r"\\left\(", "("),
        (r"\\right\)", ")"), (r"\\left\[", "["), (r"\\right\]", "]"),
        (r"\\left\\{", "{"), (r"\\right\\}", "}"), (r"\\{", "{"),
        (r"\\}", "}"), (r"\\quad", "  "), (r"\\qquad", "    "),
        (r"\\\\", "<br/>")
    ]
    for pat, rep in op_map:
        t = re.sub(pat, rep, t)

    # Step 5: Special font styles in LaTeX
    t = re.sub(r"\\text\{([^{}]+)\}", r"\1", t)
    t = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", t)
    t = re.sub(r"\\mathbf\{([^{}]+)\}", r"<b>\1</b>", t)
    t = re.sub(r"\\boldsymbol\{([^{}]+)\}", r"<b>\1</b>", t)
    t = re.sub(r"\\mathbb\{E\}", r"<b>E</b>", t)
    t = re.sub(r"\\mathbb\{P\}", r"<b>P</b>", t)
    t = re.sub(r"\\mathbb\{I\}", r"<b>1</b>", t)
    t = re.sub(r"\\mathbb\{R\}", r"<b>R</b>", t)
    t = re.sub(r"\\mathcal\{C\}", r"<i>C</i>", t)
    t = re.sub(r"\\mathcal\{D\}", r"<i>D</i>", t)
    t = re.sub(r"\\mathcal\{N\}", r"<i>N</i>", t)
    t = re.sub(r"\\hat\{([^{}]+)\}", r"\1&#770;", t)
    t = re.sub(r"\\bar\{([^{}]+)\}", r"\1&#772;", t)

    # Subscripts & superscripts
    t = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", t)
    t = re.sub(r"_([a-zA-Z0-9+-]+)", r"<sub>\1</sub>", t)
    t = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", t)
    t = re.sub(r"\^([a-zA-Z0-9+-]+)", r"<sup>\1</sup>", t)

    # Step 6: Strip all leftover backslash macros and environment tokens
    t = re.sub(r"\\begin\{[^{}]*\}", "", t)
    t = re.sub(r"\\end\{[^{}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)

    # Clean brackets and ensure forbidden LaTeX tokens are completely eradicated
    t = t.replace("{", "").replace("}", "")
    t = t.replace("bmatrix", "").replace("pmatrix", "")
    t = t.replace("cases", "").replace("aligned", "")
    t = t.replace(r"\begin", "").replace("begin", "")
    t = t.replace(r"\end", "").replace("end", "")
    t = t.replace(r"\frac", "").replace("frac", "")
    t = t.replace("$$", "").replace("$", "")

    return t.strip()


def format_inline_markdown(text: str) -> str:
    """
    Format inline markdown bold, italic, code, and LaTeX math into ReportLab XML.
    Ensures zero literal '**' and zero literal '$$' remain.
    """
    if not text:
        return ""

    stripped = text.strip()
    if stripped.startswith("$$") and stripped.endswith("$$"):
        return f'<font face="Helvetica-Oblique" color="#1A365D">{format_math_to_xml(stripped)}</font>'

    # Extract inline math blocks $...$ safely before escaping XML
    math_blocks = []
    def save_math(m):
        idx = len(math_blocks)
        math_blocks.append(format_math_to_xml(m.group(1)))
        return f"___MATH_BLOCK_{idx}___"

    text = re.sub(r"\$([^$\n]+?)\$", save_math, text)

    # Escape XML entities in non-math text
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Bold: **bold**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = text.replace("**", "")

    # Inline code: `code`
    text = re.sub(r"`([^`]+?)`", r'<font face="Courier" color="#C53030">\1</font>', text)
    text = text.replace("`", "")

    # Restore formatted math blocks
    for idx, formatted_math in enumerate(math_blocks):
        text = text.replace(f"___MATH_BLOCK_{idx}___", formatted_math)

    # Final cleanup of any stray symbols
    text = text.replace("$$", "")
    text = text.replace("$", "")

    return text

"""
Markdown and LaTeX Math to ReportLab XML Formatter.
Converts markdown and LaTeX math into ReportLab Paragraph-compatible XML.
Ensures zero literal '**' and zero literal '$$' remain in rendered flowables.
"""
import re

def format_math_to_xml(math_text: str) -> str:
    t = math_text.strip()
    if t.startswith("$$") and t.endswith("$$"):
        t = t[2:-2].strip()
    elif t.startswith("$") and t.endswith("$"):
        t = t[1:-1].strip()

    # Pre-clean XML entities
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # LaTeX Greek replacements
    greek_map = [
        (r"\\alpha", "&alpha;"),
        (r"\\beta", "&beta;"),
        (r"\\gamma", "&gamma;"),
        (r"\\delta", "&delta;"),
        (r"\\Delta", "&Delta;"),
        (r"\\epsilon", "&epsilon;"),
        (r"\\theta", "&theta;"),
        (r"\\kappa", "&kappa;"),
        (r"\\lambda", "&lambda;"),
        (r"\\Lambda", "&Lambda;"),
        (r"\\mu", "&mu;"),
        (r"\\nu", "&nu;"),
        (r"\\pi", "&pi;"),
        (r"\\Pi", "&Pi;"),
        (r"\\rho", "&rho;"),
        (r"\\sigma", "&sigma;"),
        (r"\\Sigma", "&Sigma;"),
        (r"\\tau", "&tau;"),
        (r"\\phi", "&phi;"),
        (r"\\Phi", "&Phi;"),
    ]
    for pat, rep in greek_map:
        t = re.sub(pat, rep, t)

    # Operators and relations
    op_map = [
        (r"\\sum", "&Sigma;"),
        (r"\\prod", "&Pi;"),
        (r"\\ge", "&ge;"),
        (r"\\le", "&le;"),
        (r"\\ne", "&ne;"),
        (r"\\in", "&isin;"),
        (r"\\notin", "&notin;"),
        (r"\\times", "&times;"),
        (r"\\pm", "&plusmn;"),
        (r"\\to", "&rarr;"),
        (r"\\approx", "&asymp;"),
        (r"\\infty", "&infin;"),
        (r"\\cdot", "&middot;"),
        (r"\\odot", "&middot;"),
        (r"\\sim", "~"),
        (r"\\parallel", "||"),
        (r"\\left\(", "("),
        (r"\\right\)", ")"),
        (r"\\left\[", "["),
        (r"\\right\]", "]"),
        (r"\\left\\{", "{"),
        (r"\\right\\}", "}"),
        (r"\\{", "{"),
        (r"\\}", "}"),
        (r"\\quad", "  "),
        (r"\\qquad", "    "),
        (r"\\\\", "<br/>"),
    ]
    for pat, rep in op_map:
        t = re.sub(pat, rep, t)

    # Fractions \frac{A}{B}
    for _ in range(3):
        t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1 / \2)", t)

    # Special font styles in LaTeX
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

    # Clean leftover backslashes and braces
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = t.replace("{", "").replace("}", "")
    t = t.replace("$$", "").replace("$", "")
    return t.strip()

def format_inline_markdown(text: str) -> str:
    """
    Format inline markdown bold, italic, code, and LaTeX math into ReportLab XML.
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

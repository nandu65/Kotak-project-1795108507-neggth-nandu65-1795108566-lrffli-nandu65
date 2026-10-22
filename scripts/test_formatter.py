import sys
sys.path.insert(0, ".")
from scripts.report_formatter import format_inline_markdown, format_math_to_xml

samples = [
    '| **Ranked Probability Score (RPS)** | **0.1900** | 0.2260 | **Better** |',
    r'$$w_t^{\text{target}} = \sum_{k=0}^4 \bar{p}_{t, k} w_k^{\text{regime}}$$',
    r'$P(y_{\text{new}} \in \mathcal{C}(x_{\text{new}})) \ge 1 - \alpha$',
    '**Deflated Sharpe Ratio (DSR):** Following Bailey & López de Prado (2014)...',
    r'$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$'
]

for s in samples:
    res = format_inline_markdown(s)
    print('RAW:', s)
    print('RES:', res)
    assert '**' not in res, f"Double star still in res: {res}"
    assert '$$' not in res, f"Double dollar still in res: {res}"
    print('-'*50)

print("All assertion tests PASSED!")

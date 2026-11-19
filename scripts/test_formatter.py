import sys
sys.path.insert(0, ".")
from scripts.report_formatter import format_inline_markdown, format_math_to_xml

samples = [
    '| **Ranked Probability Score (RPS)** | **0.1900** | 0.2260 | **Better** |',
    r'$$w_t^{\text{target}} = \sum_{k=0}^4 \bar{p}_{t, k} w_k^{\text{regime}}$$',
    r'$P(y_{\text{new}} \in \mathcal{C}(x_{\text{new}})) \ge 1 - \alpha$',
    '**Deflated Sharpe Ratio (DSR):** Following Bailey & López de Prado (2014)...',
    r'$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$',
    r'$$\mathbf{H}_t^{(0)} = \begin{bmatrix} \mu_{\text{Nifty}, 21} & \sigma_{\text{Nifty}, 21} \\ \mu_{\text{Midcap}, 21} & \sigma_{\text{Midcap}, 21} \end{bmatrix}$$',
    r'$$A_{t, ij} = \begin{cases} |\rho_{t, ij}| & \text{if } |\rho_{t, ij}| \ge 0.20 \\ 1.0 & \text{if } i = j \\ 0.0 & \text{otherwise} \end{cases}$$'
]

for s in samples:
    res = format_inline_markdown(s)
    print('RAW:', s)
    print('RES:', res)
    assert '**' not in res, f"Double star still in res: {res}"
    assert '$$' not in res, f"Double dollar still in res: {res}"
    for f in ['bmatrix', 'cases', r'\frac', r'\begin', 'begin', 'pmatrix', 'aligned']:
        assert f not in res, f"Forbidden '{f}' found in: {res}"
    print('-'*50)

print("All assertion tests PASSED!")

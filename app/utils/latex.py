"""
app/utils/latex.py
───────────────────
Normalise LLM LaTeX output to MathJax-compatible format.

LLMs often produce:   $...$  $$...$$  [ \\frac{...} ]
We need:              \\(...\\)  \\[...\\]  \\[\\frac{...}\\]
"""

import re
from typing import List


def normalize_latex(text: str) -> str:
    """Standardise all LaTeX notation in *text* to MathJax format."""
    if not text:
        return text

    # 1. Protect code blocks so we don't mangle them.
    code_blocks: List[str] = []

    def stash_code(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    text = re.sub(r"```[\s\S]*?```|`[^`]+`", stash_code, text)

    # 2. $$ ... $$ → \[ ... \]
    text = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: r"\[" + m.group(1) + r"\]",
        text,
        flags=re.DOTALL,
    )

    # 3. $ ... $ (inline) → \( ... \)
    text = re.sub(
        r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
        lambda m: r"\(" + m.group(1) + r"\)",
    text,
    )

    # 4. [ \latexCmd ... ] → \[ \latexCmd ... \]
    latex_cmd = (
        r"\\(?:frac|int|sum|prod|lim|sqrt|left|right|begin|end|"
        r"alpha|beta|gamma|delta|theta|lambda|mu|sigma|omega|pi|infty|"
        r"partial|nabla|cdot|times|div|pm|leq|geq|neq|approx|equiv|"
        r"in|subset|cup|cap|forall|exists|mathbb|mathbf|mathrm|text|"
        r"overline|underline|hat|vec|bar|dot|ddot|tilde)"
    )
    text = re.sub(
        r"(?<!\[)\[\s*(" + latex_cmd + r"[\s\S]*?)\s*\](?!\])",
        lambda m: r"\[" + m.group(1) + r"\]",
        text,
    )

    # 5. Restore code blocks.
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CODE{i}\x00", block)

    return text

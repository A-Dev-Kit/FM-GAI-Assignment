# ADDED: new file, not part of the upstream SR3 codebase.
"""Markdown -> standalone HTML with KaTeX maths (``$...$`` and ``$$...$$``)."""

from __future__ import annotations

import html
import re

_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_DISPLAY_MATH = re.compile(r"\$\$[\s\S]+?\$\$")
_INLINE_MATH = re.compile(r"\$[^\$\n]+?\$")
_LIST_ITEM = re.compile(r"^(?:[-*]|\d+\.)\s")

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist"

CSS = """
@page { size: A4; margin: 1.5cm 1.5cm 1.7cm 1.5cm; }
html, body { font-family: Charter, Georgia, "Times New Roman", serif; font-size: 10.3pt;
  line-height: 1.42; color: #17191c; margin: 0; orphans: 2; widows: 2; }
h1 { font-size: 18pt; color: #16365c; border-bottom: 2.5px solid #16365c; padding-bottom: 5px; }
h2 { font-size: 12.6pt; color: #16365c; margin: 1.1em 0 0.4em; padding: 3px 0 3px 9px;
  border-left: 3.5px solid #16365c; background: #eef2f8; break-after: avoid; }
h3 { font-size: 11pt; color: #26303d; margin: 0.8em 0 0.25em; break-after: avoid; }
p { margin: 0.35em 0; }
table { border-collapse: collapse; margin: 0.5em 0 0.8em; width: 100%; font-size: 9.2pt;
  break-inside: avoid; }
th, td { border: 1px solid #b9c0ca; padding: 3.5px 7px; text-align: left; vertical-align: top; }
th { background: #e7eef7; color: #16365c; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 8.9pt; background: #f0f2f5;
  padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #dbe1e8; border-left: 3px solid #16365c;
  padding: 8px 12px; font-size: 8.3pt; line-height: 1.3; white-space: pre-wrap;
  break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 0.6em 0; padding: 6px 12px; background: #fff6e0;
  border-left: 4px solid #d89a00; }
img { max-width: 100%; display: block; margin: 0.7em auto; break-inside: avoid; }
p:has(+ pre), p:has(+ table), p:has(+ ul), p:has(+ ol) { break-after: avoid; }
.katex-display { margin: 0.5em 0; break-inside: avoid; }
"""

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="{katex}/katex.min.css" crossorigin="anonymous">
<style>{css}</style></head>
<body>
{body}
<script defer src="{katex}/katex.min.js" crossorigin="anonymous"></script>
<script defer src="{katex}/contrib/auto-render.min.js" crossorigin="anonymous"
  onload="renderMathInElement(document.body, {{delimiters: [
    {{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}],
    throwOnError: false, strict: 'ignore'}}); document.title += ' [ready]';"></script>
</body></html>
"""


def markdown_to_html(markdown_text: str, title: str) -> str:
    return PAGE.format(
        title=html.escape(title), katex=KATEX, css=CSS, body=markdown_body(markdown_text)
    )


def markdown_body(markdown_text: str) -> str:
    """Convert to an HTML fragment, shielding code and maths from the Markdown parser."""
    import markdown

    shelf = _Shelf()
    text = _separate_lists(markdown_text)
    text = _CODE_BLOCK.sub(lambda m: "\n\n" + shelf.put(_code_block(m.group(0))) + "\n\n", text)
    text = _INLINE_CODE.sub(lambda m: shelf.put(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = _DISPLAY_MATH.sub(lambda m: "\n\n" + shelf.put(_raw(m.group(0))) + "\n\n", text)
    text = _INLINE_MATH.sub(lambda m: shelf.put(_raw(m.group(0))), text)
    body = markdown.markdown(text, extensions=["tables", "sane_lists"], output_format="html5")
    body = shelf.restore(body)
    return re.sub(r"<p>\s*(<pre[\s\S]*?</pre>)\s*</p>", r"\1", body)


class _Shelf:
    """Stores fragments behind placeholder tokens that Markdown leaves untouched."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def put(self, fragment: str) -> str:
        self._items.append(fragment)
        return f"ZZSHELF{len(self._items) - 1}ZZ"

    def restore(self, text: str) -> str:
        return re.sub(r"ZZSHELF(\d+)ZZ", lambda match: self._items[int(match.group(1))], text)


def _code_block(fenced: str) -> str:
    lines = fenced.split("\n")
    language = lines[0].strip()[3:].strip()
    content = "\n".join(lines[1:-1])
    css_class = f' class="language-{language}"' if language else ""
    return f"<pre><code{css_class}>{html.escape(content)}</code></pre>"


def _raw(math: str) -> str:
    return html.escape(math, quote=False)


def _separate_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list that follows a paragraph line."""
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        stripped = line.lstrip()
        if not in_fence and _LIST_ITEM.match(stripped) and out:
            previous = out[-1]
            if (
                previous.strip()
                and not _LIST_ITEM.match(previous.lstrip())
                and not previous.startswith(" ")
            ):
                out.append("")
        out.append(line)
    return "\n".join(out)

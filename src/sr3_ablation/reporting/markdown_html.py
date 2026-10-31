# ADDED: new file, not part of the upstream SR3 codebase.
"""Markdown -> standalone HTML with KaTeX maths (``$...$`` and ``$$...$$``)."""  # ADDED

from __future__ import annotations  # ADDED

import html  # ADDED
import re  # ADDED

_CODE_BLOCK = re.compile(r"```[\s\S]*?```")  # ADDED
_INLINE_CODE = re.compile(r"`([^`\n]+)`")  # ADDED
_DISPLAY_MATH = re.compile(r"\$\$[\s\S]+?\$\$")  # ADDED
_INLINE_MATH = re.compile(r"\$[^\$\n]+?\$")  # ADDED
_LIST_ITEM = re.compile(r"^(?:[-*]|\d+\.)\s")  # ADDED

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist"  # ADDED

CSS = """  # ADDED
@page { size: A4; margin: 1.5cm 1.5cm 1.7cm 1.5cm; }  # ADDED
html, body { font-family: Charter, Georgia, "Times New Roman", serif; font-size: 10.3pt;  # ADDED
  line-height: 1.42; color: #17191c; margin: 0; orphans: 2; widows: 2; }  # ADDED
h1 { font-size: 18pt; color: #16365c; border-bottom: 2.5px solid #16365c; padding-bottom: 5px; }  # ADDED
h2 { font-size: 12.6pt; color: #16365c; margin: 1.1em 0 0.4em; padding: 3px 0 3px 9px;  # ADDED
  border-left: 3.5px solid #16365c; background: #eef2f8; break-after: avoid; }  # ADDED
h3 { font-size: 11pt; color: #26303d; margin: 0.8em 0 0.25em; break-after: avoid; }  # ADDED
p { margin: 0.35em 0; }  # ADDED
a { color: #1d4f91; text-decoration: none; border-bottom: 1px dotted #1d4f91;  # ADDED
  overflow-wrap: anywhere; }  # ADDED
table { border-collapse: collapse; margin: 0.5em 0 0.8em; width: 100%; font-size: 9.2pt;  # ADDED
  break-inside: avoid; }  # ADDED
th, td { border: 1px solid #b9c0ca; padding: 3.5px 7px; text-align: left; vertical-align: top; }  # ADDED
th { background: #e7eef7; color: #16365c; }  # ADDED
td:empty { height: 2.6em; min-width: 7em; }  # ADDED
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 8.9pt; background: #f0f2f5;  # ADDED
  padding: 1px 4px; border-radius: 3px; }  # ADDED
pre { background: #f6f8fa; border: 1px solid #dbe1e8; border-left: 3px solid #16365c;  # ADDED
  padding: 8px 12px; font-size: 8.3pt; line-height: 1.3; white-space: pre-wrap;  # ADDED
  break-inside: avoid; }  # ADDED
pre code { background: none; padding: 0; }  # ADDED
td code { white-space: pre-wrap; font-size: 8.1pt; }  # ADDED
blockquote { margin: 0.6em 0; padding: 6px 12px; background: #fff6e0;  # ADDED
  border-left: 4px solid #d89a00; }  # ADDED
img { max-width: 100%; display: block; margin: 0.7em auto; break-inside: avoid; }  # ADDED
p:has(+ pre), p:has(+ table), p:has(+ ul), p:has(+ ol) { break-after: avoid; }  # ADDED
.katex-display { margin: 0.5em 0; break-inside: avoid; }  # ADDED
"""  # ADDED

PAGE = """<!DOCTYPE html>  # ADDED
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>  # ADDED
<link rel="stylesheet" href="{katex}/katex.min.css" crossorigin="anonymous">  # ADDED
<style>{css}</style></head>  # ADDED
<body>  # ADDED
{body}  # ADDED
<script defer src="{katex}/katex.min.js" crossorigin="anonymous"></script>  # ADDED
<script defer src="{katex}/contrib/auto-render.min.js" crossorigin="anonymous"  # ADDED
  onload="renderMathInElement(document.body, {{delimiters: [  # ADDED
    {{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}],  # ADDED
    throwOnError: false, strict: 'ignore'}}); document.title += ' [ready]';"></script>  # ADDED
</body></html>  # ADDED
"""  # ADDED


def markdown_to_html(markdown_text: str, title: str) -> str:  # ADDED
    """Standalone HTML page (KaTeX maths, print CSS) from report Markdown."""  # ADDED
    return PAGE.format(  # ADDED
        title=html.escape(title),  # ADDED
        katex=KATEX,  # ADDED
        css=CSS,  # ADDED
        body=markdown_body(markdown_text),  # ADDED
    )  # ADDED


def markdown_body(markdown_text: str) -> str:  # ADDED
    """Convert to an HTML fragment, shielding code and maths from the Markdown parser."""  # ADDED
    import markdown  # ADDED

    shelf = _Shelf()  # ADDED
    text = _separate_lists(markdown_text)  # ADDED
    text = _CODE_BLOCK.sub(lambda m: "\n\n" + shelf.put(_code_block(m.group(0))) + "\n\n", text)  # ADDED
    text = _INLINE_CODE.sub(lambda m: shelf.put(f"<code>{html.escape(m.group(1))}</code>"), text)  # ADDED
    text = _DISPLAY_MATH.sub(lambda m: "\n\n" + shelf.put(_raw(m.group(0))) + "\n\n", text)  # ADDED
    text = _INLINE_MATH.sub(lambda m: shelf.put(_raw(m.group(0))), text)  # ADDED
    body = markdown.markdown(text, extensions=["tables", "sane_lists"], output_format="html5")  # ADDED
    body = shelf.restore(body)  # ADDED
    return re.sub(r"<p>\s*(<pre[\s\S]*?</pre>)\s*</p>", r"\1", body)  # ADDED


class _Shelf:  # ADDED
    """Stores fragments behind placeholder tokens that Markdown leaves untouched."""  # ADDED

    def __init__(self) -> None:  # ADDED
        self._items: list[str] = []  # ADDED

    def put(self, fragment: str) -> str:  # ADDED
        self._items.append(fragment)  # ADDED
        return f"ZZSHELF{len(self._items) - 1}ZZ"  # ADDED

    def restore(self, text: str) -> str:  # ADDED
        return re.sub(r"ZZSHELF(\d+)ZZ", lambda match: self._items[int(match.group(1))], text)  # ADDED


def _code_block(fenced: str) -> str:  # ADDED
    lines = fenced.split("\n")  # ADDED
    language = lines[0].strip()[3:].strip()  # ADDED
    content = "\n".join(lines[1:-1])  # ADDED
    css_class = f' class="language-{language}"' if language else ""  # ADDED
    return f"<pre><code{css_class}>{html.escape(content)}</code></pre>"  # ADDED


def _raw(math: str) -> str:  # ADDED
    return html.escape(math, quote=False)  # ADDED


def _separate_lists(text: str) -> str:  # ADDED
    """Python-Markdown needs a blank line before a list that follows a paragraph line."""  # ADDED
    out: list[str] = []  # ADDED
    in_fence = False  # ADDED
    for line in text.split("\n"):  # ADDED
        if line.lstrip().startswith("```"):  # ADDED
            in_fence = not in_fence  # ADDED
        stripped = line.lstrip()  # ADDED
        if not in_fence and _LIST_ITEM.match(stripped) and out:  # ADDED
            previous = out[-1]  # ADDED
            if (  # ADDED
                previous.strip()  # ADDED
                and not _LIST_ITEM.match(previous.lstrip())  # ADDED
                and not previous.startswith(" ")  # ADDED
            ):  # ADDED
                out.append("")  # ADDED
        out.append(line)  # ADDED
    return "\n".join(out)  # ADDED

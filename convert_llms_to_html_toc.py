#!/usr/bin/env python3
"""
convert_llms_to_html.py
Download https://context7.com/sveltejs/svelte/llms.txt
→ build a pretty HTML document that includes
 • syntax-highlighted code blocks
 • one-click “Copy” buttons
 • description paragraphs
 • a collapsible Table of Contents with anchor links
 • “Back to top” links under each snippet
"""

import html
import pathlib
import re
import sys
import textwrap
from urllib.request import urlopen

SRC_URL = "https://context7.com/sveltejs/svelte/llms.txt"
OUT_FILE = "svelte.html"


# ───────────────────────── helpers ────────────────────────────────────────────
def fetch_text(url: str) -> str:
    with urlopen(url) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_blocks(raw: str):
    """Yield dicts: title, description, source, language, code."""
    for block in re.split(r"-{20,}", raw):
        if not block.strip():
            continue
        d = {}
        for key in ("TITLE", "DESCRIPTION", "SOURCE", "LANGUAGE"):
            m = re.search(rf"{key}:\s*(.+)", block)
            if m:
                d[key.lower()] = m.group(1).strip()
        m_code = re.search(r"CODE:\s*```(.*?)```", block, re.S)
        d["code"] = m_code.group(1).strip("\n") if m_code else ""
        yield d


def slugify(txt: str) -> str:
    """Make a URL-safe id from a title."""
    slug = re.sub(r"[^\w\s-]", "", txt.lower())  # keep words/hyphens/space
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")  # collapse separators
    return slug or "untitled"


# ───────────────────────── html builder ───────────────────────────────────────
def html_doc(blocks):
    # pre-compute ids & TOC
    ids = []
    for b in blocks:
        base = slugify(b.get("title", "untitled"))
        # ensure uniqueness
        i, new_id = 1, base
        while new_id in ids:
            i += 1
            new_id = f"{base}-{i}"
        ids.append(new_id)
        b["_id"] = new_id  # store on block

    # <head> and global assets
    yield textwrap.dedent(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8"/>
          <title>Svelte LLMS snippets</title>

          <link rel="stylesheet"
            href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
          <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
          <script>hljs.highlightAll();</script>

          <style>
            body{font-family:system-ui,sans-serif;margin:2rem;line-height:1.55;}
            pre{padding:1rem;background:#f6f8fa;overflow:auto;position:relative;}
            section{margin-bottom:3rem;}
            button.copy-btn{
              position:absolute;top:.5rem;right:.5rem;font-size:.8rem;
              padding:.25rem .5rem;border:1px solid #888;border-radius:4px;
              background:#fff;cursor:pointer;
            }
            button.copy-btn.copied{background:#d1ffd8;border-color:#3a3;}
            a.source{font-size:.9rem;}
            nav#toc{margin-bottom:2.5rem;border-left:4px solid #999;padding-left:1rem;}
            nav#toc ul{margin:0;padding-left:0.5rem;list-style-type:none;}
            nav#toc li{margin:.25rem 0;}
            a.toc-link{text-decoration:none;}
            a.backtop{display:inline-block;margin-top:.75rem;font-size:.8rem;}
          </style>

          <script>
            function copyCode(id){
              const el=document.getElementById(id);
              navigator.clipboard.writeText(el.innerText).then(()=>{
                const btn=el.parentElement.querySelector('.copy-btn');
                if(btn){
                  btn.classList.add('copied');
                  btn.textContent='Copied!';
                  setTimeout(()=>{btn.classList.remove('copied');btn.textContent='Copy';},1500);
                }
              });
            }
          </script>
        </head>
        <body>
        <h1>Svelte LLMS snippets</h1>
        """
    )

    # ─── Table of Contents ──────────────────────────────────────────────
    yield '<details open id="toc"><summary><strong>Table of Contents</strong></summary>'
    yield "<ul>"
    for b in blocks:
        title = html.escape(b.get("title", "Untitled"))
        yield f'<li><a class="toc-link" href="#{b["_id"]}">{title}</a></li>'
    yield "</ul></details>"

    # ─── Snippet sections ───────────────────────────────────────────────
    for idx, b in enumerate(blocks, 1):
        code_id = f"code-{idx}"
        lang_cls = html.escape(b.get("language", "").split()[0].lower())

        yield f'<section id="{b["_id"]}">'
        yield f"<h2>{html.escape(b.get('title', 'Untitled'))}</h2>"

        if b.get("description"):
            yield f"<p>{html.escape(b['description'])}</p>"

        if b.get("code"):
            yield (
                f'<pre><button class="copy-btn" onclick="copyCode(\'{code_id}\')">Copy</button>'
                f'<code id="{code_id}" class="{lang_cls}">{html.escape(b["code"])}</code></pre>'
            )

        if b.get("source"):
            src = html.escape(b["source"])
            yield f'<p><a class="source" href="{src}">{src}</a></p>'

        # back-to-top link
        yield '<a class="backtop" href="#toc">↑ Back to top</a>'
        yield "</section>"

    yield "</body></html>"


# ───────────────────────── main ───────────────────────────────────────────────
def main(out: pathlib.Path):
    print(f"Downloading {SRC_URL} …")
    txt = fetch_text(SRC_URL)
    blocks = list(parse_blocks(txt))
    print(f"Parsed {len(blocks)} snippet{'s' if len(blocks)!=1 else ''}.")
    out.write_text("\n".join(html_doc(blocks)), encoding="utf-8")
    print(f"✓ Wrote {out.resolve()}")


if __name__ == "__main__":
    dst = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else OUT_FILE)
    main(dst)

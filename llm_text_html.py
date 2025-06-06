import re
from html import escape

def parse_structured_blocks(lines):
    sections = []
    current_section = None
    inside_code = False
    current_language = None
    current_code_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("TITLE:"):
            if current_section:
                sections.append(current_section)
            current_section = {
                'title': stripped[6:].strip(),
                'description': '',
                'source': '',
                'code_blocks': []
            }

        elif stripped.startswith("DESCRIPTION:") and current_section:
            current_section['description'] = stripped[12:].strip()

        elif stripped.startswith("SOURCE:") and current_section:
            current_section['source'] = stripped[7:].strip()

        elif stripped.startswith("LANGUAGE:") and current_section:
            current_language = stripped[9:].strip()

        elif stripped == "CODE:" and current_section:
            inside_code = 'awaiting_triple_backtick'

        elif inside_code == 'awaiting_triple_backtick' and stripped == "```":
            inside_code = True
            current_code_lines = []

        elif inside_code is True and stripped == "```":
            # End of actual code block
            current_section['code_blocks'].append({
                'language': current_language,
                'code': '\n'.join(current_code_lines)
            })
            inside_code = False

        elif inside_code is True:
            current_code_lines.append(line.rstrip())

    if current_section:
        sections.append(current_section)

    return sections


def generate_html(sections):
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Svelte 5 for LLMs</title>",
        "  <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css'>",
        "  <script src='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js'></script>",
        "  <script>hljs.highlightAll();</script>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }",
        "    h2 { color: #2c3e50; margin-bottom: 0.2em; }",
        "    p { margin-top: 0; font-size: 1em; color: #555; }",
        "    pre { background: #f4f4f4; padding: 1em; border-left: 4px solid #ccc; overflow-x: auto; position: relative; }",
        "    code { font-family: Consolas, monospace; }",
        "    a.source { font-size: 0.9em; color: #3498db; display: inline-block; margin-bottom: 1em; }",
        "    .section { margin-bottom: 3em; }",
        "    .copy-btn { position: absolute; top: 8px; right: 8px; background: #3498db; color: #fff; border: none; border-radius: 3px; padding: 0.3em 0.7em; cursor: pointer; font-size: 0.9em; }",
        "    .copy-btn:active { background: #217dbb; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Svelte 5 Code Snippets</h1>"
    ]

    for idx, section in enumerate(sections):
        html.append("<div class='section'>")
        html.append(f"  <h2>{section['title']}</h2>")
        html.append(f"  <p>{section['description']}</p>")
        html.append(f"  <a class='source' href='{section['source']}' target='_blank'>Source</a>")
        for block_idx, block in enumerate(section['code_blocks']):
            lang = block['language']
            code_html = escape(block['code'])
            code_id = f"code-{idx}-{block_idx}"
            html.append(f"  <pre><button class='copy-btn' onclick=\"copyToClipboard('{code_id}')\">Copy</button><code id='{code_id}' class='{lang}'>" + code_html + "</code></pre>")
        html.append("</div>")

    html.extend([
        "<script>",
        "function copyToClipboard(id) {",
        "  var code = document.getElementById(id).innerText;",
        "  navigator.clipboard.writeText(code).then(function() {",
        "    /* Optionally show feedback */",
        "  });",
        "}",
        "</script>",
        "</body>",
        "</html>"
    ])
    return '\n'.join(html)


def convert_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sections = parse_structured_blocks(lines)
    html = generate_html(sections)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ HTML saved to {output_path}")


# Run this
convert_file("llms.txt", "llms_structured.html")

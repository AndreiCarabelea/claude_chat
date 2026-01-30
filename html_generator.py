import time
import re

import logging
from html import escape


def _get_htmlgen_logger():
    """Logger that writes parser diagnostics to debug.log.

    Kept lightweight so it can run under Streamlit without requiring global
    logging configuration.
    """

    logger = logging.getLogger("html_generator")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("debug.log", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    # Prevent duplicate logging if root logger is also configured.
    logger.propagate = False
    return logger


def _strip_markdown_heading(line: str):
    """Return (level, text) if line is a markdown heading, else (None, line)."""
    m = re.match(r"^(#{1,6})\s+(.*)$", line)
    if not m:
        return None, line
    return len(m.group(1)), m.group(2).strip()


def _slugify_id(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s\-_.]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s or "section"


def _convert_inline_markdown(text: str) -> str:
    """Convert a small, safe subset of markdown to HTML.

    Note: we intentionally keep LaTeX markers ($, \\, {, }) intact for MathJax.
    """
    # Escape HTML first.
    text = escape(text, quote=False)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic (avoid clobbering underscores in math; keep it conservative)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    return text


def _iter_blocks(content: str):
    """Yield blocks from content.

    Blocks are dicts with type:
      - heading: {level, raw, text}
      - hr: {}
      - math: {raw}
      - ul: {items: [str]}
      - p: {text: str}

    This prevents splitting display-math and matrix environments across <p> tags.
    """
    lines = content.split("\n")
    i = 0
    in_display_math = False
    math_lines = []
    in_env = False
    env_lines = []
    env_end_re = None
    list_items = []

    def flush_list():
        nonlocal list_items
        if list_items:
            yield {"type": "ul", "items": list_items}
            list_items = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip("\r")
        stripped = line.strip()
        i += 1

        if not stripped:
            # paragraph/list separation
            yield from flush_list()
            continue

        # Horizontal rule
        if stripped in ("---", "***"):
            yield from flush_list()
            yield {"type": "hr"}
            continue

        # Display math blocks with $$ ... $$ possibly multi-line.
        # Start (and maybe end) on this line.
        if not in_display_math and stripped.startswith("$$"):
            yield from flush_list()
            in_display_math = True
            math_lines = [line]
            # Single-line $$...$$
            if stripped.count("$$") >= 2 and stripped.endswith("$$") and len(stripped) > 4:
                in_display_math = False
                yield {"type": "math", "raw": "\n".join(math_lines)}
            continue
        if in_display_math:
            math_lines.append(line)
            if "$$" in stripped and stripped.endswith("$$") and len(math_lines) > 1:
                in_display_math = False
                yield {"type": "math", "raw": "\n".join(math_lines)}
            continue

        # LaTeX environment blocks even when $$ is not used.
        if not in_env and re.search(r"\\begin\{([a-zA-Z*]+)\}", stripped):
            yield from flush_list()
            m = re.search(r"\\begin\{([a-zA-Z*]+)\}", stripped)
            env_name = m.group(1)
            env_end_re = re.compile(rf"\\\\end\{{{re.escape(env_name)}\}}")
            in_env = True
            env_lines = [line]
            # If begin and end on same line
            if env_end_re.search(stripped):
                in_env = False
                yield {"type": "math", "raw": "\n".join(env_lines)}
            continue
        if in_env:
            env_lines.append(line)
            if env_end_re and env_end_re.search(stripped):
                in_env = False
                yield {"type": "math", "raw": "\n".join(env_lines)}
            continue

        # Markdown headings
        level, heading_text = _strip_markdown_heading(stripped)
        if level is not None:
            yield from flush_list()
            yield {"type": "heading", "level": level, "raw": stripped, "text": heading_text}
            continue

        # Bulleted lists
        if re.match(r"^[-*]\s+", stripped):
            item = re.sub(r"^[-*]\s+", "", stripped, count=1)
            list_items.append(item)
            continue

        # Regular paragraph
        yield from flush_list()
        yield {"type": "p", "text": stripped}

    # flush any trailing state
    if in_display_math and math_lines:
        yield {"type": "math", "raw": "\n".join(math_lines)}
    if in_env and env_lines:
        yield {"type": "math", "raw": "\n".join(env_lines)}
    if list_items:
        yield {"type": "ul", "items": list_items}


def _parse_chapter_heading(text: str):
    """Return (chapter_num, title) if text is a chapter heading, else None."""
    m = re.match(r"^(?:Chapter\s+)(\d+)\s*[\.:]?\s*(.*)$", text.strip())
    if not m:
        return None
    title = (m.group(2) or "").strip() or "Untitled"
    return m.group(1), title


def _parse_section_heading(text: str):
    """Return (section_num, title) if text is a section heading, else None."""
    m = re.match(r"^(?:Section\s+)(\d+\.\d+)\s*[\.:]?\s*(.*)$", text.strip())
    if not m:
        return None
    title = (m.group(2) or "").strip() or "Untitled"
    return m.group(1), title

def save_to_html(system_response):
    """
    Save the system response to a simple HTML file with a working table of contents.
    """
   
    
    logger = _get_htmlgen_logger()
    logger.info("save_to_html(): start; input chars=%d", len(system_response or ""))

    # Remove AI introduction phrases
    ai_phrases = [
        r"I'll help organize.*\n",
        r"I'll help structure.*\n",
        r"Let me structure.*\n",
        r"Let me help.*\n",
        r"I'll assist.*\n"
    ]
    
    content = system_response
    for phrase in ai_phrases:
        content = re.sub(phrase, '', content)

    logger.info("save_to_html(): after intro-strip; chars=%d", len(content))
    
    # Extract chapters and sections
    chapters = []
    current_chapter = None
    current_sections = []
    
    for i, line in enumerate(content.split('\n'), start=1):
        raw_line = line
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are not chapters or sections
        if line == "Additional Notes:" or line == "Additional Notes" or "Analysis and Corrections" in line:
            continue
            
        # Normalize markdown heading prefixes (## / ### / etc.)
        heading_level, heading_text = _strip_markdown_heading(line)
        parse_text = heading_text if heading_level is not None else line

        # Match chapter/section patterns.
        # We only accept numeric-only headings when they are explicitly Markdown headings
        # (to avoid misclassifying normal enumerations inside paragraphs).
        chapter_tuple = _parse_chapter_heading(parse_text)
        if not chapter_tuple and heading_level is not None and heading_level <= 2:
            m = re.match(r'^(\d+)[\.:]\s+(.+)$', parse_text)
            if m:
                chapter_tuple = (m.group(1), m.group(2).strip())

        section_tuple = _parse_section_heading(parse_text)
        if not section_tuple and heading_level is not None and heading_level <= 3:
            m = re.match(r'^(\d+\.\d+)[\.:]\s+(.+)$', parse_text)
            if m:
                section_tuple = (m.group(1), m.group(2).strip())
        
        # Remove any remaining ** markers for matching
        # Remove any remaining ** markers for matching
        clean_line = re.sub(r'\*\*', '', parse_text)

        if chapter_tuple:
            logger.info(
                "chapter_match@line=%d: %r => num=%s title=%r",
                i,
                clean_line,
                chapter_tuple[0],
                chapter_tuple[1],
            )
        elif section_tuple:
            logger.info(
                "section_match@line=%d: %r => num=%s title=%r",
                i,
                clean_line,
                section_tuple[0],
                section_tuple[1],
            )
        elif heading_level is not None:
            logger.info("heading_line@line=%d: %r", i, clean_line)
        elif re.match(r"^\d+\.\s+", clean_line):
            # Numeric bullets often get misclassified as chapters.
            logger.info("numeric_bullet@line=%d: %r", i, clean_line)
        
        if chapter_tuple:
            if current_chapter:
                chapters.append((current_chapter, current_sections))
            current_chapter = (chapter_tuple[0], chapter_tuple[1])
            current_sections = []
        elif section_tuple:
            section_num = section_tuple[0]
            section_title = section_tuple[1]
            # Clean up the title (remove any remaining formatting)
            section_title = re.sub(r'\*\*', '', section_title)
            current_sections.append((section_num, section_title))
        # Note: additional ad-hoc patterns removed; markdown heading normalization
        # above should catch the majority of structured outputs.
    
    # Add the last chapter
    if current_chapter:
        chapters.append((current_chapter, current_sections))

    logger.info("save_to_html(): parsed chapters=%d (pre-filter)", len(chapters))
    
    # Filter out any non-numeric entries that might have been incorrectly captured
    filtered_chapters = []
    for chapter, sections in chapters:
        try:
            # Ensure chapter number is numeric
            int(chapter[0])
            filtered_sections = []
            for section in sections:
                # Ensure section number starts with the chapter number
                if section[0].startswith(f"{chapter[0]}."):
                    filtered_sections.append(section)
            filtered_chapters.append((chapter, filtered_sections))
        except ValueError:
            # Skip non-numeric chapter numbers
            continue
    
    chapters = filtered_chapters

    logger.info("save_to_html(): filtered chapters=%d", len(chapters))
    for ch, secs in chapters:
        logger.info("toc_chapter: %s %r; sections=%d", ch[0], ch[1], len(secs))
    
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '    <title>Lecture Analysis</title>',
        '    <script>',
        '      window.MathJax = {',
        '        tex: {',
        "          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],",
        "          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],",
        '          processEscapes: true,',
        '        },',
        '        svg: { fontCache: "global" }',
        '      };',
        '    </script>',
        '    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>',
        '    <style>',
        '        :root {',
        '            --bg: #f5f5f5;',
        '            --card: #ffffff;',
        '            --text: #1a202c;',
        '            --muted: #4a5568;',
        '            --accent: #2c5282;',
        '            --border: #e2e8f0;',
        '        }',
        '        body {',
        '            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";',
        '            line-height: 1.6;',
        '            max-width: 960px;',
        '            margin: 0 auto;',
        '            padding: 20px;',
        '            color: var(--text);',
        '            background-color: var(--bg);',
        '        }',
        '        .container {',
        '            background-color: var(--card);',
        '            padding: 28px;',
        '            border-radius: 8px;',
        '            box-shadow: 0 2px 4px rgba(0,0,0,0.1);',
        '        }',
        '        .meta {',
        '            color: var(--muted);',
        '            font-size: 0.95em;',
        '            margin-bottom: 10px;',
        '        }',
        '        .section-title {',
        '            color: var(--accent);',
        '            font-weight: 700;',
        '            margin-top: 18px;',
        '        }',
        '        .toc {',
        '            background-color: #f8f9fa;',
        '            padding: 20px;',
        '            border-radius: 5px;',
        '            margin-bottom: 30px;',
        '            border: 1px solid var(--border);',
        '        }',
        '        .toc a {',
        '            text-decoration: none;',
        '            color: var(--accent);',
        '        }',
        '        .toc a:hover {',
        '            text-decoration: underline;',
        '        }',
        '        .toc-chapter {',
        '            font-weight: 700;',
        '            margin-top: 10px;',
        '        }',
        '        .toc-section {',
        '            margin-left: 20px;',
        '        }',
        '        .chapter-title {',
        '            color: var(--accent);',
        '            font-size: 1.5em;',
        '            font-weight: 800;',
        '            margin-top: 30px;',
        '            padding-top: 10px;',
        '            border-top: 1px solid var(--border);',
        '        }',
        '        .content p {',
        '            margin: 10px 0;',
        '        }',
        '        .content ul {',
        '            margin: 10px 0 10px 22px;',
        '        }',
        '        .content hr {',
        '            border: none;',
        '            border-top: 1px solid var(--border);',
        '            margin: 18px 0;',
        '        }',
        '        .math-block {',
        '            padding: 12px 14px;',
        '            border: 1px solid var(--border);',
        '            border-radius: 6px;',
        '            background: #fbfdff;',
        '            overflow-x: auto;',
        '            margin: 12px 0;',
        '        }',
        '        html { scroll-behavior: smooth; }',
        '    </style>',
        '</head>',
        '<body>',
        '    <div class="container">',
        f'        <div class="meta">Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</div>',
        '        <div class="toc">',
        '            <h2>Table of Contents</h2>'
    ]
    
    # Generate TOC
    seen_ids = set()
    for chapter, sections in chapters:
        chapter_id = f'chapter-{chapter[0]}'
        if chapter_id in seen_ids:
            logger.warning("duplicate_anchor_id: %s (chapter)", chapter_id)
        seen_ids.add(chapter_id)
        html_parts.append(
            f'            <div class="toc-chapter"><a href="#{chapter_id}">{_convert_inline_markdown(f"{chapter[0]}. {chapter[1]}")}</a></div>'
        )
        for section in sections:
            section_id = f'section-{section[0].replace(".", "-")}'
            if section_id in seen_ids:
                logger.warning("duplicate_anchor_id: %s (section)", section_id)
            seen_ids.add(section_id)
            html_parts.append(
                f'            <div class="toc-section"><a href="#{section_id}">{_convert_inline_markdown(f"{section[0]} {section[1]}")}</a></div>'
            )
    
    html_parts.append('        </div>')
    html_parts.append('        <div class="content">')
    
    # Convert content to HTML with anchors for TOC
    for block in _iter_blocks(content):
        btype = block["type"]
        if btype == "heading":
            heading_text = block["text"]
            clean_heading = re.sub(r'\*\*', '', heading_text)
            # Determine whether this heading is a Chapter or Section.
            ch = _parse_chapter_heading(clean_heading)
            sec = _parse_section_heading(clean_heading)
            if ch:
                ch_num, ch_title = ch
                chapter_id = f"chapter-{ch_num}"
                html_parts.append(
                    f'        <h2 id="{chapter_id}" class="chapter-title">{_convert_inline_markdown(f"{ch_num}. {ch_title}")}</h2>'
                )
            elif sec:
                sec_num, sec_title = sec
                section_id = f'section-{sec_num.replace(".", "-")}'
                html_parts.append(
                    f'        <h3 id="{section_id}" class="section-title">{_convert_inline_markdown(f"{sec_num} {sec_title}")}</h3>'
                )
            else:
                # Generic headings (e.g., title)
                level = max(1, min(6, int(block["level"])))
                hid = _slugify_id(clean_heading)
                html_parts.append(
                    f'        <h{level} id="h-{hid}">{_convert_inline_markdown(heading_text)}</h{level}>'
                )
        elif btype == "hr":
            html_parts.append('        <hr />')
        elif btype == "ul":
            html_parts.append('        <ul>')
            for item in block["items"]:
                html_parts.append(f'          <li>{_convert_inline_markdown(item)}</li>')
            html_parts.append('        </ul>')
        elif btype == "math":
            # Keep raw content so MathJax can parse it.
            # But DO escape HTML-sensitive characters so things like '&' in matrices
            # don't get interpreted as HTML entities.
            html_parts.append('        <div class="math-block">')
            html_parts.append(escape(block["raw"], quote=False))
            html_parts.append('        </div>')
        elif btype == "p":
            # Also promote plain-text chapter/section lines to headings (some model
            # outputs don't use markdown headings).
            ptext = block["text"]
            clean_p = re.sub(r'\*\*', '', ptext)
            ch = _parse_chapter_heading(clean_p)
            sec = _parse_section_heading(clean_p)
            if ch:
                ch_num, ch_title = ch
                chapter_id = f"chapter-{ch_num}"
                html_parts.append(
                    f'        <h2 id="{chapter_id}" class="chapter-title">{_convert_inline_markdown(f"{ch_num}. {ch_title}")}</h2>'
                )
            elif sec:
                sec_num, sec_title = sec
                section_id = f'section-{sec_num.replace(".", "-")}'
                html_parts.append(
                    f'        <h3 id="{section_id}" class="section-title">{_convert_inline_markdown(f"{sec_num} {sec_title}")}</h3>'
                )
            else:
                html_parts.append(f'        <p>{_convert_inline_markdown(ptext)}</p>')
        else:
            logger.warning("unknown_block_type: %r", btype)
    
    html_parts.extend([
        '        </div>',
        '    </div>',
        '</body>',
        '</html>'
    ])
    
    # Write to file
    with open('audio_demo.html', 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    print("Analysis saved to audio_demo.html")

    return 'audio_demo.html'

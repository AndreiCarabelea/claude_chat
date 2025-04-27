import time
import re

def save_to_html(system_response):
    """
    Save the system response to a simple HTML file with a working table of contents.
    """
   
    
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
    
    # Extract chapters and sections
    chapters = []
    current_chapter = None
    current_sections = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are not chapters or sections
        if line == "Additional Notes:" or line == "Additional Notes" or "Analysis and Corrections" in line:
            continue
            
        # Match chapter patterns:
        # - "Chapter 3: Title"
        # - "3. Title"
        # - "Chapter 3. Title"
        chapter_match = re.match(r'(?:Chapter\s+)?(\d+)[\.:]\s+(.+)', line)
        
        # Match section patterns:
        # - "Section 3.2: Title"
        # - "3.2 Title"
        # - "Section 3.2. Title"
        section_match = re.match(r'(?:Section\s+)?(\d+\.\d+)[\.:]\s+(.+)', line)
        
        # Remove any remaining ** markers for matching
        clean_line = re.sub(r'\*\*', '', line)
        
        if chapter_match:
            if current_chapter:
                chapters.append((current_chapter, current_sections))
            current_chapter = (chapter_match.group(1), chapter_match.group(2))
            current_sections = []
        elif section_match:
            section_num = section_match.group(1)
            section_title = section_match.group(2)
            # Clean up the title (remove any remaining formatting)
            section_title = re.sub(r'\*\*', '', section_title)
            current_sections.append((section_num, section_title))
        elif clean_line.startswith('Chapter'):  # Catch any other chapter format
            match = re.match(r'Chapter\s+(\d+)[:\.]?\s+(.+)', clean_line)
            if match:
                if current_chapter:
                    chapters.append((current_chapter, current_sections))
                current_chapter = (match.group(1), match.group(2))
                current_sections = []
        elif re.match(r'Section\s+\d+\.\d+', clean_line):  # Catch any other section format
            match = re.match(r'Section\s+(\d+\.\d+)[:\.]?\s+(.+)', clean_line)
            if match:
                current_sections.append((match.group(1), match.group(2)))
    
    # Add the last chapter
    if current_chapter:
        chapters.append((current_chapter, current_sections))
    
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
    
    html_parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '    <title>Lecture Analysis</title>',
        '    <style>',
        '        body {',
        '            font-family: Arial, sans-serif;',
        '            line-height: 1.6;',
        '            max-width: 800px;',
        '            margin: 0 auto;',
        '            padding: 20px;',
        '            background-color: #f5f5f5;',
        '        }',
        '        .container {',
        '            background-color: white;',
        '            padding: 20px;',
        '            border-radius: 8px;',
        '            box-shadow: 0 2px 4px rgba(0,0,0,0.1);',
        '        }',
        '        .section-title {',
        '            color: #2c5282;',
        '            font-weight: bold;',
        '        }',
        '        .toc {',
        '            background-color: #f8f9fa;',
        '            padding: 20px;',
        '            border-radius: 5px;',
        '            margin-bottom: 30px;',
        '        }',
        '        .toc a {',
        '            text-decoration: none;',
        '            color: #2c5282;',
        '        }',
        '        .toc a:hover {',
        '            text-decoration: underline;',
        '        }',
        '        .toc-chapter {',
        '            font-weight: bold;',
        '            margin-top: 10px;',
        '        }',
        '        .toc-section {',
        '            margin-left: 20px;',
        '        }',
        '        .chapter-title {',
        '            color: #2c5282;',
        '            font-size: 1.5em;',
        '            font-weight: bold;',
        '            margin-top: 30px;',
        '        }',
        '    </style>',
        '</head>',
        '<body>',
        '    <div class="container">',
        f'        <div>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</div>',
        '        <div class="toc">',
        '            <h2>Table of Contents</h2>'
    ]
    
    # Generate TOC
    for chapter, sections in chapters:
        chapter_id = f'chapter-{chapter[0]}'
        html_parts.append(f'            <div class="toc-chapter"><a href="#{chapter_id}">{chapter[0]}. {chapter[1]}</a></div>')
        for section in sections:
            section_id = f'section-{section[0].replace(".", "-")}'
            html_parts.append(f'            <div class="toc-section"><a href="#{section_id}">{section[0]} {section[1]}</a></div>')
    
    html_parts.append('        </div>')
    html_parts.append('        <div class="content">')
    
    # Convert content to HTML with anchors for TOC
    for line in content.split('\n'):
        if line.strip():
            # Clean line for matching
            clean_line = re.sub(r'\*\*', '', line.strip())
            
            # Match chapter and section patterns
            chapter_match = re.match(r'(?:Chapter\s+)?(\d+)[\.:]\s+(.+)', clean_line)
            section_match = re.match(r'(?:Section\s+)?(\d+\.\d+)[\.:]\s+(.+)', clean_line)
            
            if chapter_match:
                chapter_id = f'chapter-{chapter_match.group(1)}'
                html_parts.append(f'        <h2 id="{chapter_id}" class="chapter-title">{line}</h2>')
            elif section_match:
                section_id = f'section-{section_match.group(1).replace(".", "-")}'
                html_parts.append(f'        <h3 id="{section_id}" class="section-title">{line}</h3>')
            else:
                # Convert markdown bold to HTML bold
                line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                html_parts.append(f'        <p>{line}</p>')
    
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
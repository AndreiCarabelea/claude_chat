from anthropic import Anthropic
from pypdf import PdfReader
from functools import lru_cache
from math import ceil, floor
import time
import random
from record import record_audio
import whisper_timestamped as whisper
from langdetect import detect
import re 
from prompt import get_long_prompt
import os
import json
import hashlib
import re

#second change

client = Anthropic()
MODEL_NAME = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"][2]

book_name= "2008-rfs.pdf"
reader = PdfReader(book_name)
number_of_pages = len(reader.pages)

# Replace the current pdf_text extraction with page-based extraction
pages_text = [page.extract_text() for page in reader.pages]
pdf_text = ''.join(pages_text)

number_of_chunks = 10
chunk_size = len(pdf_text)//number_of_chunks
message_history = []


def extract_number(text):
    """
    Extracts a single integer number from the given text.
    
    Parameters:
    text (str): The input text containing a single integer number.
    
    Returns:
    int: The extracted integer number.
    None: If no integer number is found in the text.
    """
    # Use regular expression to find the first occurrence of an integer
    match = re.search(r'\b\d+\b', text)
    
    if match:
        # Convert the matched string to an integer and return it
        return int(match.group())
    else:
        # Return None if no integer is found
        return None
    

def getLanguage(phrase):
    
    isoISO6391toAWSLanguageCode = ["en", "fr", "es", "it", "pt", "de"]
    
    try:
        ret_language = detect(phrase)
    except Exception as exc:
        print(repr(exc))
        return isoISO6391toAWSLanguageCode[0]
    
    if ret_language not in isoISO6391toAWSLanguageCode:
        return isoISO6391toAWSLanguageCode[0]
        
    return  ret_language



def chunk_getter(page_number, chunk_width):
    """Get text chunks centered around a specific page"""
    # Convert to 0-based index
    page_idx = int(page_number) - 1
    
    # Calculate the range of pages to include
    start_page = max(0, page_idx - chunk_width//2)
    end_page = min(number_of_pages - 1, page_idx + chunk_width//2)
    
    # Get text from the range of pages
    return ' '.join(pages_text[start_page:end_page + 1])

def get_completion_pdf(client, simple_prompt, page_number: int, chunk_width):
    global book_name, message_history
    print(f"Analyzing pages around page {page_number} with width {chunk_width}")
    
    text_chunk = chunk_getter(page_number, chunk_width)
            
    prompt = ("Here is an extract from an academic book:\n\n"
             f"<book>{text_chunk}</book>\n\n"
             f"Use this context to answer the following question: {simple_prompt}\n"
             "Respond with None only if the excerpt does not contain detailed information about the question. "
             "Otherwise also indicate section and chapter of the book.")
    
    message_history.append({"role": 'user', "content": prompt})
    
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        temperature=0.75,
        system="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts.",
        messages=message_history
    ).content[0].text
    

def find_section_and_respond(client, simple_prompt, page_number: int, chunk_width):
    global message_history
    max_width = min(32, number_of_pages)  # Cap the maximum width
    attempts = 0
    max_attempts = 5  # Prevent infinite loops
    
    while chunk_width <= max_width and attempts < max_attempts:
        time.sleep(5)
        delta = random.uniform(-1, 1) * 0.1 * page_number
        
        print(f"Trying with chunk width: {chunk_width}")
        x_response = get_completion_pdf(client, simple_prompt, page_number + delta, chunk_width)
        
        if "None" not in x_response and len(x_response) > 300:
            print(f"Found relevant content with width: {chunk_width}")
            message_history.append({"role": 'assistant', "content": x_response})
            return x_response
        
        # Geometric growth (multiply by 2 each time)
        chunk_width = chunk_width * 2
        attempts += 1
    
    # If we get here, we couldn't find a good response
    return "Could not find relevant information in the specified section of the document."

def  get_system_response(client, simple_prompt):
    global message_history
    message_history.append({ "role": 'user', "content":  simple_prompt})
    
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        temperature = 0.75,
        system="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts.",
        messages = message_history,
        
    ).content[0].text
    
def save_to_html(system_response):
    """
    Save the system response to a simple HTML file with a working table of contents.
    """
    import time
    import re
    
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


def get_audio_hash(audio_file_path):
    """Generate a hash of the audio file for caching purposes"""
    with open(audio_file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def transcribe_audio(audio_file_path, model_type, language):
    """Transcribe audio with caching"""
    # Skip caching for recording.wav since it changes frequently
    if os.path.basename(audio_file_path) == "recording.wav":
        audio = whisper.load_audio(audio_file_path)
        whisper_model = whisper.load_model(model_type, device="cpu")
        return whisper.transcribe(whisper_model, audio, language=language)
    print(audio_file_path)
    # For all other files, use caching
    file_hash = get_audio_hash(audio_file_path)
    cache_dir = os.path.join(os.getcwd(), 'transcription_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{file_hash}_{language}.json")
    
    # Check if we have a cached version
    if os.path.exists(cache_file):
        print(f"Loading cached transcription for {os.path.basename(audio_file_path)}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # If not cached, perform the transcription
    print(f"Transcribing {os.path.basename(audio_file_path)}...")
    audio = whisper.load_audio(audio_file_path)
    
    # Ensure model is loaded
    whisper_model = whisper.load_model(model_type, device="cpu")
    
    # Perform transcription
    result = whisper.transcribe(whisper_model, audio, language=language)
    
    # Cache the result
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

if __name__ == "__main__":      
    
    while True:
        enter_mode = input("[0 (text/text);  1 (audio/text); 2 (text-pdf/analyzer); 3 (audio-pdf/analyzer); 4 (audio/analyzer);]")  
        whisper_model = None
        
        try:
            enter_mode = int(enter_mode) 
        except:
            print(enter_mode)
            continue
        
        if enter_mode == 1 or enter_mode == 3 or enter_mode == 4:
            # language_text = input("Put some text in your language")
            # native_language = getLanguage(language_text)
            native_language = "en"
             
        while True:
            
            #todo ask questions about response 
            if enter_mode == 2:
                simple_prompt = input("Enter your question from pdf")
                
                try:
                    page_number = float(input("Enter page number from pdf"))
                    
                    if page_number > 0 and page_number <= number_of_pages:
                        try:
                            explanation = find_section_and_respond(client, simple_prompt, page_number, 2)
                            print(explanation)
                        except Exception as e:  # Specify the exception type
                            print(f"Error in find_section_and_respond: {e}")
                            print("Sleep for 1 minute")
                            time.sleep(10)
                    else:
                        print(f"Page number must be between 1 and {number_of_pages}")
                except ValueError:
                    print("Invalid page number")
                    system_response = get_system_response(client, simple_prompt)
                    message_history.append({ 'role': 'assistant', 'content':  system_response})
                    print(system_response)
                
            elif enter_mode == 0:
                simple_prompt = input("Your user text")
                system_response = get_system_response(client, simple_prompt)
                message_history.append({ 'role': 'assistant', 'content':  system_response})
                print(system_response)
            elif enter_mode == 1:
                input("Press Enter when you want to start recording...")
                record_audio("recording.wav")
                
                supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                whisper_model_type = supported_models[0]
                
                # Use cached transcription function
                result = transcribe_audio("recording.wav", whisper_model_type, native_language)
                simple_prompt = result["text"]
                print(f"You said: {simple_prompt} {len(simple_prompt)}")
                system_response = get_system_response(client, simple_prompt)
                message_history.append({ 'role': 'assistant', 'content':  system_response})
                print(system_response)
                
            elif enter_mode == 3:
                print("Enter your question from pdf")
                input("Press when you want to start recording")
                record_audio("recording.wav")
                audio = whisper.load_audio("recording.wav")
                supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                whisper_model_type = supported_models[0]
                if whisper_model is None:
                    whisper_model = whisper.load_model(whisper_model_type, device="cpu") 
                result = whisper.transcribe(whisper_model, audio, language=native_language)
                simple_prompt = result["text"]
                
                while True:
                    page_number = None
                    print("Say: the information is on page ... ")
                    input("Press when you want to start recording")
                    record_audio("recording.wav")
                    audio = whisper.load_audio("recording.wav")
                    result = whisper.transcribe(whisper_model, audio, language=native_language)
                    
                    try:
                        page_number = float(extract_number(result["text"]))
                    except:
                        print(page_number)
                        continue
                    
                    
                    if page_number > number_of_pages:
                        print(page_number)
                        continue
                    else:
                        break
                    

                print(f"{simple_prompt} - {page_number}")
                junk_text = input("text/page ok ?")
                if "y" not in junk_text:
                    continue
                
                try:
                    explanation = find_section_and_respond(client, simple_prompt, page_number, 2)
                    print(explanation)
                except:
                    print("Sleep for 1 minute")
                    time.sleep(60)
                    
            elif enter_mode == 4:
                filename = "lecture.wav"
                
                supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                whisper_model_type = supported_models[0]
                
                # Use cached transcription function 
                result = transcribe_audio(filename, whisper_model_type, native_language)
                lecture_text = result["text"]
                
                lp = get_long_prompt(lecture_text)
                system_response = get_system_response(client, lp)
                message_history.append({ 'role': 'assistant', 'content':  system_response})
                print(system_response)
                
                save_to_html(system_response)
                
            break
               
                             
            
            
    
        
       





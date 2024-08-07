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

#second change

client = Anthropic()
MODEL_NAME = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"][0]

book_name= "A-First-Course-in-Probability-8th-ed.-Sheldon-Ross.pdf"
reader = PdfReader(book_name)
number_of_pages = len(reader.pages)

    
pdf_text = ''.join(page.extract_text() for page in reader.pages)
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



def chunk_getter(index, book_name):    
    return pdf_text[index  * chunk_size : index  * chunk_size + chunk_size]
    
    
    
def get_completion_pdf(client, simple_prompt, page_number: int, chunk_width):
    global book_name, message_history
    print((page_number, chunk_width))
    
    #0 based
    chunk_index = floor((page_number * number_of_chunks/number_of_pages))
    
    text_chunk = ""
    
    if chunk_width == 1:
        text_chunk = chunk_getter(chunk_index, book_name)
    else:
        n_chunks_back = (chunk_width - 1)//2
        for ci in range(chunk_index - n_chunks_back, chunk_index + n_chunks_back + 1):
            text_chunk = text_chunk + chunk_getter(ci, book_name)
            
    prompt = f"here is an extract from an academic book <book> {text_chunk} </book>. \
                Use this context to answer to the following question {simple_prompt}. Respond with None only, if the excerpt does not contain detailed information about the question. Otherwise indicate section and chapter of the book."
    
    message_history.append({"role": 'user', "content":  prompt})
    
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        temperature = 0.75,
        system="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts.",
        messages = message_history
    ).content[0].text
    

def find_section_and_respond(client, simple_prompt, page_number: int, chunk_width):
    global message_history
    while True:
        time.sleep(60)
        delta = random.uniform(-1, 1) * 0.1 * page_number
        
        x_response = get_completion_pdf(client, simple_prompt, page_number + delta, chunk_width)
        if "None" not in x_response and len(x_response) > 300:
            print(len(x_response))
            message_history.append({ "role": 'assistant', "content":  x_response})
            return x_response
        else:
            chunk_width+=2
        
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
    

if __name__ == "__main__":      
    
    while True:
        enter_mode = input("[0 (text/text);  1 (audio/text); 2 (text-pdf/analyzer); 3 (audio-pdf/analyzer); 4 (audio/analyzer);]")  
        whisper_model = None
        
        try:
            enter_mode = int(enter_mode) 
        except:
            print(enter_mode)
            continue
        
        if enter_mode == 1 or enter_mode == 3:
            # language_text = input("Put some text in your language")
            # native_language = getLanguage(language_text)
            native_language = "en"
             
        while True:
            
             
            if enter_mode == 2:
                simple_prompt = input("Enter your question from pdf")
                page_number = float(input("Enter page number from pdf"))
                
                try:
                    explanation = find_section_and_respond(client, simple_prompt, page_number, 1)
                    print(explanation)
                except:
                    print("Sleep for 1 minute")
                    time.sleep(60)
            elif enter_mode == 0:
                simple_prompt = input("Your user text")
                system_response = get_system_response(client, simple_prompt)
                message_history.append({ 'role': 'assistant', 'content':  simple_prompt})
                print(system_response)
            elif enter_mode == 1:
                record_audio("recording.wav")
                audio = whisper.load_audio("recording.wav")
                
                supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                whisper_model_type = supported_models[0]
                
                if whisper_model is None:
                    whisper_model = whisper.load_model(whisper_model_type, device="cpu") 
                
                result = whisper.transcribe(whisper_model, audio, language=native_language)
                simple_prompt = result["text"]
                print(f"You said {simple_prompt} {len(simple_prompt)}")
                system_response = get_system_response(client, simple_prompt)
                message_history.append({ 'role': 'assistant', 'content':  simple_prompt})
                print(system_response)
                
            elif enter_mode == 3:
                print("Enter your question from pdf")
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
                    record_audio("recording.wav")
                    audio = whisper.load_audio("recording.wav")
                    result = whisper.transcribe(whisper_model, audio, language=native_language)
                    
                    try:
                        page_number = float(extract_number(result["text"]))
                    except:
                        print(page_number)
                        break
                    
                    
                    if page_number > 1.1 * number_of_pages:
                        print(page_number)
                        continue
                    else:
                        break
                    
                
                
                
                print(f"{simple_prompt} - {page_number}")
                junk_text = input("text/page ok ?")
                if "y" not in junk_text:
                    continue
                
                
                if page_number  is None:
                    system_response = get_system_response(client, simple_prompt)
                    message_history.append({ 'role': 'assistant', 'content':  simple_prompt})
                    print(system_response)
                    continue
                
                try:
                    explanation = find_section_and_respond(client, simple_prompt, page_number, 1)
                    print(explanation)
                except:
                    print("Sleep for 1 minute")
                    time.sleep(60)
                    
            elif enter_mode > 4:
                break
            
    
        
       





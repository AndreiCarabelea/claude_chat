import streamlit as st
from anthropic import Anthropic
from pypdf import PdfReader
from functools import lru_cache
from math import ceil, floor
import time
import random
from record import AudioRecorder
from html_generator import save_to_html
import whisper_timestamped as whisper
from langdetect import detect
import re 
from prompt import get_long_prompt
import os
import sys
import json
import hashlib
import tempfile
import keyboard
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
grok_api_key = os.getenv("XAI_API_KEY") 

#second change

#https://docs.anthropic.com/en/docs/about-claude/models/overview

# Define model options
MODEL_OPTIONS = {
    "Claude 4.5 Haiku": "claude-haiku-4-5-20251001",
    "Claude 4.5 Sonnet": "claude-sonnet-4-5-20250929", 
    "Cluade Opus 4.1": "claude-opus-4-1-20250805"
}

GROK_MODELS = {
    "Grok 3 mini": "grok-3-mini", 
    "grok 4 fast-reasoning": "grok-4-fast-reasoning", 
    "grok 4 fast-non-reasoning": "grok-4-fast-non-reasoning"
    
}
# Initialize MODEL_NAME in session state if not already present
if 'model_name' not in st.session_state:
    st.session_state.model_name = MODEL_OPTIONS["Claude 4.5 Haiku"]  # Default model

# Initialize session state variables
if 'anthropic_api_key' not in st.session_state:
    st.session_state.anthropic_api_key = anthropic_api_key
if 'xai_api_key' not in st.session_state:
    st.session_state.xai_api_key = grok_api_key
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'pages_text' not in st.session_state:
    st.session_state.pages_text = []
if 'number_of_pages' not in st.session_state:
    st.session_state.number_of_pages = 0
if 'book_name' not in st.session_state:
    st.session_state.book_name = ""
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "stopped"
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'audio_chat_transcription' not in st.session_state:
    st.session_state.audio_chat_transcription = None
if 'audio_chat_response' not in st.session_state:
    st.session_state.audio_chat_response = None

# Setup Anthropic client will be handled after we confirm the API key is set.

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
    # Use session state data instead of global variables
    if not st.session_state.pages_text:
        return "No PDF loaded"
        
    # Convert to 0-based index
    page_idx = int(page_number) - 1
    
    # Calculate the range of pages to include
    start_page = max(0, page_idx - chunk_width//2)
    end_page = min(st.session_state.number_of_pages - 1, page_idx + chunk_width//2)
    
    # Get text from the range of pages
    return ' '.join(st.session_state.pages_text[start_page:end_page + 1])

def get_completion_pdf(anthropic_client, xai_client, simple_prompt, page_number, chunk_width):
    """Get completion from Claude for PDF content"""
    st.write(f"Analyzing pages around page {page_number} with width {chunk_width}")
    
    text_chunk = chunk_getter(page_number, chunk_width)
            
    prompt = ("Here is an extract from an academic book:\n\n"
             f"<book>{text_chunk}</book>\n\n"
             f"Use this context to answer the following question: {simple_prompt}\n"
             "Respond with None only if the excerpt does not contain detailed information about the question. "
             "Otherwise also indicate section and chapter of the book.")
    
    st.session_state.message_history.append({"role": 'user', "content": prompt})
    
    # Determine which client to use
    if st.session_state.model_name in MODEL_OPTIONS.values():
        if not anthropic_client:
            return "Anthropic client not initialized. Please check your API key."
        response = anthropic_client.messages.create(
            model=st.session_state.model_name,
            max_tokens=2048,
            temperature=0.75,
            system="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts.",
            messages=st.session_state.message_history
        )
        return response.content[0].text
    elif st.session_state.model_name in GROK_MODELS.values():
        if not xai_client:
            return "xAI client not initialized. Please check your API key."
        # For Grok, the prompt needs to be wrapped for Langchain
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts."),
            HumanMessage(content=prompt)
        ]
        response = xai_client.invoke(messages)
        return response.content
    else:
        return "Selected model not supported."

def find_section_and_respond(anthropic_client, xai_client, simple_prompt, page_number, chunk_width):
    """Incrementally increase chunk width to find relevant content"""
    max_width = min(32, st.session_state.number_of_pages)  # Cap the maximum width
    attempts = 0
    max_attempts = 5  # Prevent infinite loops
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while chunk_width <= max_width and attempts < max_attempts:
        progress = attempts / max_attempts
        progress_bar.progress(progress)
        status_text.write(f"Attempt {attempts+1}/{max_attempts}: Checking with chunk width {chunk_width}")
        
        time.sleep(1)  # Brief delay for UI update
        delta = random.uniform(-1, 1) * 0.1 * page_number
        
        x_response = get_completion_pdf(anthropic_client, xai_client, simple_prompt, page_number + delta, chunk_width)
        
        if "None" not in x_response and len(x_response) > 300:
            st.success(f"Found relevant content with width: {chunk_width}")
            st.session_state.message_history.append({"role": 'assistant', "content": x_response})
            progress_bar.progress(1.0)
            return x_response
        
        # Geometric growth (multiply by 2 each time)
        chunk_width = chunk_width * 2
        attempts += 1
    
    progress_bar.progress(1.0)
    # If we get here, we couldn't find a good response
    return "Could not find relevant information in the specified section of the document."

def get_system_response(anthropic_client, xai_client, simple_prompt):
    """Get standard response from Claude"""
    st.session_state.message_history.append({"role": 'user', "content": simple_prompt})
    
    
    try:
        # Determine which client to use
        if st.session_state.model_name in MODEL_OPTIONS.values():
            if not anthropic_client:
                return "Anthropic client not initialized. Please check your API key."
            response = anthropic_client.messages.create(
                model=st.session_state.model_name,
                max_tokens=2048,
                temperature=0.75,
                system="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts.",
                messages=st.session_state.message_history
            )
            result = response.content[0].text
        elif st.session_state.model_name in GROK_MODELS.values():
            if not xai_client:
                return "xAI client not initialized. Please check your API key."
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts."),
                HumanMessage(content=simple_prompt)
            ]
            response = xai_client.invoke(messages)
            result = response.content
        else:
            return "Selected model not supported."
        
        st.session_state.message_history.append({"role": 'assistant', "content": result})
        return result
    except Exception as e:
        st.error(f"Error in get_system_response: {str(e)}")
        st.write(f"Error type: {type(e).__name__}")
        import traceback
        st.write(f"Traceback: {traceback.format_exc()}")
        return "An error occurred while processing your request."

def get_audio_hash(audio_file_path):
    """Generate a hash of the audio file for caching purposes"""
    with open(audio_file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def transcribe_audio(audio_file_path, model_type="base", language=None, use_cache=False):
    """
    Transcribe audio with optional caching to avoid repeated transcriptions
    """
    if use_cache:
        file_hash = get_audio_hash(audio_file_path)
        
        # Create a cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcription_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{file_hash}_{language}.json")
        
        # Check if we have a cached version
        if os.path.exists(cache_file):
            st.info(f"Loading cached transcription for {os.path.basename(audio_file_path)}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    # If not cached or caching disabled, perform the transcription
    with st.spinner(f"Transcribing {os.path.basename(audio_file_path)}..."):
        audio = whisper.load_audio(audio_file_path)
        
        # Ensure model is loaded
        whisper_model = whisper.load_model(model_type, device="cpu")
        
        # Perform transcription
        result = whisper.transcribe(whisper_model, audio, language=language)
        
        # Cache the result if caching is enabled
        if use_cache:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result

# Main Streamlit app
st.title("PDF and Audio Analysis App")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key Management
    if not st.session_state.get("anthropic_api_key") and not st.session_state.get("xai_api_key"):
        st.warning("Anthropic API key not found!")
        api_key_input = st.text_input(
            "Enter your Anthropic API Key", 
            type="password", 
            key="api_key_input",
            help="You can find your API key on your Anthropic dashboard."
        )
        if api_key_input:
            st.session_state.anthropic_api_key = api_key_input
            st.success("API Key set for this session.")
            st.rerun()
    else:
        st.success("Anthropic API key is set.")

    # XAI API Key Management
    if not st.session_state.get("xai_api_key"):
        st.warning("XAI API key not found!")
        xai_key_input = st.text_input(
            "Enter your xAI API Key", 
            type="password", 
            key="xai_api_key_input",
            help="You can find your API key on your xAI dashboard (e.g., Grok)."
        )
        if xai_key_input:
            st.session_state.xai_api_key = xai_key_input
            st.success("xAI API Key set for this session.")
            st.rerun()
    else:
        st.success("xAI API key is set.")
    
    # Combine model options
    ALL_MODEL_OPTIONS = {**MODEL_OPTIONS, **GROK_MODELS}
    selected_model = st.selectbox("Select AI Model", list(ALL_MODEL_OPTIONS.keys()), index=0)
    st.session_state.model_name = ALL_MODEL_OPTIONS[selected_model]
    
    # PDF Upload
    st.subheader("PDF Upload")
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_pdf:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_pdf.getvalue())
            pdf_path = tmp_file.name
        
        # Process the PDF
        reader = PdfReader(pdf_path)
        st.session_state.number_of_pages = len(reader.pages)
        st.session_state.pages_text = [page.extract_text() for page in reader.pages]
        st.session_state.book_name = uploaded_pdf.name
        st.success(f"PDF loaded with {st.session_state.number_of_pages} pages")
    
    # Clear conversation
    if st.sidebar.button("Clear Conversation"):
        st.session_state.message_history = []
        st.success("Conversation history cleared!")

# Main app logic
if not st.session_state.get("anthropic_api_key") and not st.session_state.get("xai_api_key"):
    st.error("Please enter at least one API key (Anthropic or xAI) in the sidebar to continue.")
else:
    anthropic_client = None
    xai_client = None

    if st.session_state.get("anthropic_api_key"):
        try:
            anthropic_client = Anthropic(api_key=st.session_state.anthropic_api_key)
        except Exception as e:
            st.error(f"Failed to initialize Anthropic client. Please check your API key. Error: {e}")

    if st.session_state.get("xai_api_key"):
        try:
            xai_client = ChatOpenAI(
                openai_api_key=st.session_state.xai_api_key,
                openai_api_base="https://api.x.ai/v1",
                model=st.session_state.model_name # This will be updated in get_system_response
            )
        except Exception as e:
            st.error(f"Failed to initialize xAI client. Please check your API key. Error: {e}")

    if anthropic_client or xai_client:
        # Mode selection
        mode = st.selectbox(
            "Select Mode", 
            ["Text Chat (Mode 0)", 
             "Audio Chat (Mode 1)", 
             "Text-PDF Analysis (Mode 2)", 
             "Audio-PDF Analysis (Mode 3)", 
             "Audio Analysis (Mode 4)"]
        )

        # Mode 0: Text Chat
        if mode == "Text Chat (Mode 0)":
            st.header("Text Chat")
            
            user_input = st.text_area("Your message:")
            
            if st.button("Send"):
                if user_input:
                    st.write(f"**You:** {user_input}")
                    with st.spinner("AI is thinking..."):
                        response = get_system_response(anthropic_client, xai_client, user_input)
                    st.write(f"**AI:** {response}")

        # Mode 1: Audio Chat
        elif mode == "Audio Chat (Mode 1)":
            st.header("Audio Chat")
            
            st.write("Record your message:")
            
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Start Recording"):
                    # Clear previous results
                    st.session_state.audio_chat_transcription = None
                    st.session_state.audio_chat_response = None
                    if st.session_state.recorder is None:
                        st.session_state.recorder = AudioRecorder()
                    st.session_state.recorder.start()
                    st.session_state.recording_status = "recording"
                    st.info("Recording... Click 'Stop Recording' to finish.")

            with col2:
                if st.button("Stop Recording"):
                    if st.session_state.recorder and st.session_state.recording_status == "recording":
                        st.session_state.recorder.stop("recording.wav")
                        st.session_state.recording_status = "stopped"
                        st.session_state.recorder = None  # Reset recorder
                        st.success("Recording finished!")

                        # Process the recording
                        with st.spinner("Transcribing..."):
                            supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                            whisper_model_type = supported_models[1]
                            
                            result = transcribe_audio("recording.wav", whisper_model_type, "en", use_cache=False)
                            st.session_state.audio_chat_transcription = result["text"]
                            
                        # Get response from Claude
                        with st.spinner("AI is thinking..."):
                            response = get_system_response(anthropic_client, xai_client, st.session_state.audio_chat_transcription)
                            st.session_state.audio_chat_response = response

                        st.rerun()
                    else:
                        st.warning("Not currently recording.")

            # Display the results full-width
            if st.session_state.audio_chat_transcription:
                st.write(f"**You said:** {st.session_state.audio_chat_transcription}")
            if st.session_state.audio_chat_response:
                st.write(f"**AI:** {st.session_state.audio_chat_response}")

        # Mode 2: Text-PDF Analysis
        elif mode == "Text-PDF Analysis (Mode 2)":
            st.header("Text-PDF Analysis")
            
            if not st.session_state.pages_text:
                st.warning("Please upload a PDF document in the sidebar first.")
            else:
                st.write(f"Currently analyzing: {st.session_state.book_name}")
                st.write(f"Document has {st.session_state.number_of_pages} pages")
                
                simple_prompt = st.text_area("Enter your question about the PDF:")
                page_number = st.number_input("Enter page number:", min_value=1, max_value=st.session_state.number_of_pages, value=1)
                
                if st.button("Ask Question"):
                    if simple_prompt:
                        with st.spinner("Analyzing document..."):
                            explanation = find_section_and_respond(anthropic_client, xai_client, simple_prompt, page_number, 2)
                        st.write("**Answer:**")
                        st.write(explanation)

        # Mode 3: Audio-PDF Analysis
        elif mode == "Audio-PDF Analysis (Mode 3)":
            st.header("Audio-PDF Analysis")
            
            if not st.session_state.pages_text:
                st.warning("Please upload a PDF document in the sidebar first.")
            else:
                st.write(f"Currently analyzing: {st.session_state.book_name}")
                
                # Step 1: Record question
                st.subheader("Step 1: Record your question")
                
                if 'question_text' not in st.session_state:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Start Recording Question"):
                            if st.session_state.recorder is None:
                                st.session_state.recorder = AudioRecorder()
                            st.session_state.recorder.start()
                            st.session_state.recording_status = "recording_question"
                            st.info("Recording question...")

                    with col2:
                        if st.button("Stop Recording Question"):
                            if st.session_state.recorder and st.session_state.recording_status == "recording_question":
                                st.session_state.recorder.stop("recording.wav")
                                st.session_state.recording_status = "stopped"
                                st.session_state.recorder = None
                                st.success("Recording finished!")
                                
                                # Process the recording
                                supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                                whisper_model_type = supported_models[0]
                                
                                result = transcribe_audio("recording.wav", whisper_model_type, "en", use_cache=False)
                                st.session_state.question_text = result["text"]
                                
                                st.write(f"**Your question:** {st.session_state.question_text}")
                                st.rerun()

                else:
                    st.write(f"**Your question:** {st.session_state.question_text}")
                    if st.button("Record Again"):
                        del st.session_state.question_text
                        st.rerun()
                
                # Step 2: Get page number
                if 'question_text' in st.session_state:
                    st.subheader("Step 2: Specify the page number")
                    
                    if 'page_number' not in st.session_state:
                        st.write("Say: the information is on page ...")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Start Recording Page Number"):
                                if st.session_state.recorder is None:
                                    st.session_state.recorder = AudioRecorder()
                                st.session_state.recorder.start()
                                st.session_state.recording_status = "recording_page"
                                st.info("Recording page number...")

                        with col2:
                            if st.button("Stop Recording Page Number"):
                                if st.session_state.recorder and st.session_state.recording_status == "recording_page":
                                    st.session_state.recorder.stop("recording.wav")
                                    st.session_state.recording_status = "stopped"
                                    st.session_state.recorder = None
                                    st.success("Recording finished!")
                                    
                                    # Process the recording
                                    supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                                    whisper_model_type = supported_models[0]
                                    
                                    result = transcribe_audio("recording.wav", whisper_model_type, "en", use_cache=False)
                                    page_text = result["text"]
                                    
                                    try:
                                        extracted_page = extract_number(page_text)
                                        if extracted_page and 1 <= extracted_page <= st.session_state.number_of_pages:
                                            st.session_state.page_number = extracted_page
                                            st.write(f"**Page number recognized:** {extracted_page}")
                                            st.rerun()
                                        else:
                                            st.error(f"Could not extract a valid page number from '{page_text}'")
                                    except Exception as e:
                                        st.error(f"Failed to process the page number. Error: {e}")
                        
                        # Alternative: manual page input
                        manual_page = st.number_input("Or enter page manually:", min_value=1, max_value=st.session_state.number_of_pages)
                        if st.button("Use This Page"):
                            st.session_state.page_number = manual_page
                            st.rerun()
                    else:
                        st.write(f"**Selected page:** {st.session_state.page_number}")
                        if st.button("Change Page"):
                            del st.session_state.page_number
                            st.rerun()
                
                # Step 3: Show results
                if 'question_text' in st.session_state and 'page_number' in st.session_state:
                    st.subheader("Step 3: Get Answer")
                    
                    st.write(f"**Question:** {st.session_state.question_text}")
                    st.write(f"**Page:** {st.session_state.page_number}")
                    
                    if st.button("Get Answer"):
                        with st.spinner("Analyzing PDF..."):
                            explanation = find_section_and_respond(anthropic_client, xai_client, st.session_state.question_text, st.session_state.page_number, 2)
                        
                        st.write("**Answer:**")
                        st.write(explanation)
                    
                    if st.button("Start Over"):
                        if 'question_text' in st.session_state:
                            del st.session_state.question_text
                        if 'page_number' in st.session_state:
                            del st.session_state.page_number
                        st.rerun()

        # Mode 4: Audio Analysis
        elif mode == "Audio Analysis (Mode 4)":
            st.header("Audio Lecture Analysis")
            
            # Upload lecture audio
            uploaded_lecture = st.file_uploader("Upload lecture audio", type=["wav", "mp3"])
            
            if uploaded_lecture:
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_lecture.getvalue())
                    lecture_path = tmp_file.name
                
                if st.button("Analyze Lecture"):
                    # Process the lecture
                    with st.spinner("Transcribing lecture... This may take several minutes for long recordings."):
                        supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                        whisper_model_type = supported_models[0]
                        
                        result = transcribe_audio(lecture_path, whisper_model_type, "en", use_cache=True)
                        lecture_text = result["text"]
                    
                    st.success("Transcription complete!")
                    
                    with st.spinner("AI is analyzing the lecture content..."):
                        lp = get_long_prompt(lecture_text)
                        system_response = get_system_response(anthropic_client, xai_client, lp)
                    
                    # Save the analysis to HTML
                    html_file = save_to_html(system_response)
                    st.success(f"Analysis saved to {html_file}")
                    
                    # Display the analysis
                    st.subheader("Lecture Analysis")
                    st.write(system_response)
                    
                    # Provide download option
                    with open(html_file, "rb") as file:
                        st.download_button(
                            label="Download HTML Analysis",
                            data=file,
                            file_name="lecture_analysis.html",
                            mime="text/html"
                        )

# Display conversation history
st.sidebar.subheader("Conversation History")
if st.session_state.message_history:
    for i, msg in enumerate(st.session_state.message_history[-10:]):  # Show last 10 messages
        role = "👤 You" if msg["role"] == "user" else "🤖 Claude"
        st.sidebar.text(f"{role}: {msg['content'][:50]}...")





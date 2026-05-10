import streamlit as st
from anthropic import Anthropic
from pypdf import PdfReader
from functools import lru_cache
from math import ceil, floor
import time
import random
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
import uuid
from datetime import datetime, timedelta, timezone
from streamlit_mic_recorder import mic_recorder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
grok_api_key = os.getenv("XAI_API_KEY") 

APP_TEMP_DIR = os.path.join(tempfile.gettempdir(), "academic_learning_assistant")
TEMP_FILE_MAX_AGE_SECONDS = 60 * 60 * 6
USER_ACCESS_FILE = os.path.join(os.path.dirname(__file__), "user_access.json")
TRIAL_DAYS = 15
PAYPAL_DONATE_URL = os.getenv("PAYPAL_DONATE_URL", "")

#second change

#https://docs.anthropic.com/en/docs/about-claude/models/overview

# Define model options
MODEL_OPTIONS = {
    "Claude Haiku 4.5": "claude-haiku-4-5-20251001",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "Claude Opus 4.6": "claude-opus-4-6"
}

GROK_MODELS = {
    "Grok 4.3": "grok-4.3",
    "Grok 4.1 Fast Reason": "grok-4-1-fast-reasoning",
    "Grok 4.1 Fast Non-Reason": "grok-4-1-fast-non-reasoning"
}

# Combine model options for easy reference
ALL_MODEL_OPTIONS = {**MODEL_OPTIONS, **GROK_MODELS}

MODEL_PRICING_USD_PER_MILLION = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "grok-4.3": {"input": 2.00, "output": 8.00},
    "grok-4-1-fast-reasoning": {"input": 5.00, "output": 20.00},
    "grok-4-1-fast-non-reasoning": {"input": 1.50, "output": 6.00},
}


def _extract_usage_from_response(response):
    input_tokens = 0
    output_tokens = 0

    usage = getattr(response, "usage", None)
    if usage is not None:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return input_tokens, output_tokens

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        input_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
        output_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
        if input_tokens or output_tokens:
            return input_tokens, output_tokens

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage", {}) or {}
        input_tokens = int(
            token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0
        )
        output_tokens = int(
            token_usage.get(
                "completion_tokens",
                token_usage.get("output_tokens", token_usage.get("reasoning_tokens", 0)),
            )
            or 0
        )

    return input_tokens, output_tokens


def _build_cost_info(model_name, input_tokens, output_tokens):
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model_name)
    info = {
        "model": model_name,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "input_price_per_million": None,
        "output_price_per_million": None,
        "input_cost": None,
        "output_cost": None,
        "total_cost": None,
    }

    if not pricing:
        return info

    input_price = float(pricing["input"])
    output_price = float(pricing["output"])
    input_cost = (info["input_tokens"] / 1_000_000) * input_price
    output_cost = (info["output_tokens"] / 1_000_000) * output_price

    info["input_price_per_million"] = input_price
    info["output_price_per_million"] = output_price
    info["input_cost"] = input_cost
    info["output_cost"] = output_cost
    info["total_cost"] = input_cost + output_cost
    return info


def _cost_info_from_response(response):
    model_name = st.session_state.model_name
    input_tokens, output_tokens = _extract_usage_from_response(response)
    return _build_cost_info(model_name, input_tokens, output_tokens)


def _merge_cost_info(total_info, new_info):
    if not new_info:
        return total_info
    if not total_info:
        return dict(new_info)

    merged = dict(total_info)
    merged["input_tokens"] = int(merged.get("input_tokens", 0)) + int(new_info.get("input_tokens", 0))
    merged["output_tokens"] = int(merged.get("output_tokens", 0)) + int(new_info.get("output_tokens", 0))

    input_cost_a = merged.get("input_cost")
    input_cost_b = new_info.get("input_cost")
    output_cost_a = merged.get("output_cost")
    output_cost_b = new_info.get("output_cost")

    merged["input_cost"] = (
        (input_cost_a or 0.0) + (input_cost_b or 0.0)
        if (input_cost_a is not None or input_cost_b is not None)
        else None
    )
    merged["output_cost"] = (
        (output_cost_a or 0.0) + (output_cost_b or 0.0)
        if (output_cost_a is not None or output_cost_b is not None)
        else None
    )
    if merged["input_cost"] is not None or merged["output_cost"] is not None:
        merged["total_cost"] = (merged["input_cost"] or 0.0) + (merged["output_cost"] or 0.0)
    else:
        merged["total_cost"] = None

    return merged


def _format_cost_line(cost_info):
    if not cost_info:
        return "Cost: unavailable"

    model = cost_info.get("model", "unknown")
    input_tokens = int(cost_info.get("input_tokens", 0) or 0)
    output_tokens = int(cost_info.get("output_tokens", 0) or 0)
    total_cost = cost_info.get("total_cost")

    if total_cost is None:
        return f"Cost ({model}): usage {input_tokens} in / {output_tokens} out tokens; pricing unavailable"

    return (
        f"Cost ({model}): ${total_cost:.6f} "
        f"({input_tokens} input + {output_tokens} output tokens)"
    )
# Initialize MODEL_NAME in session state if not already present
if 'model_name' not in st.session_state:
    st.session_state.model_name = MODEL_OPTIONS["Claude Sonnet 4.6"]  # Default model

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
if 'audio_chat_transcription' not in st.session_state:
    st.session_state.audio_chat_transcription = None
if 'audio_chat_response' not in st.session_state:
    st.session_state.audio_chat_response = None
if 'audio_chat_wav_bytes' not in st.session_state:
    st.session_state.audio_chat_wav_bytes = None
if 'audio_chat_last_hash' not in st.session_state:
    st.session_state.audio_chat_last_hash = None
if 'audio_chat_transcription_input' not in st.session_state:
    st.session_state.audio_chat_transcription_input = ""
if 'audio_chat_transcription_editor' not in st.session_state:
    st.session_state.audio_chat_transcription_editor = ""
if 'question_text_confirmed' not in st.session_state:
    st.session_state.question_text_confirmed = False
if 'question_text_input' not in st.session_state:
    st.session_state.question_text_input = ""
if 'pdf_question_editor' not in st.session_state:
    st.session_state.pdf_question_editor = ""
if 'last_response_cost' not in st.session_state:
    st.session_state.last_response_cost = None

# Setup Anthropic client will be handled after we confirm the API key is set.


def _ensure_app_temp_dir():
    os.makedirs(APP_TEMP_DIR, exist_ok=True)
    return APP_TEMP_DIR


def _safe_remove_file(file_path):
    if not file_path:
        return
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _purge_stale_temp_files(max_age_seconds=TEMP_FILE_MAX_AGE_SECONDS):
    temp_dir = _ensure_app_temp_dir()
    cutoff = time.time() - max_age_seconds

    for entry in os.scandir(temp_dir):
        if not entry.is_file():
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                os.remove(entry.path)
        except FileNotFoundError:
            continue
        except Exception:
            continue


_purge_stale_temp_files()


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_utc(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _detect_device_type(user_agent):
    ua = (user_agent or "").lower()
    if any(token in ua for token in ["mobile", "iphone", "android"]):
        return "mobile"
    if any(token in ua for token in ["ipad", "tablet"]):
        return "tablet"
    return "desktop"


def _get_header_value(headers, key_name):
    if not headers:
        return ""
    for key, value in headers.items():
        if str(key).lower() == key_name.lower():
            return str(value)
    return ""


def _get_client_context():
    context = getattr(st, "context", None)
    headers = getattr(context, "headers", {}) if context else {}

    forwarded_for = _get_header_value(headers, "x-forwarded-for")
    real_ip = _get_header_value(headers, "x-real-ip")
    remote_addr = _get_header_value(headers, "remote-addr")
    ip = (forwarded_for.split(",")[0].strip() if forwarded_for else "") or real_ip or remote_addr or "unknown-ip"

    user_agent = _get_header_value(headers, "user-agent") or "unknown-agent"
    device_type = _detect_device_type(user_agent)

    qp = st.query_params
    browser_cookie = qp.get("vid", "")
    if not browser_cookie:
        browser_cookie = uuid.uuid4().hex
        st.query_params["vid"] = browser_cookie

    return {
        "ip": ip,
        "user_agent": user_agent,
        "device_type": device_type,
        "browser_cookie": browser_cookie,
    }


def _build_user_id(client_ctx):
    raw = f"{client_ctx['ip']}|{client_ctx['device_type']}|{client_ctx['browser_cookie']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_access_data():
    if not os.path.exists(USER_ACCESS_FILE):
        return {"users": {}}
    try:
        with open(USER_ACCESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("users"), dict):
                return data
    except Exception:
        pass
    return {"users": {}}


def _save_access_data(data):
    with open(USER_ACCESS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_or_create_user_record(user_id, client_ctx):
    data = _load_access_data()
    users = data.setdefault("users", {})
    now_iso = _utcnow_iso()

    if user_id not in users:
        users[user_id] = {
            "created_at": now_iso,
            "trial_started_at": now_iso,
            "trial_days": TRIAL_DAYS,
            "donation_unlocked": False,
            "donation_unlocked_at": None,
            "last_seen_at": now_iso,
            "ip": client_ctx["ip"],
            "device_type": client_ctx["device_type"],
            "browser_cookie": client_ctx["browser_cookie"],
            "user_agent": client_ctx["user_agent"],
        }
    else:
        users[user_id]["last_seen_at"] = now_iso
        users[user_id]["ip"] = client_ctx["ip"]
        users[user_id]["device_type"] = client_ctx["device_type"]
        users[user_id]["browser_cookie"] = client_ctx["browser_cookie"]
        users[user_id]["user_agent"] = client_ctx["user_agent"]

    _save_access_data(data)
    return users[user_id]


def _compute_access_state(user_record):
    started_at = _parse_iso_utc(user_record.get("trial_started_at"))
    if started_at is None:
        started_at = datetime.now(timezone.utc)

    trial_days = int(user_record.get("trial_days", TRIAL_DAYS) or TRIAL_DAYS)
    trial_end = started_at + timedelta(days=trial_days)
    now = datetime.now(timezone.utc)

    trial_active = now <= trial_end
    donation_unlocked = bool(user_record.get("donation_unlocked", False))
    full_access = trial_active or donation_unlocked

    remaining = trial_end - now
    days_left = max(0, remaining.days + (1 if remaining.seconds > 0 else 0))

    return {
        "trial_active": trial_active,
        "donation_unlocked": donation_unlocked,
        "full_access": full_access,
        "days_left": days_left,
        "trial_end": trial_end,
    }


def _mark_current_user_donated(user_id):
    data = _load_access_data()
    users = data.setdefault("users", {})
    user = users.get(user_id)
    if not user:
        return False
    user["donation_unlocked"] = True
    user["donation_unlocked_at"] = _utcnow_iso()
    _save_access_data(data)
    return True

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
    

def _normalize_math_delimiters(text):
    """
    Wrap bare LaTeX expressions in $ or $$ delimiters for MathJax rendering.
    
    Handles:
    1. The '**Formula:** <math> — <explanation>' pattern (Mode 6)
    2. Inline LaTeX commands in general text
    3. Avoids double-wrapping already-delimited expressions
    """
    if not text:
        return text

    # Already contains math delimiters — skip to avoid double-wrapping
    if "$" in text:
        return text

    # ------------------------------------------------------------------
    # LaTeX command patterns — anything that indicates a math expression
    # ------------------------------------------------------------------
    LATEX_COMMAND = re.compile(
        r"\\(?:"
        r"sum|int|prod|frac|sqrt|infty|cdot|times|div|pm|mp|"
        r"alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|"
        r"lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
        r"partial|nabla|rightarrow|leftarrow|Rightarrow|Leftarrow|"
        r"left|right|langle|rangle|lim|to|text|begin|end|"
        r"hat|bar|tilde|vec|dot|widehat|widetilde|"
        r"mathbb|mathcal|mathbf|mathit|mathrm|mathsf|mathtt|"
        r"cap|cup|subset|supset|in|notin|forall|exists|"
        r"sim|approx|equiv|propto|leq|geq|neq|"
        r"cdotp|colon|circ|bullet|oplus|otimes|odot|"
        r"angle|triangle|square|diamond|"
        r"exp|ln|log|sin|cos|tan|cot|sec|csc|"
        r"arcsin|arccos|arctan|sinh|cosh|tanh|"
        r"max|min|det|gcd|hom|ker|Pr|sup|inf|"
        r"arg|deg|dim|lg|liminf|limsup|"
        r"binom|bmod|pmod|"
        r"overline|underline|overrightarrow|overleftarrow|"
        r"breve|check|"
        r"Bigg?|bigg?"
        r")\b"
    )

    # ------------------------------------------------------------------
    # Step 1 — Handle the structured "Formula:" pattern from Mode 6
    #   Looks like:  **Formula:** <math> — <explanation>
    #   or:           - **Formula:** <math> — <explanation>
    # ------------------------------------------------------------------
    def _wrap_formula_block(m):
        prefix = m.group(1)      # e.g. "**Formula:** " or "- **Formula:** "
        formula = m.group(2).strip()
        sep = m.group(3)         # " — " or " – " or " - "
        explanation = m.group(4)

        if LATEX_COMMAND.search(formula):
            # Standalone formula → display math
            return f"{prefix}$${formula}$${sep}{explanation}"
        return m.group(0)

    text = re.sub(
        r"(\*{0,2}Formula:\*{0,2}\s*)(.+?)(\s*[—–\-]\s*)(.+)",
        _wrap_formula_block,
        text,
    )

    # ------------------------------------------------------------------
    # Step 2 — General inline LaTeX in text
    #   Wrap lines that contain LaTeX commands but no $ delimiters
    # ------------------------------------------------------------------
    lines = text.split("\n")
    result = []

    for line in lines:
        if not LATEX_COMMAND.search(line):
            result.append(line)
            continue

        stripped = line.strip()

        # Heuristic: count math-syntax characters vs total length
        # High ratio → the line is predominantly math → wrap in $$
        math_chars = len(
            re.findall(r"[{}()\[\]\^\_=+\-*/\\]", stripped)
        )
        ratio = math_chars / max(1, len(stripped))

        # Bullet lines:  "- text"  or  "* text"
        bullet_match = re.match(r"^(\s*)([-*]\s+)(.*)", line)
        if bullet_match:
            indent = bullet_match.group(1)
            bullet = bullet_match.group(2)
            rest = bullet_match.group(3).strip()
            if ratio > 0.12:
                result.append(f"{indent}{bullet}$${rest}$$")
            else:
                result.append(line)
        elif ratio > 0.12:
            # Standalone math line (no surrounding prose)
            result.append(f"$${stripped}$$")
        else:
            # Inline math embedded in prose — wrap the LaTeX
            # sub-sequences individually
            def _wrap_inline(m):
                expr = m.group(0)
                if "$" not in expr:
                    return f"${expr}$"
                return expr

            line = re.sub(
                r"\\[a-zA-Z]+(?:\{[^}]*\})*(?:\^\{[^}]*\})*(?:_\{[^}]*\})*",
                _wrap_inline,
                line,
            )
            result.append(line)

    return "\n".join(result)


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
    # st.write(f"Analyzing pages around page {page_number} with width {chunk_width}")
    
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
            return {"text": "Anthropic client not initialized. Please check your API key.", "cost_info": None}
        response = anthropic_client.messages.create(
            model=st.session_state.model_name,
            max_tokens=2048,
            temperature=0.75,
            system="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts. When presenting mathematical formulas, use LaTeX with $...$ for inline math and $...$ for display math.",
            messages=st.session_state.message_history
        )
        return {
            "text": response.content[0].text,
            "cost_info": _cost_info_from_response(response),
        }
    elif st.session_state.model_name in GROK_MODELS.values():
        if not xai_client:
            return {"text": "xAI client not initialized. Please check your API key.", "cost_info": None}
        # For Grok, the prompt needs to be wrapped for Langchain
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts. When presenting mathematical formulas, use LaTeX with $...$ for inline math and $...$ for display math."),
            HumanMessage(content=prompt)
        ]
        response = xai_client.invoke(messages)
        return {
            "text": response.content,
            "cost_info": _cost_info_from_response(response),
        }
    else:
        return {"text": "Selected model not supported.", "cost_info": None}

def find_section_and_respond(anthropic_client, xai_client, simple_prompt, page_number, chunk_width):
    """Incrementally increase chunk width to find relevant content"""
    max_width = min(32, st.session_state.number_of_pages)  # Cap the maximum width
    attempts = 0
    max_attempts = 5  # Prevent infinite loops
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_cost_info = None
    
    while chunk_width <= max_width and attempts < max_attempts:
        progress = attempts / max_attempts
        progress_bar.progress(progress)
        # status_text.write(f"Attempt {attempts+1}/{max_attempts}: Checking with chunk width {chunk_width}")
        
        time.sleep(1)  # Brief delay for UI update
        delta = random.uniform(-1, 1) * 0.1 * page_number
        
        completion_result = get_completion_pdf(
            anthropic_client,
            xai_client,
            simple_prompt,
            page_number + delta,
            chunk_width,
        )
        x_response = completion_result.get("text", "")
        total_cost_info = _merge_cost_info(total_cost_info, completion_result.get("cost_info"))
        
        if "None" not in x_response and len(x_response) > 300:
            st.success(f"Found relevant content with width: {chunk_width}")
            st.session_state.message_history.append({"role": 'assistant', "content": x_response})
            st.session_state.last_response_cost = total_cost_info
            progress_bar.progress(1.0)
            return x_response
        
        # Geometric growth (multiply by 2 each time)
        chunk_width = chunk_width * 2
        attempts += 1
    
    progress_bar.progress(1.0)
    st.session_state.last_response_cost = total_cost_info
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
                system="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts. When presenting mathematical formulas, use LaTeX with $...$ for inline math and $...$ for display math.",
                messages=st.session_state.message_history
            )
            result = response.content[0].text
            st.session_state.last_response_cost = _cost_info_from_response(response)
        elif st.session_state.model_name in GROK_MODELS.values():
            if not xai_client:
                return "xAI client not initialized. Please check your API key."
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content="You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts. When presenting mathematical formulas, use LaTeX with $...$ for inline math and $...$ for display math."),
                HumanMessage(content=simple_prompt)
            ]
            response = xai_client.invoke(messages)
            result = response.content
            st.session_state.last_response_cost = _cost_info_from_response(response)
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


def get_pdf_summary(anthropic_client, xai_client, start_page, end_page, percentage):
    """
    Summarize a range of pages from the PDF using dynamic chunking.
    Chunk size is calculated based on text density and target summary percentage 
    to ensure the model's output token limit is respected.
    """
    
    # 1. Calculate optimal chunk size
    # Target: Ensure output summary fits within a safe token limit (e.g., 2000 tokens)
    # Rationale: 
    # - Max output tokens is typically 4096.
    # - We want to be safe (2000) to allow for reasoning and formatting.
    # - Formula: Input_Tokens * Percentage = Output_Tokens
    # - So: Max_Input_Tokens = Safe_Output_Tokens / Percentage
    
    SAFE_OUTPUT_TOKENS = 2000
    SAFE_OUTPUT_CHARS = SAFE_OUTPUT_TOKENS * 4  # Approx 4 chars per token
    
    target_input_chars = SAFE_OUTPUT_CHARS / (percentage / 100)
    
    # Calculate average chars per page in the selected range to estimate density
    total_chars = 0
    range_pages = st.session_state.pages_text[start_page-1:end_page]
    valid_pages = 0
    for p_text in range_pages:
        if p_text:
            total_chars += len(p_text)
            valid_pages += 1
    
    avg_chars_per_page = total_chars / valid_pages if valid_pages > 0 else 1000 # Default fallback
    
    # Determine chunk size (number of pages)
    # We want: Chunk_Size * Avg_Chars_Page <= Target_Input_Chars
    calculated_chunk_size = int(target_input_chars / avg_chars_per_page)
    
    # Clamp chunk size to reasonable limits (1 to 10 pages)
    # 1 page minimum. 10 pages max to avoid context window issues or very long processing.
    CHUNK_SIZE = max(1, min(10, calculated_chunk_size))
    
    # Chunk configuration
    page_chunks = []
    for i in range(start_page, end_page + 1, CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE - 1, end_page)
        page_chunks.append((i, chunk_end))
    
    st.session_state.message_history.append({"role": 'user', "content": f"Summarize pages {start_page}-{end_page} ({percentage}%) - Dynamic Chunk Size: {CHUNK_SIZE}"})
    
    system_msg = "You are a university teacher. Express yourself in scientific terms and explain with clarity the concepts. When presenting mathematical content, focus on general formulas and symbolic representations rather than specific numerical calculations. Use LaTeX with $...$ for inline math and $$...$$ for display formulas."
    full_summary = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_chunks = len(page_chunks)
    total_cost_info = None

    try:
        for idx, (chunk_start, chunk_end) in enumerate(page_chunks):
            status_text.write(f"Summarizing pages {chunk_start}-{chunk_end} (Chunk {idx+1}/{total_chunks}, Size: {CHUNK_SIZE})...")
            
            # Extract text for this chunk
            text_chunk = ' '.join(st.session_state.pages_text[chunk_start-1:chunk_end])
            
            prompt = (f"Please provide a detailed summary of the following academic text from pages {chunk_start} to {chunk_end}. "
                      f"The summary should be approximately {percentage}% of the original length. "
                      "Focus on keeping the main concepts and skipping irrelevant factual data. "
                      "When encountering formulas or mathematical derivations, ignore specific numerical calculations and instead represent the general formula or symbolic representation. "
                      "Use LaTeX with $...$ for inline math and $$...$$ for display formulas.\n\n"
                      f"<text>{text_chunk}</text>")
            
            # Determine which client to use
            if st.session_state.model_name in MODEL_OPTIONS.values():
                if not anthropic_client:
                    return "Anthropic client not initialized. Please check your API key."
                response = anthropic_client.messages.create(
                    model=st.session_state.model_name,
                    max_tokens=4096,
                    temperature=0.75,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}]
                )
                chunk_result = response.content[0].text
                total_cost_info = _merge_cost_info(total_cost_info, _cost_info_from_response(response))
            elif st.session_state.model_name in GROK_MODELS.values():
                if not xai_client:
                    return "xAI client not initialized. Please check your API key."
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt)
                ]
                response = xai_client.invoke(messages)
                chunk_result = response.content
                total_cost_info = _merge_cost_info(total_cost_info, _cost_info_from_response(response))
            else:
                return "Selected model not supported."
            
            full_summary.append(chunk_result)
            progress_bar.progress((idx + 1) / total_chunks)
        
        status_text.empty()
        progress_bar.empty()
        
        final_result = "\n\n".join(full_summary)
        st.session_state.message_history.append({"role": 'assistant', "content": final_result})
        st.session_state.last_response_cost = total_cost_info
        return final_result

    except Exception as e:
        st.error(f"Error in get_pdf_summary: {str(e)}")
        return "An error occurred while generating the summary."


def parse_page_ranges(page_range_text: str, max_pages: int):
    """Parse a page selection string into a merged list of (start_page, end_page).

    Supported formats:
    - Single range: "2-12"
    - Disjoint ranges / singles: "1-2, 4, 6-8"

    Returns:
        List[Tuple[int, int]] sorted and merged (overlapping/adjacent ranges are merged).
    """

    text = "" if page_range_text is None else str(page_range_text).strip()
    if not text:
        raise ValueError("Please enter page ranges (e.g., 1-2, 4, 6-8).")

    # Split on commas/newlines; allow whitespace around separators.
    raw_parts = [p.strip() for p in re.split(r"[\n,]+", text) if p.strip()]
    if not raw_parts:
        raise ValueError("Please enter page ranges (e.g., 1-2, 4, 6-8).")

    parsed = []
    for part in raw_parts:
        # Accept either "N" or "A-B" (exactly one hyphen).
        if "-" in part:
            pieces = [x.strip() for x in part.split("-")]
            if len(pieces) != 2 or not pieces[0] or not pieces[1]:
                raise ValueError(
                    f"Invalid range '{part}'. Use formats like 2-12 or 1-2, 4, 6-8."
                )
            a_str, b_str = pieces
        else:
            a_str = b_str = part.strip()

        try:
            a = int(a_str)
            b = int(b_str)
        except ValueError:
            raise ValueError(
                f"Invalid page number in '{part}'. Use formats like 2-12 or 1-2, 4, 6-8."
            )

        if a < 1 or b < 1 or a > max_pages or b > max_pages:
            raise ValueError(
                f"Invalid page range. Please enter values between 1 and {max_pages}."
            )
        if a > b:
            raise ValueError(
                f"Invalid range '{part}'. Start page must be <= end page."
            )

        parsed.append((a, b))

    parsed.sort(key=lambda x: (x[0], x[1]))

    # Merge overlapping or adjacent ranges to reduce repeated work.
    merged = []
    for a, b in parsed:
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    return [(a, b) for a, b in merged]


def get_pdf_formula_summary(anthropic_client, xai_client, start_page, end_page):
    """Extract formulas/equations from a PDF page range with short explanations.

    Notes:
    - This is intentionally NOT a narrative summary.
    - Output should contain only formulas/equations (as faithfully as possible from extracted text)
      and a short explanation for each.
    """

    # Chunking: aim to keep each request within a safe input size.
    SAFE_INPUT_TOKENS = 12000
    SAFE_INPUT_CHARS = SAFE_INPUT_TOKENS * 4  # ~4 chars per token heuristic

    range_pages = st.session_state.pages_text[start_page - 1 : end_page]
    total_chars = sum(len(p or "") for p in range_pages)
    valid_pages = sum(1 for p in range_pages if p)
    avg_chars_per_page = total_chars / valid_pages if valid_pages else 1000

    calculated_chunk_size = int(SAFE_INPUT_CHARS / max(1, avg_chars_per_page))
    CHUNK_SIZE = max(1, min(8, calculated_chunk_size))

    page_chunks = []
    for i in range(start_page, end_page + 1, CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE - 1, end_page)
        page_chunks.append((i, chunk_end))

    st.session_state.message_history.append(
        {
            "role": "user",
            "content": f"Formula_Summarizing pages {start_page}-{end_page} (Chunk Size: {CHUNK_SIZE})",
        }
    )

    system_msg = (
        "You are a university teacher. Extract only mathematical formulas/equations from the provided text. "
        "Ignore purely numerical computations (e.g., 12/3=4, plugging numbers into a formula, arithmetic steps). "
        "Include ONLY symbolic formulas that contain variables/parameters (letters or Greek symbols). "
        "For each formula, provide a very short explanation (1 sentence). "
        "Always wrap formulas in LaTeX math delimiters: $...$ for inline and $$...$$ for display. "
        "Do not write a general summary. Do not include a compression ratio. "
        "If the text contains no explicit formulas/equations, respond with: 'No formulas found.'"
    )

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_chunks = len(page_chunks)
    results = []
    total_cost_info = None

    try:
        for idx, (chunk_start, chunk_end) in enumerate(page_chunks):
            status_text.write(
                f"Extracting formulas from pages {chunk_start}-{chunk_end} (Chunk {idx+1}/{total_chunks}, Size: {CHUNK_SIZE})..."
            )

            text_chunk = " ".join(st.session_state.pages_text[chunk_start - 1 : chunk_end])

            prompt = (
                f"From pages {chunk_start} to {chunk_end}, extract ONLY formulas/equations that appear in the text. "
                "EXCLUDE purely numerical calculations; INCLUDE ONLY symbolic formulas with variables/parameters. "
                "Return a bullet list. Each bullet must be: - **Formula:** $<formula>$ — <1 short sentence explanation>. "
                "Always use LaTeX math delimiters: $...$ for formulas. "
                "If the formula is not cleanly readable, reconstruct it as best as possible, using standard LaTeX mathematical notation. "
                "Do not include any other commentary, preface, or summary.\n\n"
                f"<text>{text_chunk}</text>"
            )

            if st.session_state.model_name in MODEL_OPTIONS.values():
                if not anthropic_client:
                    return "Anthropic client not initialized. Please check your API key."
                response = anthropic_client.messages.create(
                    model=st.session_state.model_name,
                    max_tokens=1200,
                    temperature=0.2,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                chunk_result = response.content[0].text
                total_cost_info = _merge_cost_info(total_cost_info, _cost_info_from_response(response))
            elif st.session_state.model_name in GROK_MODELS.values():
                if not xai_client:
                    return "xAI client not initialized. Please check your API key."
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [SystemMessage(content=system_msg), HumanMessage(content=prompt)]
                response = xai_client.invoke(messages)
                chunk_result = response.content
                total_cost_info = _merge_cost_info(total_cost_info, _cost_info_from_response(response))
            else:
                return "Selected model not supported."

            chunk_text = (chunk_result or "").strip()

            # Avoid repeating this across chunks; emit only if nothing is found overall.
            if chunk_text and chunk_text != "No formulas found.":
                # Best-effort post-filtering to drop bullets that are purely numeric.
                kept_lines = []
                for line in chunk_text.splitlines():
                    if "Formula" in line:
                        after = line.split("**Formula:**", 1)[-1].strip() if "**Formula:**" in line else line
                        formula_part = after
                        for sep in (" — ", " – ", " - "):
                            if sep in after:
                                formula_part = after.split(sep, 1)[0].strip()
                                break
                        if any(ch.isalpha() for ch in formula_part):
                            kept_lines.append(line)
                    else:
                        # Keep non-empty continuation lines only if they contain alphabetic symbols.
                        if line.strip() and any(ch.isalpha() for ch in line):
                            kept_lines.append(line)

                filtered = "\n".join(kept_lines).strip()
                if filtered:
                    results.append(filtered)
            progress_bar.progress((idx + 1) / total_chunks)

        status_text.empty()
        progress_bar.empty()

        if not results:
            final_result = "No formulas found."
        else:
            final_result = "\n\n".join(results).strip()
        st.session_state.message_history.append({"role": "assistant", "content": final_result})
        st.session_state.last_response_cost = total_cost_info
        return final_result

    except Exception as e:
        st.error(f"Error in get_pdf_formula_summary: {str(e)}")
        return "An error occurred while extracting formulas."


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
        st.write(f"Debug: audio length {len(audio)}, duration {len(audio)/16000:.2f}s")
        
        # If audio is too short, return empty transcription to avoid whisper errors
        if len(audio) < 8000:  # less than 0.5 seconds
            st.warning("Audio too short, skipping transcription.")
            return {"text": "", "segments": []}
        
        # Ensure model is loaded with retry for tokenizer download issues
        max_attempts = 3
        whisper_model = None
        for attempt in range(max_attempts):
            try:
                whisper_model = whisper.load_model(model_type, device="cpu")
                break
            except RuntimeError as e:
                if "Failed to load tokenizer" in str(e) and attempt < max_attempts - 1:
                    st.warning(f"Tokenizer loading failed, retrying... ({attempt + 1}/{max_attempts})")
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    raise
        
        # Perform transcription
        try:
            result = whisper.transcribe(whisper_model, audio, language=language,
                                        compute_word_confidence=True,
                                        no_speech_threshold=0.6)
        except IndexError as e:
            st.error(f"IndexError in whisper_timestamped: {e}")
            st.write(f"Audio length: {len(audio)}, duration: {len(audio)/16000:.2f}s")
            import traceback
            st.write(f"Traceback: {traceback.format_exc()}")
            raise

# Cache the result if caching is enabled
        if use_cache:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result


def _extract_wav_bytes(mic_output):
    """Normalize output from streamlit-mic-recorder across versions."""
    if mic_output is None:
        return None
    if isinstance(mic_output, (bytes, bytearray)):
        return bytes(mic_output)
    if isinstance(mic_output, dict):
        # Known keys across different releases
        for k in ("bytes", "audio_bytes", "data"):
            v = mic_output.get(k)
            if isinstance(v, (bytes, bytearray)):
                return bytes(v)
    return None


def _write_wav_bytes_to_tempfile(wav_bytes, prefix="recording_"):
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
        prefix=prefix,
        dir=_ensure_app_temp_dir(),
    )
    tmp.write(wav_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name

# Main Streamlit app
st.title("Academic Learning Assistant")
st.markdown("<h1 style='text-align: center;'></h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inject MathJax for LaTeX rendering (handles $...$ and $$...$$ delimiters)
# ---------------------------------------------------------------------------
st.markdown(
    """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>
<script>
(function() {
  var interval = setInterval(function() {
    if (window.MathJax && window.MathJax.typesetPromise) {
      clearInterval(interval);
      MathJax.typesetPromise();
      var target = document.querySelector('[data-testid="stApp"]') || document.body;
      var observer = new MutationObserver(function(mutations) {
        var shouldTypeset = false;
        for (var i = 0; i < mutations.length; i++) {
          if (mutations[i].addedNodes.length) { shouldTypeset = true; break; }
        }
        if (shouldTypeset) {
          MathJax.typesetPromise().catch(function(err) {  });
        }
      });
      observer.observe(target, { childList: true, subtree: true });
    }
  }, 150);
})();
</script>
""",
    unsafe_allow_html=True,
)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.caption("API keys entered here are stored only for this browser session.")
    
    selected_model = st.selectbox("Select AI Model", list(ALL_MODEL_OPTIONS.keys()), index=0, key="model_selector")
    st.session_state.model_name = ALL_MODEL_OPTIONS[selected_model]

    if selected_model in MODEL_OPTIONS:
        st.caption("Selected model provider: Anthropic")
    elif selected_model in GROK_MODELS:
        st.caption("Selected model provider: xAI (Grok)")

    col_clear_a, col_clear_b = st.columns(2)
    with col_clear_a:
        if st.button("Clear Anthropic Key"):
            st.session_state.anthropic_api_key = ""
            st.success("Anthropic API key cleared for this session.")
            st.rerun()
    with col_clear_b:
        if st.button("Clear xAI Key"):
            st.session_state.xai_api_key = ""
            st.success("xAI API key cleared for this session.")
            st.rerun()

    st.divider()
    st.subheader("Access & Trial")
    client_ctx = _get_client_context()
    user_id = _build_user_id(client_ctx)
    user_record = _get_or_create_user_record(user_id, client_ctx)
    access_state = _compute_access_state(user_record)

    if access_state["trial_active"]:
        st.info(f"Free trial active: {access_state['days_left']} day(s) remaining.")
    elif access_state["donation_unlocked"]:
        st.success("Full access unlocked by donation.")
    else:
        st.warning("Trial ended. Modes 2-6 are locked until donation.")

    if not access_state["full_access"]:
        if PAYPAL_DONATE_URL:
            st.link_button("Donate with PayPal", PAYPAL_DONATE_URL)
            st.caption("After donation, click 'I Donated' to unlock modes 2-6.")
        else:
            st.info("Set PAYPAL_DONATE_URL in .env to enable the donation button.")

        if st.button("I Donated"):
            if _mark_current_user_donated(user_id):
                st.success("Thank you. Full access unlocked for this user.")
                st.rerun()
            st.error("Unable to update donation status. Please try again.")
    
    # Check if selected model is Anthropic and key is missing
    if selected_model in MODEL_OPTIONS:
        if not st.session_state.get("anthropic_api_key"):
            st.info("Anthropic API key required for selected model")
            api_key_input = st.text_input(
                "Enter your Anthropic API Key", 
                type="password", 
                key="api_key_input",
                help="You can find your API key on your Anthropic dashboard."
            )
            if api_key_input:
                st.session_state.anthropic_api_key = api_key_input
                st.success("Anthropic API Key set for this session.")
                st.rerun()
        else:
            st.success("Anthropic API key is set.")
    
    # Check if selected model is Grok and key is missing
    if selected_model in GROK_MODELS:
        if not st.session_state.get("xai_api_key"):
            st.info("xAI API key required for selected model")
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
    
    # PDF Upload
    st.subheader("PDF Upload")
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_pdf:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=_ensure_app_temp_dir()) as tmp_file:
            tmp_file.write(uploaded_pdf.getvalue())
            pdf_path = tmp_file.name
        
        try:
            # Process the PDF
            reader = PdfReader(pdf_path)
            st.session_state.number_of_pages = len(reader.pages)
            st.session_state.pages_text = [page.extract_text() for page in reader.pages]
            st.session_state.book_name = uploaded_pdf.name
            st.success(f"PDF loaded with {st.session_state.number_of_pages} pages")
        finally:
            _safe_remove_file(pdf_path)
    
    # Clear conversation
    if st.sidebar.button("Clear Conversation"):
        st.session_state.message_history = []
        st.success("Conversation history cleared!")

# Main app logic
# Determine if the selected model requires a specific API key
selected_model_name = None
for name, value in ALL_MODEL_OPTIONS.items():
    if value == st.session_state.model_name:
        selected_model_name = name
        break

# Check if the selected model has the required API key
can_proceed = True
error_message = ""

if selected_model_name in MODEL_OPTIONS:
    # Check for Anthropic API key
    if not st.session_state.get("anthropic_api_key"):
        can_proceed = False
        error_message = "Please enter an Anthropic API key in the sidebar to use the selected model."
elif selected_model_name in GROK_MODELS:
    # Check for xAI API key
    if not st.session_state.get("xai_api_key"):
        can_proceed = False
        error_message = "Please enter an xAI API key in the sidebar to use the selected model."

if not can_proceed:
    st.error(error_message)
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
                model=st.session_state.model_name
            )
        except Exception as e:
            st.error(f"Failed to initialize xAI client. Please check your API key. Error: {e}")

    if anthropic_client or xai_client:
        # Mode selection
        free_modes = [
            "Text Chat (Mode 0)",
            "Audio Chat (Mode 1)",
        ]
        all_modes = [
            "Text Chat (Mode 0)",
            "Audio Chat (Mode 1)",
            "Text-PDF Analysis (Mode 2)",
            "Audio-PDF Analysis (Mode 3)",
            "Audio Analysis (Mode 4)",
            "Summarizing (Mode 5)",
            "Formula_Summarizing (Mode 6)",
        ]
        available_modes = all_modes if access_state.get("full_access") else free_modes

        if not access_state.get("full_access"):
            st.warning("Trial ended. Only Mode 0 and Mode 1 are available until donation unlock.")

        mode = st.selectbox(
            "Select Mode", 
            available_modes,
            key="mode_selector"
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
                    st.markdown(f"**AI:**\n\n{_normalize_math_delimiters(response)}")
                    st.caption(_format_cost_line(st.session_state.get("last_response_cost")))

        # Mode 1: Audio Chat
        elif mode == "Audio Chat (Mode 1)":
            st.header("Audio Chat")
            
            st.write("Record your message (uses your browser microphone):")

            mic_output = mic_recorder(
                start_prompt="Start Recording",
                stop_prompt="Stop Recording",
                key="audio_chat_mic",
            )

            wav_bytes = _extract_wav_bytes(mic_output)
            if wav_bytes:
                audio_hash = hashlib.md5(wav_bytes).hexdigest()
                if st.session_state.audio_chat_last_hash != audio_hash:
                    # New recording captured
                    st.session_state.audio_chat_last_hash = audio_hash
                    st.session_state.audio_chat_wav_bytes = wav_bytes
                    st.session_state.audio_chat_transcription = None
                    st.session_state.audio_chat_response = None

                    wav_path = _write_wav_bytes_to_tempfile(wav_bytes, prefix="audio_chat_")
                    try:
                        with st.spinner("Transcribing..."):
                            supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                            whisper_model_type = supported_models[0]
                            result = transcribe_audio(wav_path, whisper_model_type, "en", use_cache=False)
                            st.session_state.audio_chat_transcription = result["text"]
                            st.session_state.audio_chat_transcription_input = result["text"]
                            st.session_state.audio_chat_transcription_editor = result["text"]
                    finally:
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass

                    st.rerun()

            # Display the results full-width
            if st.session_state.audio_chat_wav_bytes:
                st.audio(st.session_state.audio_chat_wav_bytes, format="audio/wav")
            if st.session_state.audio_chat_transcription:
                edited_transcription = st.text_area(
                    "Review and edit transcription before sending:",
                    key="audio_chat_transcription_editor",
                    height=110,
                )
                st.session_state.audio_chat_transcription_input = edited_transcription

                if st.button("Send to AI", key="audio_chat_send"):
                    if edited_transcription.strip():
                        st.session_state.audio_chat_transcription = edited_transcription.strip()
                        with st.spinner("AI is thinking..."):
                            response = get_system_response(
                                anthropic_client,
                                xai_client,
                                st.session_state.audio_chat_transcription,
                            )
                            st.session_state.audio_chat_response = response
                        st.rerun()
                    else:
                        st.warning("Transcription is empty. Please edit it or record again.")

                if st.button("Record Again", key="audio_chat_record_again"):
                    st.session_state.audio_chat_transcription = None
                    st.session_state.audio_chat_transcription_input = ""
                    st.session_state.audio_chat_transcription_editor = ""
                    st.session_state.audio_chat_response = None
                    st.rerun()

                st.write(f"**Final transcription sent:** {st.session_state.audio_chat_transcription}")
            if st.session_state.audio_chat_response:
                st.markdown(f"**AI:**\n\n{_normalize_math_delimiters(st.session_state.audio_chat_response)}")
                st.caption(_format_cost_line(st.session_state.get("last_response_cost")))

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
                        st.markdown(_normalize_math_delimiters(explanation))
                        st.caption(_format_cost_line(st.session_state.get("last_response_cost")))
        elif mode == "Audio-PDF Analysis (Mode 3)":
            st.header("Audio-PDF Analysis")
            
            if not st.session_state.pages_text:
                st.warning("Please upload a PDF document in the sidebar first.")
            else:
                st.write(f"Currently analyzing: {st.session_state.book_name}")
                
                # Step 1: Record question
                st.subheader("Step 1: Record your question")
                
                if 'question_text' not in st.session_state:
                    mic_q = mic_recorder(
                        start_prompt="Start Recording Question",
                        stop_prompt="Stop Recording Question",
                        key="pdf_question_mic",
                    )
                    wav_bytes_q = _extract_wav_bytes(mic_q)
                    if wav_bytes_q:
                        q_hash = hashlib.md5(wav_bytes_q).hexdigest()
                        if st.session_state.get("pdf_question_last_hash") != q_hash:
                            st.session_state.pdf_question_last_hash = q_hash

                            wav_path = _write_wav_bytes_to_tempfile(wav_bytes_q, prefix="pdf_question_")
                            try:
                                supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                                whisper_model_type = supported_models[0]
                                result = transcribe_audio(wav_path, whisper_model_type, "en", use_cache=False)
                                st.session_state.question_text = result["text"]
                                st.session_state.question_text_input = result["text"]
                                st.session_state.pdf_question_editor = result["text"]
                                st.session_state.question_text_confirmed = False
                            finally:
                                try:
                                    os.remove(wav_path)
                                except Exception:
                                    pass

                            st.rerun()

                else:
                    if not st.session_state.get("question_text_confirmed", False):
                        edited_question = st.text_area(
                            "Review and edit your transcribed question:",
                            key="pdf_question_editor",
                            height=110,
                        )
                        st.session_state.question_text_input = edited_question

                        if st.button("Confirm Question", key="pdf_confirm_question"):
                            if edited_question.strip():
                                st.session_state.question_text = edited_question.strip()
                                st.session_state.question_text_confirmed = True
                                st.rerun()
                            else:
                                st.warning("Question is empty. Please edit it or record again.")

                    st.write(f"**Your question:** {st.session_state.question_text}")
                    if st.button("Record Again", key="pdf_question_record_again"):
                        del st.session_state.question_text
                        st.session_state.question_text_input = ""
                        st.session_state.pdf_question_editor = ""
                        st.session_state.question_text_confirmed = False
                        st.rerun()
                    if st.session_state.get("question_text_confirmed", False) and st.button("Edit Question", key="pdf_question_edit"):
                        st.session_state.question_text_confirmed = False
                        st.rerun()
                
                # Step 2: Get page number
                if 'question_text' in st.session_state and st.session_state.get("question_text_confirmed", False):
                    st.subheader("Step 2: Specify the page number")
                    
                    if 'page_number' not in st.session_state:
                        st.write("Say: the information is on page ...")

                        mic_p = mic_recorder(
                            start_prompt="Start Recording Page Number",
                            stop_prompt="Stop Recording Page Number",
                            key="pdf_page_mic",
                        )
                        wav_bytes_p = _extract_wav_bytes(mic_p)
                        if wav_bytes_p:
                            p_hash = hashlib.md5(wav_bytes_p).hexdigest()
                            if st.session_state.get("pdf_page_last_hash") != p_hash:
                                st.session_state.pdf_page_last_hash = p_hash

                                wav_path = _write_wav_bytes_to_tempfile(wav_bytes_p, prefix="pdf_page_")
                                try:
                                    supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
                                    whisper_model_type = supported_models[0]

                                    result = transcribe_audio(wav_path, whisper_model_type, "en", use_cache=False)
                                    page_text = result["text"]

                                    extracted_page = extract_number(page_text)
                                    if extracted_page and 1 <= extracted_page <= st.session_state.number_of_pages:
                                        st.session_state.page_number = extracted_page
                                        st.rerun()
                                    else:
                                        st.error(f"Could not extract a valid page number from '{page_text}'")
                                finally:
                                    try:
                                        os.remove(wav_path)
                                    except Exception:
                                        pass
                         
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
                if 'question_text' in st.session_state and st.session_state.get("question_text_confirmed", False) and 'page_number' in st.session_state:
                    st.subheader("Step 3: Get Answer")
                    
                    st.write(f"**Question:** {st.session_state.question_text}")
                    st.write(f"**Page:** {st.session_state.page_number}")
                    
                    if st.button("Get Answer"):
                        with st.spinner("Analyzing PDF..."):
                            explanation = find_section_and_respond(anthropic_client, xai_client, st.session_state.question_text, st.session_state.page_number, 2)
                        
                        st.write("**Answer:**")
                        st.markdown(_normalize_math_delimiters(explanation))
                        st.caption(_format_cost_line(st.session_state.get("last_response_cost")))
                    
                    if st.button("Start Over"):
                        if 'question_text' in st.session_state:
                            del st.session_state.question_text
                        st.session_state.question_text_input = ""
                        st.session_state.pdf_question_editor = ""
                        st.session_state.question_text_confirmed = False
                        if 'page_number' in st.session_state:
                            del st.session_state.page_number
                        st.rerun()

        # Mode 4: Audio Analysis
        elif mode == "Audio Analysis (Mode 4)":
            st.header("Audio Lecture Analysis")
            
            # Upload lecture audio
            uploaded_lecture = st.file_uploader("Upload lecture audio", type=["wav", "mp3"])
            if uploaded_lecture and st.button("Analyze Lecture"):
                lecture_suffix = os.path.splitext(uploaded_lecture.name or "")[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=lecture_suffix, dir=_ensure_app_temp_dir()) as tmp_file:
                    tmp_file.write(uploaded_lecture.getvalue())
                    lecture_path = tmp_file.name

                try:
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
                    try:
                        with open(html_file, "rb") as file:
                            html_bytes = file.read()
                    finally:
                        _safe_remove_file(html_file)
                    st.success("Analysis prepared for download.")
                    
                    # Display the analysis
                    st.subheader("Lecture Analysis")
                    st.markdown(_normalize_math_delimiters(system_response))
                    st.caption(_format_cost_line(st.session_state.get("last_response_cost")))
                    
                    # Provide download option
                    st.download_button(
                        label="Download HTML Analysis",
                        data=html_bytes,
                        file_name="lecture_analysis.html",
                        mime="text/html"
                    )
                finally:
                    _safe_remove_file(lecture_path)

        # Mode 5: Summarizing
        elif mode == "Summarizing (Mode 5)":
            st.header("Summarizing")
            
            if not st.session_state.pages_text:
                st.warning("Please upload a PDF document in the sidebar first.")
            else:
                st.write(f"Currently summarizing: {st.session_state.book_name}")
                st.write(f"Document has {st.session_state.number_of_pages} pages")
                
                # Input for page ranges
                page_range = st.text_input("Enter page ranges (e.g. 1-2, 4, 6-8):", value=f"1-{st.session_state.number_of_pages}")
                
                # Input for percentage
                percentage = st.slider("Select summary percentage:", min_value=10, max_value=90, value=30, step=5)
                
                if st.button("Summarize"):
                    try:
                        ranges = parse_page_ranges(page_range, st.session_state.number_of_pages)
                        
                        combined_sections = []
                        total_cost_info = None
                        with st.spinner("Generating summary..."):
                            for start_p, end_p in ranges:
                                summary = get_pdf_summary(anthropic_client, xai_client, start_p, end_p, percentage)
                                total_cost_info = _merge_cost_info(total_cost_info, st.session_state.get("last_response_cost"))
                                
                                if len(ranges) > 1:
                                    combined_sections.append(f"#### Pages {start_p}-{end_p}\n\n{summary}")
                                else:
                                    combined_sections.append(summary)
                        
                        final_summary = "\n\n".join(combined_sections).strip()
                        st.session_state.last_response_cost = total_cost_info
                        st.subheader("Summary")
                        st.markdown(_normalize_math_delimiters(final_summary))
                        st.caption(_format_cost_line(st.session_state.get("last_response_cost")))
                    except ValueError as e:
                        st.error(str(e) or "Please enter valid page ranges.")

        # Mode 6: Formula_Summarizing
        elif mode == "Formula_Summarizing (Mode 6)":
            st.header("Formula_Summarizing")

            if not st.session_state.pages_text:
                st.warning("Please upload a PDF document in the sidebar first.")
            else:
                st.write(f"Currently extracting formulas from: {st.session_state.book_name}")
                st.write(f"Document has {st.session_state.number_of_pages} pages")

                page_range = st.text_input(
                    "Enter page ranges (e.g. 1-2, 4, 6-8):",
                    value=f"1-{st.session_state.number_of_pages}",
                    key="formula_page_range",
                )

                if st.button("Extract Formulas"):
                    try:
                        ranges = parse_page_ranges(
                            page_range, st.session_state.number_of_pages
                        )

                        combined_sections = []
                        total_cost_info = None
                        with st.spinner("Extracting formulas..."):
                            for start_p, end_p in ranges:
                                formulas = get_pdf_formula_summary(
                                    anthropic_client,
                                    xai_client,
                                    start_p,
                                    end_p,
                                )
                                total_cost_info = _merge_cost_info(total_cost_info, st.session_state.get("last_response_cost"))

                                if (formulas or "").strip() == "No formulas found.":
                                    continue

                                if len(ranges) > 1:
                                    combined_sections.append(
                                        f"#### Pages {start_p}-{end_p}\n\n{formulas}"
                                    )
                                else:
                                    combined_sections.append(formulas)

                        final_formulas = (
                            "No formulas found."
                            if not combined_sections
                            else "\n\n".join(combined_sections).strip()
                        )
                        st.session_state.last_response_cost = total_cost_info

                        st.subheader("Formulas")
                        st.markdown(_normalize_math_delimiters(final_formulas))
                        st.caption(_format_cost_line(st.session_state.get("last_response_cost")))
                    except ValueError as e:
                        st.error(str(e) or "Please enter valid page ranges.")

# Display conversation history
st.sidebar.subheader("Conversation History")
if st.session_state.message_history:
    for i, msg in enumerate(st.session_state.message_history[-10:]):  # Show last 10 messages
        role = "👤 You" if msg["role"] == "user" else "🤖 Claude"
        st.sidebar.text(f"{role}: {msg['content'][:50]}...")





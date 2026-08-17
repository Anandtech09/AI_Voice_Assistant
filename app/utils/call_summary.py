"""
Call summary writer for Stelar Interior AI Voice Agent.

After each call, writes a structured .txt file with all collected
client details, conversation transcript, and key highlights for the team.
Uses Gemini LLM to intelligently extract client details from transcripts (English, Malayalam, etc.).
Deduplicates file creation so exactly ONE .txt summary file is created per call.
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any

from app.utils.logger import logger

# Directory where call summaries are stored
CALL_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "call_logs")

# Map call_sid -> filename to ensure exactly ONE file per call
_call_file_map: Dict[str, str] = {}


def ensure_log_directory():
    """Create the call_logs directory if it doesn't exist."""
    os.makedirs(CALL_LOGS_DIR, exist_ok=True)


def clean_transcript_text(text: str) -> str:
    """Clean up noise tokens and foreign character junk from transcript text."""
    # Remove <noise> tags
    text = re.sub(r'<noise>', '', text, flags=re.IGNORECASE)
    # Remove foreign non-Latin/non-Malayalam character noise (e.g. Arabic, Tamil, Devanagari noise)
    text = re.sub(r'[^\x00-\x7F\u0D00-\u0D7F\s.,?!\'"-]', '', text)
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_transcript_from_messages(messages: list) -> str:
    """
    Extract a clean text transcript from Pipecat LLMContext messages.
    Filters out system instructions, JSON tool calls/results, and internal frames.
    """
    if not messages:
        return "No conversation recorded."

    transcript_lines = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")

        # Skip system prompt, tool, or function messages
        if role in ("system", "tool", "function"):
            continue

        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            text = " ".join(parts)

        text = clean_transcript_text(text)
        if not text:
            continue

        # Filter out system instructions or JSON payloads
        if text.startswith("{") or text.startswith("["):
            continue
        if "Welcome the caller" in text or "Start the conversation" in text:
            continue
        if "knowledge base" in text.lower() and "found" in text.lower():
            continue

        speaker = "Caller" if role == "user" else "AI Assistant"
        transcript_lines.append(f"[{speaker}] {text}")

    return "\n  ".join(transcript_lines) if transcript_lines else "No dialogue recorded."


def parse_details_from_transcript(transcript: str, existing_details: dict) -> dict:
    """
    Extract client details (name, phone number, location, work type, visit date)
    from transcript using Gemini LLM reasoning.
    """
    details = dict(existing_details)

    if not transcript or transcript == "No conversation recorded.":
        return details

    try:
        from google import genai
        from app.config import settings

        client = genai.Client(api_key=settings.gemini_api_key)

        prompt = f"""You are an expert AI receptionist assistant for Stelar Interior.
Analyze the following voice call transcript (which may be in English, Malayalam, or mixed language) and extract the client details into a JSON object.

JSON keys required:
- caller_name: (the caller's name in English, or empty string if not provided)
- phone_number: (digits only, e.g. 9400879509, or empty string if not provided)
- location: (city or location name in English, e.g. Trivandrum, Kochi, or empty string if not provided)
- work_type: (interior design work types mentioned, e.g. Wardrobe & Storage, Modular Kitchen, Wall Design, or empty string if not provided)
- preferred_date: (preferred site visit date/time in English, e.g. August 17, 2026, or empty string if not provided)
- requirements: (brief summary of specific requirements in English)
- conversation_summary: (brief 1-2 sentence summary of what was discussed)

Return ONLY valid JSON. Do not include markdown formatting or code block backticks.

Transcript:
{transcript}"""

        resp = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt
        )

        text = resp.text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)

        data = json.loads(text)
        if isinstance(data, dict):
            for k in ("caller_name", "phone_number", "location", "work_type", "preferred_date", "requirements", "conversation_summary"):
                if not details.get(k) and data.get(k):
                    val = str(data[k]).strip()
                    if val and val.lower() not in ("null", "none", "not provided", "empty"):
                        details[k] = val

        logger.info("🧠 LLM successfully extracted client details from transcript.")
    except Exception as e:
        logger.warning(f"⚠️ LLM transcript extraction fallback error: {e}")

    return details


def save_summary(
    caller_name: str = "",
    phone_number: str = "",
    location: str = "",
    work_type: str = "",
    preferred_date: str = "",
    requirements: str = "",
    conversation_summary: str = "",
    transcript: str = "",
    call_sid: str = "",
) -> str:
    """
    Write or update a single call summary .txt file per call_sid.
    """
    ensure_log_directory()

    timestamp = datetime.now()

    # Deduplicate file per call_sid
    if call_sid and call_sid in _call_file_map:
        filename = _call_file_map[call_sid]
    else:
        filename = f"call_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        if call_sid:
            _call_file_map[call_sid] = filename

    filepath = os.path.join(CALL_LOGS_DIR, filename)

    existing_details = {
        "caller_name": caller_name,
        "phone_number": phone_number,
        "location": location,
        "work_type": work_type,
        "preferred_date": preferred_date,
        "requirements": requirements,
        "conversation_summary": conversation_summary,
    }

    if transcript:
        parsed = parse_details_from_transcript(transcript, existing_details)
        caller_name = parsed.get("caller_name", "")
        phone_number = parsed.get("phone_number", "")
        location = parsed.get("location", "")
        work_type = parsed.get("work_type", "")
        preferred_date = parsed.get("preferred_date", "")
        requirements = parsed.get("requirements", "")
        conversation_summary = parsed.get("conversation_summary", conversation_summary)

    lines = [
        "=" * 60,
        "  STELAR INTERIOR — CALL SUMMARY",
        "=" * 60,
        "",
        f"  Date & Time           : {timestamp.strftime('%A, %B %d, %Y at %I:%M %p')}",
        "",
        "-" * 60,
        "  CLIENT DETAILS",
        "-" * 60,
        f"  Name                  : {caller_name or 'Not provided'}",
        f"  Phone                 : {phone_number or 'Not provided'}",
        f"  Location              : {location or 'Not provided'}",
        f"  Work Type             : {work_type or 'Not provided'}",
        f"  Preferred Visit Date  : {preferred_date or 'Not provided'}",
        "",
        "-" * 60,
        "  REQUIREMENTS",
        "-" * 60,
        f"  {requirements or (f'Interested in {work_type}' if work_type else 'No specific requirements mentioned.')}",
        "",
        "-" * 60,
        "  CONVERSATION SUMMARY",
        "-" * 60,
        f"  {conversation_summary or 'Call completed.'}",
        "",
    ]

    if transcript:
        lines.extend([
            "-" * 60,
            "  FULL CALL TRANSCRIPT",
            "-" * 60,
            f"  {transcript}",
            "",
        ])

    lines.extend([
        "=" * 60,
        "  ACTION REQUIRED: Confirm visit appointment with client.",
        "=" * 60,
        "",
    ])

    content = "\n".join(lines)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"📝 Call summary saved ({filename})")
    except Exception as e:
        logger.error(f"❌ Failed to save call summary: {e}")
        return ""

    return filename

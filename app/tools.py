"""
Tool/function definitions for the Stelar Interior AI Voice Agent.

These tools are registered with Gemini Live and can be invoked
during voice conversations when the AI determines they're needed.

Tools:
- get_current_datetime: Returns current date, time, and day
- get_weather: Returns live weather data (with geocode caching)
- search_knowledge_base: Searches the Stelar Interior knowledge base
- save_call_summary: Saves collected client details to a .txt file

Latency optimizations:
- Reusable aiohttp session with connection pooling (avoids TCP handshake per call)
- Geocoding results cache (avoids duplicate API calls for same location)
- Tight timeouts (3s total, 1.5s connect) to fail fast
- Weather codes stored as a module-level constant (not rebuilt per call)
"""

import aiohttp
from datetime import datetime, timedelta
from pipecat.services.llm_service import FunctionCallParams

from app.config import settings
from app.knowledge import search_knowledge
from app.utils.call_summary import save_summary

# ── Reusable HTTP session (connection pooling) ───────────────────────
# Creating a new aiohttp.ClientSession per request wastes ~100-200ms on
# TCP handshake + TLS negotiation. A shared session reuses connections.
_http_session: aiohttp.ClientSession = None
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=3, connect=1.5)


async def _get_session() -> aiohttp.ClientSession:
    """Get or create the shared HTTP session for external API calls."""
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(timeout=_HTTP_TIMEOUT)
    return _http_session


# ── Geocoding cache ──────────────────────────────────────────────────
# Avoids calling the geocoding API again for locations already resolved.
# Key: lowercase location name, Value: (latitude, longitude, resolved_name)
_geocode_cache: dict = {}

# ── Weather code descriptions (constant, never changes) ──────────────
_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle",
    53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain",
    63: "Moderate rain", 65: "Heavy rain", 71: "Slight snow fall",
    73: "Moderate snow fall", 75: "Heavy snow fall", 80: "Slight rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


# ── Tool Definitions (JSON Schema for Gemini) ───────────────────────

TOOL_DEFINITIONS = [
    {
        "function_declarations": [
            {
                "name": "get_current_datetime",
                "description": "Get the current date, time, and day of the week. Use this when the caller asks what time it is, what today's date is, or what day it is.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Optional timezone (e.g., 'Asia/Kolkata'). Defaults to server timezone.",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "get_weather",
                "description": "Get the current weather for a location. Use this when the caller asks about weather conditions or temperature.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or location (e.g., 'Kochi', 'Bangalore')",
                        }
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "search_knowledge_base",
                "description": "Search the Stelar Interior knowledge base for information about the company, interior design services, materials, process, warranty, visit scheduling, and FAQs. Use this FIRST when the caller asks any question related to the company or interior design.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up in the knowledge base",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "save_call_summary",
                "description": "Save a summary of the call with all collected client details. You MUST call this before ending every conversation. Include all information gathered during the call.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "caller_name": {
                            "type": "string",
                            "description": "The caller's full name",
                        },
                        "phone_number": {
                            "type": "string",
                            "description": "The caller's contact phone number",
                        },
                        "location": {
                            "type": "string",
                            "description": "The caller's property address or area",
                        },
                        "work_type": {
                            "type": "string",
                            "description": "Type of interior work needed (e.g., 'full home interior', 'modular kitchen', 'office interior')",
                        },
                        "preferred_date": {
                            "type": "string",
                            "description": "Caller's preferred date and time for the site visit",
                        },
                        "requirements": {
                            "type": "string",
                            "description": "Specific requirements, ideas, or preferences mentioned by the caller",
                        },
                        "conversation_summary": {
                            "type": "string",
                            "description": "Brief summary of what was discussed during the call, including key topics and decisions",
                        },
                    },
                    "required": ["conversation_summary"],
                },
            },
        ]
    }
]


# ── Tool Handler Functions ───────────────────────────────────────────

async def get_current_datetime(params: FunctionCallParams):
    """Return current date, time, and relative visit dates (tomorrow, day after tomorrow)."""
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    day_after_tomorrow = now + timedelta(days=2)
    in_three_days = now + timedelta(days=3)

    result = {
        "date": now.strftime("%B %d, %Y"),
        "day": now.strftime("%A"),
        "today_date": now.strftime("%B %d, %Y"),
        "today_day": now.strftime("%A"),
        "time": now.strftime("%I:%M %p"),
        "tomorrow": tomorrow.strftime("%A, %B %d, %Y"),
        "day_after_tomorrow": day_after_tomorrow.strftime("%A, %B %d, %Y"),
        "in_three_days": in_three_days.strftime("%A, %B %d, %Y"),
        "full": now.strftime("%A, %B %d, %Y at %I:%M %p"),
        "instruction": "Use the 'tomorrow' or 'day_after_tomorrow' string when confirming relative visit requests!",
    }
    await params.result_callback(result)
    return result


async def get_weather(params: FunctionCallParams):
    """
    Fetch live weather data using connection pooling and geocode caching.

    Optimizations over the original implementation:
    - Reuses HTTP session (saves ~100-200ms TCP handshake per call)
    - Caches geocoding results (skips geocode API for repeated locations)
    - 3s timeout instead of 5s (fail fast, don't block the voice pipeline)
    """
    location = params.arguments.get("location", "").strip()
    if not location:
        result = {"error": "Sorry, I could not find that location."}
        await params.result_callback(result)
        return result

    try:
        session = await _get_session()
        location_key = location.lower()

        # 1. Geocode — check cache first, then API
        if location_key in _geocode_cache:
            lat, lon, resolved_name = _geocode_cache[location_key]
        else:
            geo_url = f"{settings.geocoding_api_url}/search?name={location}&count=1"
            async with session.get(geo_url) as geo_resp:
                if geo_resp.status != 200:
                    raise Exception("Geocoding service unavailable")
                geo_data = await geo_resp.json()

            results = geo_data.get("results")
            if not results:
                result = {"error": f"Sorry, I could not find the weather for {location}."}
                await params.result_callback(result)
                return result

            lat = results[0]["latitude"]
            lon = results[0]["longitude"]
            resolved_name = results[0].get("name", location)
            _geocode_cache[location_key] = (lat, lon, resolved_name)

        # 2. Fetch current weather
        weather_url = f"{settings.weather_api_url}/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        async with session.get(weather_url) as weather_resp:
            if weather_resp.status != 200:
                raise Exception("Weather service unavailable")
            weather_data = await weather_resp.json()

        current = weather_data.get("current_weather")
        if not current:
            raise Exception("No weather data returned")

        result = {
            "location": resolved_name,
            "temperature": f"{current.get('temperature')}°C",
            "condition": _WEATHER_CODES.get(current.get("weathercode", 0), "Unknown"),
            "wind_speed": f"{current.get('windspeed')} km/h",
        }

    except Exception:
        result = {"error": f"Sorry, I was unable to fetch the weather for {location} right now."}

    await params.result_callback(result)
    return result


async def search_knowledge_base_tool(params: FunctionCallParams):
    """Search the Stelar Interior knowledge base using the inverted index."""
    query = params.arguments.get("query", "")
    results = search_knowledge(query)

    if results:
        result = {
            "found": True,
            "answers": results,
            "source": "Stelar Interior knowledge base",
        }
    else:
        result = {
            "found": False,
            "message": "No relevant information found. Use your general interior design knowledge to answer.",
        }

    await params.result_callback(result)
    return result


async def save_call_summary_tool(params: FunctionCallParams):
    """Save the call summary with all collected client details to a .txt file."""
    filename = save_summary(
        caller_name=params.arguments.get("caller_name", ""),
        phone_number=params.arguments.get("phone_number", ""),
        location=params.arguments.get("location", ""),
        work_type=params.arguments.get("work_type", ""),
        preferred_date=params.arguments.get("preferred_date", ""),
        requirements=params.arguments.get("requirements", ""),
        conversation_summary=params.arguments.get("conversation_summary", ""),
    )

    if filename:
        result = {"status": "saved", "file": filename}
    else:
        result = {"status": "error", "message": "Could not save call summary."}

    await params.result_callback(result)
    return result

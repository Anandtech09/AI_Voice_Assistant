"""
Unit tests for the Stelar Interior AI Voice Agent.

Tests individual modules in isolation with mocked dependencies.
Run with: python -X utf8 -m pytest tests/test_unit.py -v
"""

import os
import json
import shutil
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio


# ═══════════════════════════════════════════════════════════════════════
#  1. KNOWLEDGE BASE — Inverted Index Tests
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeBaseIndex:
    """Test the inverted index build and search in knowledge.py."""

    def _get_sample_kb(self):
        """Return a small sample knowledge base for testing."""
        return [
            {
                "question": "What services does Stelar Interior offer?",
                "answer": "We offer modular kitchens, wardrobes, and full home interiors.",
                "keywords": ["services", "offer", "modular kitchen"],
                "category": "Services",
            },
            {
                "question": "How much does interior design cost?",
                "answer": "Pricing depends on requirements. Free first consultation.",
                "keywords": ["pricing", "cost", "how much", "budget"],
                "category": "Pricing",
            },
            {
                "question": "How can I schedule a site visit?",
                "answer": "Call us to schedule a free site visit.",
                "keywords": ["schedule", "visit", "booking", "site visit"],
                "category": "Visit",
            },
        ]

    def test_build_index_creates_keyword_index(self):
        """Verify the inverted index maps single-word keywords correctly."""
        from app import knowledge

        kb = self._get_sample_kb()
        knowledge._build_index(kb)

        # Single-word keywords should be in the index
        assert "services" in knowledge._keyword_index
        assert "pricing" in knowledge._keyword_index
        assert "schedule" in knowledge._keyword_index

        # Each entry should have (index, weight) tuples
        services_entries = knowledge._keyword_index["services"]
        assert any(idx == 0 for idx, _ in services_entries)

    def test_build_index_creates_phrase_index(self):
        """Verify multi-word keywords are stored in the phrase index."""
        from app import knowledge

        kb = self._get_sample_kb()
        knowledge._build_index(kb)

        # "modular kitchen", "how much", "site visit" are multi-word
        phrases = [phrase for phrase, _ in knowledge._phrase_index]
        assert "modular kitchen" in phrases
        assert "how much" in phrases
        assert "site visit" in phrases

    def test_build_index_excludes_stop_words_from_questions(self):
        """Verify stop words from question text are not indexed."""
        from app import knowledge

        kb = self._get_sample_kb()
        knowledge._build_index(kb)

        # Stop words that ONLY appear in questions (not in any keyword)
        # should NOT be indexed
        assert "what" not in knowledge._keyword_index
        assert "does" not in knowledge._keyword_index
        assert "the" not in knowledge._keyword_index

    def test_search_knowledge_finds_keyword_match(self):
        """Verify search returns results for keyword matches."""
        from app import knowledge

        # Reset the module's cached data so our sample KB is used
        knowledge._knowledge_data = self._get_sample_kb()
        knowledge._build_index(knowledge._knowledge_data)

        results = knowledge.search_knowledge("What services do you offer?")

        assert len(results) > 0
        assert results[0]["category"] == "Services"

    def test_search_knowledge_phrase_match_scores_higher(self):
        """Verify multi-word phrase matches get higher scores."""
        from app import knowledge

        knowledge._knowledge_data = self._get_sample_kb()
        knowledge._build_index(knowledge._knowledge_data)

        results = knowledge.search_knowledge("I want to schedule a site visit")

        assert len(results) > 0
        # "site visit" phrase match should push Visit category to top
        assert results[0]["category"] == "Visit"

    def test_search_knowledge_returns_max_results(self):
        """Verify search respects the max_results parameter."""
        from app import knowledge

        knowledge._knowledge_data = self._get_sample_kb()
        knowledge._build_index(knowledge._knowledge_data)

        results = knowledge.search_knowledge("services pricing visit", max_results=2)

        assert len(results) <= 2

    def test_search_knowledge_empty_query_returns_empty(self):
        """Verify empty queries return no results."""
        from app import knowledge

        knowledge._knowledge_data = self._get_sample_kb()
        knowledge._build_index(knowledge._knowledge_data)

        results = knowledge.search_knowledge("")
        assert results == []

    def test_search_knowledge_no_match_returns_empty(self):
        """Verify unrelated queries return no results."""
        from app import knowledge

        knowledge._knowledge_data = self._get_sample_kb()
        knowledge._build_index(knowledge._knowledge_data)

        results = knowledge.search_knowledge("quantum physics relativity")
        assert results == []

    def test_search_result_structure(self):
        """Verify each search result has the expected keys."""
        from app import knowledge

        knowledge._knowledge_data = self._get_sample_kb()
        knowledge._build_index(knowledge._knowledge_data)

        results = knowledge.search_knowledge("services")

        assert len(results) > 0
        result = results[0]
        assert "question" in result
        assert "answer" in result
        assert "category" in result
        # Score should NOT be exposed in results
        assert "score" not in result


# ═══════════════════════════════════════════════════════════════════════
#  2. CALL SUMMARY — File Writing Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCallSummary:
    """Test the call summary writer in call_summary.py."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self, tmp_path):
        """Use a temp directory for call logs during tests."""
        self.original_dir = None
        import app.utils.call_summary as cs
        self.original_dir = cs.CALL_LOGS_DIR
        cs.CALL_LOGS_DIR = str(tmp_path / "call_logs")
        yield
        cs.CALL_LOGS_DIR = self.original_dir

    def test_save_summary_creates_file(self):
        """Verify save_summary creates a .txt file."""
        from app.utils.call_summary import save_summary, CALL_LOGS_DIR

        filename = save_summary(
            caller_name="Rahul Kumar",
            phone_number="+919876543210",
            conversation_summary="Test call",
        )

        assert filename != ""
        assert filename.startswith("call_")
        assert filename.endswith(".txt")
        assert os.path.exists(os.path.join(CALL_LOGS_DIR, filename))

    def test_save_summary_contains_client_details(self):
        """Verify the saved file contains all client details."""
        from app.utils.call_summary import save_summary, CALL_LOGS_DIR

        filename = save_summary(
            caller_name="Priya Nair",
            phone_number="+919876543210",
            location="Kakkanad, Kochi",
            work_type="Full home interior",
            preferred_date="Saturday, August 23",
            requirements="Modern kitchen with island layout",
            conversation_summary="Client wants full 3BHK interior",
        )

        filepath = os.path.join(CALL_LOGS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Priya Nair" in content
        assert "+919876543210" in content
        assert "Kakkanad, Kochi" in content
        assert "Full home interior" in content
        assert "Saturday, August 23" in content
        assert "Modern kitchen with island layout" in content
        assert "Client wants full 3BHK interior" in content
        assert "STELAR INTERIOR" in content
        assert "ACTION REQUIRED" in content

    def test_save_summary_handles_missing_details(self):
        """Verify missing fields show 'Not provided' in the file."""
        from app.utils.call_summary import save_summary, CALL_LOGS_DIR

        filename = save_summary(conversation_summary="Brief off-topic call")

        filepath = os.path.join(CALL_LOGS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Not provided" in content
        assert "Brief off-topic call" in content

    def test_save_summary_creates_directory(self):
        """Verify save_summary creates the call_logs dir if it doesn't exist."""
        from app.utils.call_summary import save_summary, CALL_LOGS_DIR

        # Directory shouldn't exist yet (autouse fixture uses tmp_path)
        assert not os.path.exists(CALL_LOGS_DIR)

        save_summary(conversation_summary="Test")

        assert os.path.exists(CALL_LOGS_DIR)

    def test_save_summary_unique_filenames(self):
        """Verify two calls produce unique filenames (timestamp-based)."""
        from app.utils.call_summary import save_summary
        import time

        filename1 = save_summary(conversation_summary="Call 1")
        time.sleep(1.1)  # Ensure different second
        filename2 = save_summary(conversation_summary="Call 2")

        assert filename1 != filename2


# ═══════════════════════════════════════════════════════════════════════
#  3. TOOLS — Tool Definition Schema Tests
# ═══════════════════════════════════════════════════════════════════════

class TestToolDefinitions:
    """Test tool definition schemas are valid for Gemini."""

    def test_tool_definitions_has_function_declarations(self):
        """Verify TOOL_DEFINITIONS has the required structure."""
        from app.tools import TOOL_DEFINITIONS

        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) == 1
        assert "function_declarations" in TOOL_DEFINITIONS[0]

    def test_all_four_tools_defined(self):
        """Verify all 4 tools are defined."""
        from app.tools import TOOL_DEFINITIONS

        declarations = TOOL_DEFINITIONS[0]["function_declarations"]
        tool_names = [d["name"] for d in declarations]

        assert "get_current_datetime" in tool_names
        assert "get_weather" in tool_names
        assert "search_knowledge_base" in tool_names
        assert "save_call_summary" in tool_names
        assert len(tool_names) == 4

    def test_each_tool_has_required_fields(self):
        """Verify each tool has name, description, and parameters."""
        from app.tools import TOOL_DEFINITIONS

        for decl in TOOL_DEFINITIONS[0]["function_declarations"]:
            assert "name" in decl, f"Tool missing 'name'"
            assert "description" in decl, f"Tool '{decl.get('name')}' missing 'description'"
            assert "parameters" in decl, f"Tool '{decl.get('name')}' missing 'parameters'"
            assert decl["parameters"]["type"] == "object"

    def test_weather_tool_requires_location(self):
        """Verify get_weather has 'location' as a required parameter."""
        from app.tools import TOOL_DEFINITIONS

        weather = next(
            d for d in TOOL_DEFINITIONS[0]["function_declarations"]
            if d["name"] == "get_weather"
        )
        assert "location" in weather["parameters"]["required"]

    def test_save_call_summary_requires_conversation_summary(self):
        """Verify save_call_summary requires 'conversation_summary'."""
        from app.tools import TOOL_DEFINITIONS

        summary_tool = next(
            d for d in TOOL_DEFINITIONS[0]["function_declarations"]
            if d["name"] == "save_call_summary"
        )
        assert "conversation_summary" in summary_tool["parameters"]["required"]

    def test_save_call_summary_has_all_client_fields(self):
        """Verify save_call_summary accepts all client detail fields."""
        from app.tools import TOOL_DEFINITIONS

        summary_tool = next(
            d for d in TOOL_DEFINITIONS[0]["function_declarations"]
            if d["name"] == "save_call_summary"
        )
        props = summary_tool["parameters"]["properties"]

        expected_fields = [
            "caller_name", "phone_number", "location",
            "work_type", "preferred_date", "requirements",
            "conversation_summary",
        ]
        for field in expected_fields:
            assert field in props, f"Missing field '{field}' in save_call_summary"


# ═══════════════════════════════════════════════════════════════════════
#  4. TOOLS — Datetime Tool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDatetimeTool:
    """Test the get_current_datetime tool handler."""

    @pytest.mark.asyncio
    async def test_get_current_datetime_returns_result(self):
        """Verify get_current_datetime returns date, time, and day."""
        from app.tools import get_current_datetime

        mock_params = MagicMock()
        mock_params.arguments = {}
        mock_params.result_callback = AsyncMock()

        result = await get_current_datetime(mock_params)

        assert "date" in result
        assert "time" in result
        assert "day" in result
        assert "full" in result
        mock_params.result_callback.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_get_current_datetime_format(self):
        """Verify datetime result has human-readable format."""
        from app.tools import get_current_datetime

        mock_params = MagicMock()
        mock_params.arguments = {"timezone": "Asia/Kolkata"}
        mock_params.result_callback = AsyncMock()

        result = await get_current_datetime(mock_params)

        # Date should have month name (e.g., "August 15, 2026")
        assert any(month in result["date"] for month in [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        # Time should have AM/PM
        assert "AM" in result["time"] or "PM" in result["time"]


# ═══════════════════════════════════════════════════════════════════════
#  5. TOOLS — Weather Tool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestWeatherTool:
    """Test the get_weather tool handler with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_weather_empty_location_returns_error(self):
        """Verify empty location returns an error result."""
        from app.tools import get_weather

        mock_params = MagicMock()
        mock_params.arguments = {"location": ""}
        mock_params.result_callback = AsyncMock()

        result = await get_weather(mock_params)

        assert "error" in result
        mock_params.result_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_weather_missing_location_returns_error(self):
        """Verify missing location parameter returns an error result."""
        from app.tools import get_weather

        mock_params = MagicMock()
        mock_params.arguments = {}
        mock_params.result_callback = AsyncMock()

        result = await get_weather(mock_params)

        assert "error" in result

    def test_weather_codes_complete(self):
        """Verify all common WMO weather codes are mapped."""
        from app.tools import _WEATHER_CODES

        # Key codes that must be present
        assert _WEATHER_CODES[0] == "Clear sky"
        assert _WEATHER_CODES[3] == "Overcast"
        assert _WEATHER_CODES[61] == "Slight rain"
        assert _WEATHER_CODES[95] == "Thunderstorm"
        assert len(_WEATHER_CODES) >= 18

    def test_geocode_cache_is_dict(self):
        """Verify geocode cache data structure exists."""
        from app.tools import _geocode_cache
        assert isinstance(_geocode_cache, dict)


# ═══════════════════════════════════════════════════════════════════════
#  6. TOOLS — Knowledge Base Tool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeBaseTool:
    """Test the search_knowledge_base_tool handler."""

    @pytest.mark.asyncio
    async def test_kb_tool_found(self):
        """Verify KB tool returns found=True when results exist."""
        from app.tools import search_knowledge_base_tool

        mock_params = MagicMock()
        mock_params.arguments = {"query": "services"}
        mock_params.result_callback = AsyncMock()

        with patch("app.tools.search_knowledge", return_value=[
            {"question": "Q1", "answer": "A1", "category": "Services"}
        ]):
            result = await search_knowledge_base_tool(mock_params)

        assert result["found"] is True
        assert len(result["answers"]) == 1
        assert result["source"] == "Stelar Interior knowledge base"

    @pytest.mark.asyncio
    async def test_kb_tool_not_found(self):
        """Verify KB tool returns found=False when no results."""
        from app.tools import search_knowledge_base_tool

        mock_params = MagicMock()
        mock_params.arguments = {"query": "xyz123"}
        mock_params.result_callback = AsyncMock()

        with patch("app.tools.search_knowledge", return_value=[]):
            result = await search_knowledge_base_tool(mock_params)

        assert result["found"] is False
        assert "message" in result


# ═══════════════════════════════════════════════════════════════════════
#  7. TOOLS — Call Summary Tool Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCallSummaryTool:
    """Test the save_call_summary_tool handler."""

    @pytest.mark.asyncio
    async def test_summary_tool_success(self):
        """Verify save_call_summary_tool returns saved status on success."""
        from app.tools import save_call_summary_tool

        mock_params = MagicMock()
        mock_params.arguments = {
            "caller_name": "Test User",
            "conversation_summary": "Test conversation",
        }
        mock_params.result_callback = AsyncMock()

        with patch("app.tools.save_summary", return_value="call_2026-08-15.txt"):
            result = await save_call_summary_tool(mock_params)

        assert result["status"] == "saved"
        assert result["file"] == "call_2026-08-15.txt"

    @pytest.mark.asyncio
    async def test_summary_tool_failure(self):
        """Verify save_call_summary_tool returns error on failure."""
        from app.tools import save_call_summary_tool

        mock_params = MagicMock()
        mock_params.arguments = {"conversation_summary": "Test"}
        mock_params.result_callback = AsyncMock()

        with patch("app.tools.save_summary", return_value=""):
            result = await save_call_summary_tool(mock_params)

        assert result["status"] == "error"


# ═══════════════════════════════════════════════════════════════════════
#  8. CONFIG — Settings Tests
# ═══════════════════════════════════════════════════════════════════════

class TestConfig:
    """Test configuration and settings."""

    def test_settings_websocket_url(self):
        """Verify websocket_url converts https to wss."""
        from app.config import Settings

        s = Settings(
            gemini_api_key="test",
            twilio_account_sid="test",
            twilio_auth_token="test",
            twilio_phone_number="+1234",
            your_phone_number="+5678",
            ngrok_url="https://example.ngrok-free.app",
        )
        assert s.websocket_url == "wss://example.ngrok-free.app/ws"

    def test_settings_twiml_url(self):
        """Verify twiml_url appends /twiml to ngrok URL."""
        from app.config import Settings

        s = Settings(
            gemini_api_key="test",
            twilio_account_sid="test",
            twilio_auth_token="test",
            twilio_phone_number="+1234",
            your_phone_number="+5678",
            ngrok_url="https://example.ngrok-free.app",
        )
        assert s.twiml_url == "https://example.ngrok-free.app/twiml"

    def test_settings_default_port(self):
        """Verify default port is 8000."""
        from app.config import Settings

        s = Settings(
            gemini_api_key="test",
            twilio_account_sid="test",
            twilio_auth_token="test",
            twilio_phone_number="+1234",
            your_phone_number="+5678",
            ngrok_url="https://example.ngrok-free.app",
        )
        assert s.port == 8000
        assert s.host == "0.0.0.0"

    def test_settings_default_api_urls(self):
        """Verify default weather and geocoding API URLs."""
        from app.config import Settings

        s = Settings(
            gemini_api_key="test",
            twilio_account_sid="test",
            twilio_auth_token="test",
            twilio_phone_number="+1234",
            your_phone_number="+5678",
            ngrok_url="https://test.ngrok.app",
        )
        assert "open-meteo.com" in s.weather_api_url
        assert "geocoding-api.open-meteo.com" in s.geocoding_api_url


# ═══════════════════════════════════════════════════════════════════════
#  9. PROMPTS — System Prompt Content Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPrompts:
    """Test system prompt contains critical instructions."""

    def test_prompt_mentions_stelar_interior(self):
        """Verify system prompt references Stelar Interior."""
        from app.prompts import SYSTEM_PROMPT
        assert "Stelar Interior" in SYSTEM_PROMPT

    def test_prompt_has_no_pricing_policy(self):
        """Verify system prompt forbids sharing pricing."""
        from app.prompts import SYSTEM_PROMPT
        assert "pricing" in SYSTEM_PROMPT.lower() or "price" in SYSTEM_PROMPT.lower()
        assert "site visit" in SYSTEM_PROMPT.lower()

    def test_prompt_has_off_topic_handling(self):
        """Verify system prompt includes off-topic redirect rules."""
        from app.prompts import SYSTEM_PROMPT
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "off-topic" in prompt_lower or "off topic" in prompt_lower

    def test_prompt_has_call_summary_instruction(self):
        """Verify system prompt instructs AI to save call summary."""
        from app.prompts import SYSTEM_PROMPT
        assert "save_call_summary" in SYSTEM_PROMPT

    def test_prompt_has_visit_scheduling(self):
        """Verify system prompt covers visit scheduling."""
        from app.prompts import SYSTEM_PROMPT
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "visit" in prompt_lower
        assert "team" in prompt_lower

    def test_prompt_has_bilingual_support(self):
        """Verify system prompt mentions English and Malayalam."""
        from app.prompts import SYSTEM_PROMPT
        assert "English" in SYSTEM_PROMPT
        assert "Malayalam" in SYSTEM_PROMPT

    def test_prompt_is_compact(self):
        """Verify the prompt is compact for low latency (under 1500 chars)."""
        from app.prompts import SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) < 1500, f"Prompt is {len(SYSTEM_PROMPT)} chars — too long for low latency"

    def test_greeting_mentions_stelar(self):
        """Verify greeting references Stelar Interior."""
        from app.prompts import GREETING
        assert "Stelar Interior" in GREETING

    def test_greeting_malayalam_exists(self):
        """Verify Malayalam greeting exists and is non-empty."""
        from app.prompts import GREETING_MALAYALAM
        assert len(GREETING_MALAYALAM) > 0
        # Should contain Malayalam unicode characters
        assert "സ്റ്റെലാർ" in GREETING_MALAYALAM


# ═══════════════════════════════════════════════════════════════════════
#  10. KNOWLEDGE BASE JSON — Data Integrity Tests
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeBaseJSON:
    """Test the knowledge.json file structure and content."""

    @pytest.fixture(autouse=True)
    def load_kb(self):
        """Load the knowledge base JSON for testing."""
        kb_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "knowledge", "knowledge.json"
        )
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

    def test_kb_is_non_empty(self):
        """Verify knowledge base has entries."""
        assert len(self.kb) >= 15

    def test_each_entry_has_required_fields(self):
        """Verify each KB entry has question, answer, keywords, category."""
        for i, entry in enumerate(self.kb):
            assert "question" in entry, f"Entry {i} missing 'question'"
            assert "answer" in entry, f"Entry {i} missing 'answer'"
            assert "keywords" in entry, f"Entry {i} missing 'keywords'"
            assert "category" in entry, f"Entry {i} missing 'category'"

    def test_keywords_are_non_empty_lists(self):
        """Verify each entry has at least one keyword."""
        for i, entry in enumerate(self.kb):
            assert isinstance(entry["keywords"], list), f"Entry {i}: keywords should be a list"
            assert len(entry["keywords"]) > 0, f"Entry {i}: keywords should not be empty"

    def test_no_technova_references(self):
        """Verify knowledge base has zero TechNova references."""
        kb_str = json.dumps(self.kb).lower()
        assert "technova" not in kb_str

    def test_has_stelar_interior_references(self):
        """Verify knowledge base mentions Stelar Interior."""
        kb_str = json.dumps(self.kb)
        assert "Stelar Interior" in kb_str

    def test_pricing_entry_redirects_to_visit(self):
        """Verify pricing entry does NOT share specific prices."""
        pricing_entries = [e for e in self.kb if e["category"] == "Pricing"]
        assert len(pricing_entries) > 0

        for entry in pricing_entries:
            answer_lower = entry["answer"].lower()
            # Should mention site visit or consultation
            assert "visit" in answer_lower or "consultation" in answer_lower
            # Should NOT contain currency symbols or specific amounts
            assert "₹" not in entry["answer"]
            assert "$" not in entry["answer"]

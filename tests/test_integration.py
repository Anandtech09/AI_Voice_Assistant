"""
Integration tests for the Stelar Interior AI Voice Agent.

Tests components working together: FastAPI endpoints, knowledge base
end-to-end search, tool schema + handler integration, and call log pipeline.
Run with: python -X utf8 -m pytest tests/test_integration.py -v
"""

import os
import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app


# ═══════════════════════════════════════════════════════════════════════
#  1. FASTAPI ENDPOINT INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestHealthCheckEndpoint:
    """Test the GET / health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self):
        """Verify health check returns 200 OK."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_response_body(self):
        """Verify health check response contains expected fields."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
        assert "message" in data


class TestTwimlEndpoint:
    """Test the /twiml endpoint for TwiML generation."""

    @pytest.mark.asyncio
    async def test_twiml_get_returns_xml(self):
        """Verify GET /twiml returns XML content type."""
        with patch("app.main.generate_twiml_response", return_value="<Response><Connect><Stream/></Connect></Response>"):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/twiml")

        assert response.status_code == 200
        assert "xml" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_twiml_post_returns_xml(self):
        """Verify POST /twiml returns XML content type."""
        with patch("app.main.generate_twiml_response", return_value="<Response><Connect><Stream/></Connect></Response>"):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/twiml")

        assert response.status_code == 200
        assert "xml" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_twiml_contains_stream(self):
        """Verify TwiML response includes a Stream element."""
        with patch("app.main.generate_twiml_response", return_value='<Response><Connect><Stream url="wss://test/ws"/></Connect></Response>'):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/twiml")

        assert "Stream" in response.text


class TestCallStatusEndpoint:
    """Test the POST /call-status callback endpoint."""

    @pytest.mark.asyncio
    async def test_call_status_completed(self):
        """Verify completed call status returns OK."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/call-status",
                data={
                    "CallSid": "CA12345",
                    "CallStatus": "completed",
                    "CallDuration": "45",
                },
            )

        assert response.status_code == 200
        assert response.text == "OK"

    @pytest.mark.asyncio
    async def test_call_status_failed(self):
        """Verify failed call status returns OK (logs the error)."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/call-status",
                data={
                    "CallSid": "CA12345",
                    "CallStatus": "failed",
                },
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_call_status_ringing(self):
        """Verify in-progress call status returns OK."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/call-status",
                data={
                    "CallSid": "CA12345",
                    "CallStatus": "ringing",
                },
            )

        assert response.status_code == 200


class TestStartCallEndpoint:
    """Test the POST /start-call endpoint with mocked Twilio."""

    @pytest.mark.asyncio
    async def test_start_call_success(self):
        """Verify start-call returns success when Twilio call succeeds."""
        with patch("app.main.make_outbound_call", return_value="CA_TEST_SID_123"):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/start-call")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["call_sid"] == "CA_TEST_SID_123"

    @pytest.mark.asyncio
    async def test_start_call_with_custom_number(self):
        """Verify start-call passes custom number to Twilio."""
        with patch("app.main.make_outbound_call", return_value="CA_CUSTOM_123") as mock_call:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/start-call",
                    json={"to_number": "+919999999999"},
                )

        assert response.status_code == 200
        mock_call.assert_called_once_with("+919999999999")

    @pytest.mark.asyncio
    async def test_start_call_failure(self):
        """Verify start-call returns 500 when Twilio raises an error."""
        with patch("app.main.make_outbound_call", side_effect=Exception("Twilio error")):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/start-call")

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"


class TestBulkCallEndpoint:
    """Test the POST /bulk-call endpoint for multiple outbound calls."""

    @pytest.mark.asyncio
    async def test_bulk_call_success(self):
        """Verify bulk-call initiates calls sequentially."""
        with patch("app.main.make_outbound_call", side_effect=["SID_1", "SID_2"]):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/bulk-call",
                    json={
                        "numbers": ["+919876543210", "+919876543211"],
                        "delay_seconds": 0,
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_call_invalid_input(self):
        """Verify bulk-call returns 400 when numbers list is missing or invalid format."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/bulk-call",
                json={"numbers": "not_a_list"},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"



# ═══════════════════════════════════════════════════════════════════════
#  2. KNOWLEDGE BASE END-TO-END INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeBaseEndToEnd:
    """Test loading the real knowledge.json and searching it."""

    @pytest.fixture(autouse=True)
    def reset_kb(self):
        """Reset knowledge base cache before each test."""
        from app import knowledge
        knowledge._knowledge_data = []
        knowledge._keyword_index = {}
        knowledge._phrase_index = []
        knowledge._index_built = False
        yield

    def test_load_real_knowledge_base(self):
        """Verify the actual knowledge.json loads without errors."""
        from app.knowledge import load_knowledge_base

        kb = load_knowledge_base()

        assert len(kb) >= 15
        assert all("question" in entry for entry in kb)
        assert all("answer" in entry for entry in kb)

    def test_search_services_returns_relevant(self):
        """Verify searching 'services' returns service-related entries."""
        from app.knowledge import search_knowledge

        results = search_knowledge("What services do you offer?")

        assert len(results) > 0
        categories = [r["category"] for r in results]
        assert "Services" in categories

    def test_search_modular_kitchen_phrase(self):
        """Verify multi-word phrase 'modular kitchen' matches correctly."""
        from app.knowledge import search_knowledge

        results = search_knowledge("Do you do modular kitchen design?")

        assert len(results) > 0
        # Should find the modular kitchen entry
        answers_text = " ".join(r["answer"] for r in results).lower()
        assert "kitchen" in answers_text

    def test_search_pricing_returns_no_prices(self):
        """Verify pricing search results don't contain actual prices."""
        from app.knowledge import search_knowledge

        results = search_knowledge("How much does it cost?")

        assert len(results) > 0
        for result in results:
            assert "₹" not in result["answer"]
            assert "$" not in result["answer"]

    def test_search_visit_scheduling(self):
        """Verify visit-related searches return Visit category."""
        from app.knowledge import search_knowledge

        results = search_knowledge("I want to book a site visit")

        assert len(results) > 0
        categories = [r["category"] for r in results]
        assert "Visit" in categories

    def test_search_warranty(self):
        """Verify warranty search finds the warranty entry."""
        from app.knowledge import search_knowledge

        results = search_knowledge("Do you provide warranty?")

        assert len(results) > 0
        answers_text = " ".join(r["answer"] for r in results).lower()
        assert "warranty" in answers_text

    def test_search_materials(self):
        """Verify materials search returns material info."""
        from app.knowledge import search_knowledge

        results = search_knowledge("What materials do you use?")

        assert len(results) > 0
        answers_text = " ".join(r["answer"] for r in results).lower()
        assert "plywood" in answers_text or "material" in answers_text

    def test_search_gibberish_returns_empty(self):
        """Verify nonsense queries return no results."""
        from app.knowledge import search_knowledge

        results = search_knowledge("xyzzy frobnicator quantum")
        assert results == []


# ═══════════════════════════════════════════════════════════════════════
#  3. CALL SUMMARY END-TO-END PIPELINE TEST
# ═══════════════════════════════════════════════════════════════════════

class TestCallSummaryPipeline:
    """Test the full pipeline: tool handler → file writer → .txt on disk."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self, tmp_path):
        """Use temp directory for call logs."""
        import app.utils.call_summary as cs
        self.original_dir = cs.CALL_LOGS_DIR
        cs.CALL_LOGS_DIR = str(tmp_path / "call_logs")
        self.log_dir = cs.CALL_LOGS_DIR
        yield
        cs.CALL_LOGS_DIR = self.original_dir

    @pytest.mark.asyncio
    async def test_full_pipeline_tool_to_file(self):
        """Test the complete flow: tool handler → save_summary → .txt file."""
        from app.tools import save_call_summary_tool

        mock_params = MagicMock()
        mock_params.arguments = {
            "caller_name": "Integration Test User",
            "phone_number": "+919876543210",
            "location": "Kakkanad, Kochi",
            "work_type": "Modular Kitchen",
            "preferred_date": "Next Saturday",
            "requirements": "L-shaped kitchen with island",
            "conversation_summary": "Client called about modular kitchen for new apartment.",
        }
        mock_params.result_callback = AsyncMock()

        result = await save_call_summary_tool(mock_params)

        # Verify the tool handler returned success
        assert result["status"] == "saved"
        assert result["file"].endswith(".txt")

        # Verify the file actually exists on disk
        filepath = os.path.join(self.log_dir, result["file"])
        assert os.path.exists(filepath)

        # Verify file contents
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Integration Test User" in content
        assert "+919876543210" in content
        assert "Kakkanad, Kochi" in content
        assert "Modular Kitchen" in content
        assert "Next Saturday" in content
        assert "L-shaped kitchen with island" in content
        assert "STELAR INTERIOR" in content

    @pytest.mark.asyncio
    async def test_pipeline_with_minimal_data(self):
        """Test the pipeline with only required field (conversation_summary)."""
        from app.tools import save_call_summary_tool

        mock_params = MagicMock()
        mock_params.arguments = {
            "conversation_summary": "Caller was off-topic, did not discuss interior design.",
        }
        mock_params.result_callback = AsyncMock()

        result = await save_call_summary_tool(mock_params)

        assert result["status"] == "saved"

        filepath = os.path.join(self.log_dir, result["file"])
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Not provided" in content
        assert "off-topic" in content


# ═══════════════════════════════════════════════════════════════════════
#  4. TOOL SCHEMA + HANDLER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

class TestToolSchemaHandlerIntegration:
    """Verify tool schemas match their actual handler implementations."""

    def test_all_declared_tools_have_handlers(self):
        """Verify every tool in TOOL_DEFINITIONS has a matching handler in bot.py."""
        from app.tools import TOOL_DEFINITIONS

        declared_names = [
            d["name"]
            for d in TOOL_DEFINITIONS[0]["function_declarations"]
        ]

        # These are the handler function names in tools.py
        from app import tools
        handler_map = {
            "get_current_datetime": tools.get_current_datetime,
            "get_weather": tools.get_weather,
            "search_knowledge_base": tools.search_knowledge_base_tool,
            "save_call_summary": tools.save_call_summary_tool,
        }

        for name in declared_names:
            assert name in handler_map, f"Tool '{name}' declared but no handler found"
            assert callable(handler_map[name]), f"Handler for '{name}' is not callable"

    def test_tool_parameter_types_are_valid(self):
        """Verify all tool parameter types are valid JSON Schema types."""
        from app.tools import TOOL_DEFINITIONS

        valid_types = {"string", "number", "integer", "boolean", "array", "object"}

        for decl in TOOL_DEFINITIONS[0]["function_declarations"]:
            assert decl["parameters"]["type"] in valid_types

            for prop_name, prop_def in decl["parameters"].get("properties", {}).items():
                assert prop_def["type"] in valid_types, (
                    f"Tool '{decl['name']}', param '{prop_name}' has invalid type: {prop_def['type']}"
                )

    def test_required_params_are_subset_of_properties(self):
        """Verify required params are actually defined in properties."""
        from app.tools import TOOL_DEFINITIONS

        for decl in TOOL_DEFINITIONS[0]["function_declarations"]:
            props = set(decl["parameters"].get("properties", {}).keys())
            required = set(decl["parameters"].get("required", []))

            assert required.issubset(props), (
                f"Tool '{decl['name']}': required params {required - props} not in properties"
            )

"""
Pipecat pipeline setup for the Stelar Interior AI Voice Agent.

Configures the voice agent pipeline with:
- Twilio WebSocket transport (audio I/O)
- Gemini Live LLM service (speech-to-speech AI with echo proofing)
- Tool/function registration (datetime, weather, KB, call summary)
- Automatic call summary on disconnect
- Real user barge-in support without echo cut-offs
"""

from google.genai.types import ThinkingConfig

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.google.gemini_live import GeminiLiveLLMService
from pipecat.services.google.gemini_live.llm import GeminiVADParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.services.llm_service import FunctionCallParams

from app.config import settings
from app.prompts import SYSTEM_PROMPT
from app.tools import (
    get_current_datetime,
    get_weather,
    search_knowledge_base_tool,
    save_call_summary_tool,
    TOOL_DEFINITIONS,
)
from app.utils.call_summary import save_summary, extract_transcript_from_messages
from app.utils.logger import (
    logger,
    log_call_connected,
    log_call_ended,
    log_pipeline_event,
    log_tool_invoked,
    log_tool_result,
)
from app.processors.audio_gate import AudioGateProcessor
from app.processors.idle_monitor import IdleMonitorProcessor


class EchoProofGeminiLLMService(GeminiLiveLLMService):
    """
    Subclass of GeminiLiveLLMService that suppresses server-side VAD echo interruptions
    unless local Silero VAD confirms that the user is actively speaking.
    Prevents false voice breaks, sentence cut-offs, and delayed response bugs.
    """
    def __init__(self, *args, user_aggregator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._user_aggregator = user_aggregator

    async def broadcast_interruption(self):
        if self._user_aggregator:
            is_user_speaking = getattr(self._user_aggregator, "_user_speaking", False)
            if not is_user_speaking:
                logger.debug("🛡️ Suppressed false Gemini server VAD echo interruption (user is not speaking)")
                return
        await super().broadcast_interruption()


# Track whether summary was saved during the call (via Gemini tool call)
_summary_saved_for_call = {}


async def run_bot(websocket, stream_sid: str = None, call_sid: str = None):
    """
    Set up and run the Pipecat voice agent pipeline.

    Pipeline: Twilio Audio In → VAD → EchoProof Gemini Live → Twilio Audio Out
    """

    log_pipeline_event("INITIALIZING", f"Stream SID: {stream_sid}")

    # Reset summary tracking for this call
    call_key = call_sid or stream_sid or "unknown"
    _summary_saved_for_call[call_key] = False

    # Collect details during the call for fallback summary
    collected_details = {
        "caller_name": "",
        "phone_number": "",
        "location": "",
        "work_type": "",
        "preferred_date": "",
        "requirements": "",
        "conversation_summary": "",
    }

    # 1. Configure Twilio Transport
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
        ),
    )

    # 2. Setup Conversation Context & VAD
    messages = [
        {
            "role": "user",
            "content": "Welcome the caller to Stelar Interior. Tell them we offer a free site visit, and ask how you can help with their interior design today.",
        }
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    start_secs=0.2,
                    stop_secs=0.2,
                    confidence=0.5,
                    min_volume=0.1,
                )
            ),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=1.7)],
            ),
            user_turn_stop_timeout=None,
        ),
    )

    # 3. Configure Echo-Proof Gemini Live LLM Service
    llm = EchoProofGeminiLLMService(
        api_key=settings.gemini_api_key,
        user_aggregator=user_aggregator,
        settings=GeminiLiveLLMService.Settings(
            model="models/gemini-2.5-flash-native-audio-latest",
            voice="Puck",
            system_instruction=SYSTEM_PROMPT,
            thinking=ThinkingConfig(thinking_budget=0),
            vad=GeminiVADParams(disabled=False),
        ),
        tools=TOOL_DEFINITIONS,
    )

    # 4. Register Tool Handlers
    async def handle_get_datetime(params: FunctionCallParams):
        """Handle get_current_datetime tool call."""
        log_tool_invoked("get_current_datetime", params.arguments)
        result = await get_current_datetime(params)
        log_tool_result("get_current_datetime", result)

    async def handle_get_weather(params: FunctionCallParams):
        """Handle get_weather tool call."""
        log_tool_invoked("get_weather", params.arguments)
        result = await get_weather(params)
        log_tool_result("get_weather", result)

    async def handle_search_kb(params: FunctionCallParams):
        """Handle search_knowledge_base tool call."""
        log_tool_invoked("search_knowledge_base", params.arguments)
        result = await search_knowledge_base_tool(params)
        log_tool_result("search_knowledge_base", result)

    async def handle_save_summary(params: FunctionCallParams):
        """Handle save_call_summary tool call — writes .txt to call_logs/ with transcript."""
        log_tool_invoked("save_call_summary", params.arguments)

        # Extract full transcript from LLMContext
        transcript = extract_transcript_from_messages(context.messages)

        # Pass transcript into save_summary
        kwargs = dict(params.arguments)
        kwargs["transcript"] = transcript
        kwargs["call_sid"] = call_key
        
        # Run in a background thread to prevent the 45-second LLM extraction from blocking Pipecat
        loop = asyncio.get_running_loop()
        def bg_save():
            try:
                save_summary(**kwargs)
            except Exception as e:
                logger.error(f"Error saving summary in background: {e}")
                
        loop.run_in_executor(None, bg_save)

        log_tool_result("save_call_summary", {"status": "saving_in_background"})

        # Track that summary was saved via Gemini tool call
        _summary_saved_for_call[call_key] = True

        # Capture details for logging
        for key in collected_details:
            if key in params.arguments and params.arguments[key]:
                collected_details[key] = params.arguments[key]

    llm.register_function("get_current_datetime", handle_get_datetime)
    llm.register_function("get_weather", handle_get_weather)
    llm.register_function("search_knowledge_base", handle_search_kb)
    llm.register_function("save_call_summary", handle_save_summary)

    # 5. Build the Pipeline
    audio_gate = AudioGateProcessor()
    idle_monitor = IdleMonitorProcessor(timeout_seconds=20.0)
    
    pipeline = Pipeline(
        [
            transport.input(),
            audio_gate,
            idle_monitor,
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # 6. Create the Pipeline Task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    # 7. Event Handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_instance, client):
        log_call_connected(call_sid or "unknown")
        log_pipeline_event("CLIENT CONNECTED", "Audio streaming started")
        await idle_monitor.start_monitor(task)
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_instance, client):
        await idle_monitor.stop_monitor()
        log_call_ended(call_sid or "unknown")
        log_pipeline_event("CLIENT DISCONNECTED", "Audio streaming ended")

        # Save a fallback summary with full transcript if Gemini didn't call save_call_summary
        if not _summary_saved_for_call.get(call_key, False):
            logger.info("📝 Gemini didn't save summary — writing fallback summary with full transcript on disconnect")
            transcript = extract_transcript_from_messages(context.messages)

            save_summary(
                caller_name=collected_details.get("caller_name", ""),
                phone_number=collected_details.get("phone_number", ""),
                location=collected_details.get("location", ""),
                work_type=collected_details.get("work_type", ""),
                preferred_date=collected_details.get("preferred_date", ""),
                requirements=collected_details.get("requirements", ""),
                conversation_summary=f"Call ended (SID: {call_key}). Full transcript attached below.",
                transcript=transcript,
                call_sid=call_key,
            )

        # Cleanup tracking
        _summary_saved_for_call.pop(call_key, None)

        await task.cancel()

    # 8. Run the Pipeline
    log_pipeline_event("STARTING", "Pipeline is now running")

    runner = PipelineRunner()
    await runner.run(task)

    log_pipeline_event("STOPPED", "Pipeline has stopped")

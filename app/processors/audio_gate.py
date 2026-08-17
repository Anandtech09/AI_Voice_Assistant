from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from app.utils.logger import logger

class AudioGateProcessor(FrameProcessor):
    """
    A half-duplex audio gate that drops incoming user audio 
    while the bot is speaking. This prevents the bot's own TTS output 
    (echoing through the phone) from reaching the VAD or LLM, solving 
    false interruptions and LLM queue latency loops.
    """
    def __init__(self):
        super().__init__()
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track bot speaking state (these frames usually flow UPSTREAM from the output transport)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            logger.debug("🔒 Audio Gate Closed (Bot started speaking)")
            await self.push_frame(frame, direction)
            return

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            logger.debug("🔓 Audio Gate Opened (Bot stopped speaking)")
            await self.push_frame(frame, direction)
            return

        # Intercept audio coming DOWNSTREAM from the input transport
        elif isinstance(frame, InputAudioRawFrame):
            if self._bot_speaking:
                # Drop the audio frame to prevent echo loop
                return
            else:
                # Forward audio if bot is silent
                await self.push_frame(frame, direction)
                return

        # Forward all other frames normally
        await self.push_frame(frame, direction)

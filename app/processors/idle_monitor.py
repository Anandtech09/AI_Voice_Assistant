import asyncio
import time
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    VADUserStartedSpeakingFrame,
    BotStartedSpeakingFrame,
    InputTextRawFrame
)
from app.utils.logger import logger

class IdleMonitorProcessor(FrameProcessor):
    """
    Monitors conversation activity. If neither the user nor the bot
    speaks for `timeout_seconds`, it injects a silent text prompt
    to the LLM instructing it to engage the user.
    """
    def __init__(self, timeout_seconds: float = 30.0):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self._last_activity = time.time()
        self._task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (VADUserStartedSpeakingFrame, BotStartedSpeakingFrame)):
            self._last_activity = time.time()

    async def start_monitor(self, pipeline_task):
        """Start the background monitoring loop."""
        self._last_activity = time.time()
        self._task = asyncio.create_task(self._monitor_loop(pipeline_task))

    async def stop_monitor(self):
        """Stop the background monitoring loop."""
        if self._task:
            self._task.cancel()
            self._task = None

    async def _monitor_loop(self, pipeline_task):
        try:
            while True:
                await asyncio.sleep(2)
                if time.time() - self._last_activity > self.timeout_seconds:
                    logger.info(f"⏳ User idle for {self.timeout_seconds}s. Engaging...")
                    self._last_activity = time.time()  # Reset to prevent spam
                    
                    # Inject a text frame to prompt the LLM to speak
                    await pipeline_task.queue_frames([
                        InputTextRawFrame("The caller has been silent for 20 seconds. Ask if they are still there and if they need any more help.")
                    ])
        except asyncio.CancelledError:
            pass

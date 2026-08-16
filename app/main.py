"""
FastAPI server for the Stelar Interior AI Voice Agent.

Provides HTTP and WebSocket endpoints:
- GET  /            → Health check
- POST /start-call  → Initiate a single outbound call via Twilio
- POST /bulk-call   → Initiate outbound calls to multiple numbers
- GET|POST /twiml   → TwiML webhook (inbound & outbound calls connect here)
- POST /call-status → Receive call status updates from Twilio
- WS   /ws          → Twilio Media Stream WebSocket (the AI pipeline runs here)

INBOUND calls: Set your Twilio number's Voice webhook to https://<ngrok-url>/twiml
OUTBOUND calls: Hit POST /start-call or POST /bulk-call
"""

import asyncio
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import settings
from app.bot import run_bot
from app.services.twilio_service import make_outbound_call, generate_twiml_response
from app.utils.logger import (
    logger,
    log_call_ended,
    log_call_error,
)

app = FastAPI(
    title="Stelar Interior AI Voice Agent",
    description="Real-time AI voice agent using Pipecat + Gemini Live + Twilio",
    version="1.0.0",
)


# ── Health Check ─────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    """Health check endpoint — also shows inbound webhook URL for easy copy-paste."""
    return {
        "status": "ok",
        "service": "Stelar Interior AI Voice Agent",
        "message": "Server is running. Use POST /start-call or POST /bulk-call to initiate calls.",
        "endpoints": {
            "outbound_call": "POST /start-call",
            "bulk_outbound": "POST /bulk-call",
            "inbound_webhook": f"{settings.ngrok_url}/twiml",
        },
        "instructions": {
            "outbound": "POST /start-call with optional {\"to_number\": \"+91XXXXXXXXXX\"}",
            "inbound": f"Set your Twilio number's Voice webhook to: {settings.ngrok_url}/twiml",
            "bulk": "POST /bulk-call with {\"numbers\": [\"+91...\", \"+91...\"], \"delay_seconds\": 5}",
        },
    }


# ── Outbound: Single Call ────────────────────────────────────────────

@app.post("/start-call")
async def start_call(request: Request):
    """
    Initiate a single outbound phone call via Twilio.

    Optional JSON body:
    {
        "to_number": "+91XXXXXXXXXX"  // defaults to YOUR_PHONE_NUMBER from .env
    }
    """
    try:
        to_number = None
        try:
            body = await request.json()
            to_number = body.get("to_number")
        except Exception:
            pass

        call_sid = make_outbound_call(to_number)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Call initiated successfully",
                "call_sid": call_sid,
                "to_number": to_number or settings.your_phone_number,
            },
        )

    except Exception as e:
        logger.error(f"Failed to start call: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to initiate call: {str(e)}",
            },
        )


# ── Outbound: Bulk Call (multiple numbers) ───────────────────────────

@app.post("/bulk-call")
async def bulk_call(request: Request):
    """
    Initiate outbound calls to multiple phone numbers sequentially.

    JSON body:
    {
        "numbers": ["+919876543210", "+919876543211", "+919876543212"],
        "delay_seconds": 5  // optional delay between calls (default: 5)
    }

    Usage with Excel:
    1. Export your Excel column of phone numbers to a JSON array
    2. POST it here — the agent will call each number with a delay in between
    3. Each call gets its own call_logs/*.txt summary

    Note: Twilio trial accounts have rate limits. Use delay_seconds >= 5.
    """
    try:
        body = await request.json()
        numbers = body.get("numbers", [])
        delay_seconds = body.get("delay_seconds", 5)

        if not numbers or not isinstance(numbers, list):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Provide a 'numbers' array: {\"numbers\": [\"+91...\", \"+91...\"]}",
                },
            )

        # Validate all numbers start with +
        invalid = [n for n in numbers if not isinstance(n, str) or not n.startswith("+")]
        if invalid:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Invalid numbers (must start with +): {invalid}",
                },
            )

        results = []
        logger.info(f"📞 Bulk call starting | {len(numbers)} numbers | Delay: {delay_seconds}s")

        for i, number in enumerate(numbers):
            try:
                call_sid = make_outbound_call(number)
                results.append({
                    "number": number,
                    "status": "initiated",
                    "call_sid": call_sid,
                })
                logger.info(f"📞 Bulk call [{i+1}/{len(numbers)}] | {number} | SID: {call_sid}")
            except Exception as e:
                results.append({
                    "number": number,
                    "status": "failed",
                    "error": str(e),
                })
                logger.error(f"📞 Bulk call [{i+1}/{len(numbers)}] | {number} | FAILED: {e}")

            # Wait between calls to avoid Twilio rate limits
            if i < len(numbers) - 1:
                await asyncio.sleep(delay_seconds)

        succeeded = sum(1 for r in results if r["status"] == "initiated")
        failed = sum(1 for r in results if r["status"] == "failed")

        return JSONResponse(
            status_code=200,
            content={
                "status": "completed",
                "total": len(numbers),
                "succeeded": succeeded,
                "failed": failed,
                "results": results,
            },
        )

    except Exception as e:
        logger.error(f"Bulk call error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ── Inbound + Outbound: TwiML Webhook ───────────────────────────────

@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml_endpoint(request: Request):
    """
    Return TwiML XML that instructs Twilio to connect the call
    to our WebSocket endpoint via Media Streams.

    This is the WEBHOOK URL you set in Twilio Console for inbound calls.
    It also serves outbound calls — Twilio hits this when the callee answers.

    Twilio Console setup for inbound:
      Phone Numbers → Your Number → Voice Configuration
      → "A Call Comes In" → Webhook → https://<ngrok-url>/twiml → HTTP POST
    """
    twiml = generate_twiml_response()

    # Log direction based on Twilio form data
    form_data = {}
    try:
        form_data = dict(await request.form())
    except Exception:
        pass

    direction = form_data.get("Direction", "unknown")
    from_number = form_data.get("From", "unknown")
    to_number = form_data.get("To", "unknown")

    if direction == "inbound" or "inbound" in direction.lower():
        logger.info(f"📲 INBOUND CALL | From: {from_number} → To: {to_number} | Returning TwiML")
    else:
        logger.info(f"📞 TwiML requested | Direction: {direction} | From: {from_number} → To: {to_number}")

    return Response(content=twiml, media_type="application/xml")


# ── Call Status Callback ─────────────────────────────────────────────

@app.post("/call-status")
async def call_status(request: Request):
    """Receive call status updates from Twilio (initiated, ringing, answered, completed)."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    status = form_data.get("CallStatus", "unknown")
    call_duration = form_data.get("CallDuration", "0")

    if status == "completed":
        log_call_ended(call_sid, float(call_duration))
    elif status in ("failed", "busy", "no-answer", "canceled"):
        log_call_error(call_sid, f"Call status: {status}")
    else:
        logger.info(f"📞 Call status update | SID: {call_sid} | Status: {status}")

    return PlainTextResponse("OK")


# ── WebSocket: Twilio Media Stream ───────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handle the Twilio Media Stream WebSocket connection.
    Both inbound and outbound calls end up here after TwiML processing.
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connection accepted from Twilio")

    stream_sid = None
    call_sid = None

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            event = data.get("event")
            if event == "start":
                stream_sid = data.get("streamSid")
                call_sid = data.get("start", {}).get("callSid")
                logger.info(f"Received Twilio start event | streamSid: {stream_sid} | callSid: {call_sid}")
                break
            elif event == "connected":
                logger.info("Received Twilio connected event")
            else:
                logger.debug(f"Received Twilio event: {event}")

        if not stream_sid:
            logger.error("Could not obtain streamSid from Twilio handshake")
            await websocket.close()
            return

        await run_bot(
            websocket=websocket,
            stream_sid=stream_sid,
            call_sid=call_sid,
        )

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected by Twilio")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        log_call_error(call_sid or "unknown", str(e))
    finally:
        logger.info("🔌 WebSocket connection closed")


# ── Server Startup ───────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Stelar Interior AI Voice Agent Starting")
    logger.info(f"   Host: {settings.host}")
    logger.info(f"   Port: {settings.port}")
    logger.info(f"   ngrok URL: {settings.ngrok_url}")
    logger.info(f"   WebSocket: {settings.websocket_url}")
    logger.info(f"   TwiML: {settings.twiml_url}")
    logger.info("")
    logger.info("📞 OUTBOUND CALL:")
    logger.info("   curl -X POST http://localhost:8000/start-call")
    logger.info("")
    logger.info("📞 BULK OUTBOUND CALL:")
    logger.info('   curl -X POST http://localhost:8000/bulk-call -H "Content-Type: application/json" -d \'{"numbers": ["+91..."], "delay_seconds": 5}\'')
    logger.info("")
    logger.info("📲 INBOUND CALL SETUP:")
    logger.info(f"   Twilio Console → Phone Numbers → Your Number")
    logger.info(f"   Voice webhook URL → {settings.twiml_url}")
    logger.info(f"   Then call your Twilio number and the AI will answer!")
    logger.info("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )

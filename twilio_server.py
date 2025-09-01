from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.rest import Client
import audioop
from asyncio import create_task, sleep
import base64
import json
import os
import uvicorn
from Assistant import VoiceAssistant
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from typing import List, Optional,AsyncGenerator
from numpy.typing import NDArray
import numpy as np
import torch
from silero_vad import load_silero_vad
from faster_whisper import WhisperModel
from TTS.api import TTS #0.22.0
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from dotenv import load_dotenv
load_dotenv()

class GPTVoiceAssistant(VoiceAssistant):
    def __init__(self):
        super().__init__(
            input_sample_rate_khz=8,
            tts_sample_rate_khz=24,
            output_sample_rate_khz=8,
            vad_chunk_duration_ms=32,
            asr_chunk_duration_ms=30000,
            output_chunk_duration_ms=20
        )
        self.message_history: List[BaseMessage]=[
            SystemMessage("""
                Your name is Megan.
                You are a smart, brilliant, healpful voice assistant. 
                The upcoming user queries are detected from user speech, 
                you answer to these user queries in text, in a conversation style.
                Your answer will be converted to speech.
                So make it sound natural. Keep user engaged.
            """),
        ]
        self.stream_sid:Optional[str]=None
        self.websocket:Optional[WebSocket]=None
        self.llm = ChatOpenAI(model="gpt-4o", streaming=True)
        self.vad=load_silero_vad()
        self.asr=WhisperModel(
            "large-v3", 
            device="cuda", 
            compute_type= "float16"
        )
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
        self.tts= TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        self.tts_sample_path="sample_voice.wav"

    async def  On_Interruption(self):
        print("interrupted")

    async def Thinking_Generator(self, query:str)-> AsyncGenerator[str, None]:
        self.message_history.append(HumanMessage(content=query))
        async for chunk in self.llm.astream(self.message_history):
            if isinstance(chunk.content,str):
                yield chunk.content

    async def Speech_Generator(self,txt:str)->AsyncGenerator[NDArray[np.float32],None]:
        yield np.array(self.tts.tts(
                    text=txt,
                    speaker_wav=self.tts_sample_path,
                    language="en",
                ),dtype=np.float32)
    

    async def On_Brain_Response(self, res: str):
        if not len(self.message_history) or not isinstance(self.message_history[-1], AIMessage) or not isinstance(self.message_history[-1].content, str):
            self.message_history.append(AIMessage(content=res))
        else:
            self.message_history[-1]=AIMessage(content=self.message_history[-1].content+res)
    

    async def Detect_Speech(self, chunk: NDArray[np.float32])->bool:
        audio_tensor = torch.from_numpy(chunk)     
        speech_prob = self.vad(audio_tensor, self.input_sample_rate_khz*1000).item()
        return speech_prob>= 0.8

    async def Transcribe_Speech(self, chunk: NDArray[np.float32])->str:
        segments, _ = self.asr.transcribe(
            chunk,
            beam_size=5,
            word_timestamps=False,
            task="transcribe",
            language="en"
        )
        full_text = "".join(segment.text for segment in segments).strip()
        return full_text


    async def Handle_Output_Audio_i16_Bytes(self, chunk: bytes) -> None:
        if chunk and self.websocket:
            mulaw_chunk = audioop.lin2ulaw(chunk, 2)

            # Encode to base64
            b64_audio = base64.b64encode(mulaw_chunk).decode("utf-8")

            # 1. Send the media payload
            media_message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": b64_audio}
            }
            await self.websocket.send_text(json.dumps(media_message))
            await sleep(0.02)


from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup tasks
    print("Application is starting")
    # Perform initialization, e.g., connect to database
    app.state.assistant=GPTVoiceAssistant()

    assistant_task=create_task(app.state.assistant.start())

    print("Application started")

    yield
    # Shutdown tasks
    print("Application is shutting down")
    # Perform cleanup
    if assistant_task:
        assistant_task.cancel()
    print("application closed")
# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(lifespan=lifespan)

# Twilio credentials from environment variables for security
# It's better practice to load secrets from the environment
TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_AUTH = os.environ.get("TWILIO_AUTH")
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# Public URL (from cloudflared tunnel)
# Make sure this matches your tunnel's public address exactly
PUBLIC_URL = os.environ.get("TWILIO_SERVER_PUB_URL_NAME") #without http or https; if the websocket is hosted on the another server then put that server's public address here


# -------------------------
# Incoming calls
# -------------------------
@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio calls this when someone dials your Twilio number.
    This endpoint now returns TwiML that starts the stream and keeps the call alive.
    """
    resp = VoiceResponse()
    
    start = Start()
    start.stream(url=f"wss://{PUBLIC_URL}/media-stream")
    resp.append(start)

    resp.say("Media stream has started.")
    resp.pause(length=60) # Keep the call active for 60 seconds

    return Response(str(resp), media_type="application/xml")


# -------------------------
# Outgoing calls
# -------------------------
@app.post("/call")
async def make_call(request: Request):
    """
    Trigger an outbound call from your Twilio number.
    The TwiML is updated to keep the call alive.
    """
    try:
        data = await request.json()
        to_number = data["to"]
        from_number = data["from"]
    except (json.JSONDecodeError, KeyError):
        return Response("Invalid JSON. Please provide 'to' and 'from' numbers.", status_code=400)

    twiml_response = f"""
    <Response>
        <Connect>
            <Stream url="wss://{PUBLIC_URL}/media-stream" bidirectional="true"/>
        </Connect>
        <Say>Connecting your stream. Please wait.</Say>
        <Pause length="60" />
    </Response>
    """

    call = twilio_client.calls.create(
        to=to_number,
        from_=from_number,
        twiml=twiml_response
    )
    return {"sid": call.sid}


# -------------------------
# Media stream (WebSocket)
# -------------------------
@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    print("Hit ws endpoint")
    await ws.accept()

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)


            if data["event"] == "start":
                stream_sid=data["start"]["streamSid"]
                app.state.assistant.stream_sid= data["start"]["streamSid"]
                app.state.assistant.websocket=ws
                print("Media stream started with SID:", stream_sid)
                greet="Hello dear, I am Megan, I am your super intelligent voice assistant, How can I help you today ?"
                await app.state.assistant.queue_speech(greet)
            elif data["event"]=="media":
                audio_bytes = base64.b64decode(data["media"]["payload"])
                pcm16 = audioop.ulaw2lin(audio_bytes, 2)
                await app.state.assistant.Put_Input_Audio_i16_Bytes(pcm16)

            elif data["event"] == "stop":
                print("Call ended.")
                break

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await ws.close()

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(
        app, 
        host="localhost", 
        port=8000, 
        proxy_headers=True, 
        ws_ping_interval=20, 
        ws_ping_timeout=20
    )

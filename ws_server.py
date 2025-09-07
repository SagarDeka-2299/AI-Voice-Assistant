from asyncio import create_task, sleep
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from Assistant import VoiceAssistant, ASR_Worker, TTS_Worker
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from typing import List, Optional,AsyncGenerator, Callable
from numpy.typing import NDArray
import numpy as np
import torch
from silero_vad import load_silero_vad

from dotenv import load_dotenv
load_dotenv()


class Whisper_ASR_Worker(ASR_Worker):
    def init_asr(self) -> Callable[[NDArray[np.float32]], str]:
        from faster_whisper import WhisperModel
        model = WhisperModel(
            "large-v3" if torch.cuda.is_available() else "small", 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            compute_type= "float16" if torch.cuda.is_available() else "int8"
        )
        print("✅ ASR model loaded 🗣️ -> 🔤")
        def inference(audio_np: NDArray[np.float32])->str:
            segments, _ = model.transcribe(
                audio_np,
                beam_size=5,
                word_timestamps=False,
                task="transcribe",
                language="en"
            )
            full_text = "".join(segment.text for segment in segments).strip()
            return full_text
        return inference
class XTTS_Worker(TTS_Worker):
    def init_tts(self) -> Callable[[str], NDArray[np.float32]]:
        from TTS.api import TTS #0.22.0
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        import torch
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
        tts= TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        print("✅ TTS model loaded 🔤 -> 🗣️")
        def Speech_Generator(txt:str)->NDArray[np.float32]:
            return np.array(tts.tts(
                        text=txt,
                        speaker_wav="sample_voice.wav",
                        language="en",
                    ),dtype=np.float32)
        return Speech_Generator
class Kokoro_TTS_Worker(TTS_Worker):
    def init_tts(self) -> Callable[[str], NDArray[np.float32]]:
        from kokoro import KPipeline
        pipe = KPipeline(lang_code="a")
        print("✅ TTS model loaded 🔤 -> 🗣️")
        def synthesise(txt:str)->NDArray[np.float32]:
            gen = pipe(txt, voice="af_bella")
            audio_chunks = []
            for segment in gen:
                # in the latest repo, segment.audio is already a torch.Tensor
                wav = segment.audio
                if wav!=None:
                    audio_chunks.append(wav.numpy())
            # concatenate all chunks into one waveform
            audio_np = np.concatenate(audio_chunks)
            return audio_np
        return synthesise
        


class GPTVoiceAssistant(VoiceAssistant):
    def __init__(self):
        super().__init__(
            input_sample_rate_khz=16,
            tts_sample_rate_khz=24,
            output_sample_rate_khz=16,
            vad_chunk_duration_ms=32,
            asr_chunk_duration_ms=30000,
            output_chunk_duration_ms=20,
            asr_worker_initializer=Whisper_ASR_Worker,
            tts_worker_initializer=XTTS_Worker if torch.cuda.is_available() else Kokoro_TTS_Worker
        )
        self.message_history: List[BaseMessage]=[
            SystemMessage("""
                Your name is Megan.
                You are a smart, brilliant, healpful voice assistant's brain. 
                The upcoming user queries are detected from user speech, 
                you answer to these user queries in text, in a conversation style.
                Your answer will be converted to speech.
                So make it sound natural. Keep user engaged.
                In case the user query is not complete, just short statement, reply with hmm, mhmm, uhmmm, u r right...etc
            """),
        ]
        self.websocket:Optional[WebSocket]=None
        self.llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)


    
    def on_user_query(self, query:str):
        if not query or not len(query):
            return
        self.message_history.append(HumanMessage(content=query))
        if not len(self.message_history) or not isinstance(self.message_history[-1], HumanMessage) or not isinstance(self.message_history[-1].content, str):
            self.message_history.append(HumanMessage(content=query))
        else:
            self.message_history[-1]=HumanMessage(content=self.message_history[-1].content+query)
        
    def init_vad(self)->Callable[[NDArray[np.float32]],bool]:
        """This method should be overridden to load the VAD model and return the inference method that takes audio chunk float32 and returns boolean"""
        vad=load_silero_vad()
        print("✅ VAD model loaded 🔊 -🔎-> 🗣️")
        def Detect_Speech(chunk: NDArray[np.float32])->bool:
            audio_tensor = torch.from_numpy(chunk)     
            speech_prob = vad(audio_tensor, self.input_sample_rate_khz*1000).item()
            return speech_prob>= 0.8
        return Detect_Speech



    async def On_Brain_Response(self, thought:str):
        if not len(thought):
            return
        if not len(self.message_history) or not isinstance(self.message_history[-1], AIMessage) or not isinstance(self.message_history[-1].content, str):
            self.message_history.append(AIMessage(content=thought))
        else:
            self.message_history[-1]=AIMessage(content=self.message_history[-1].content+thought)

    async def On_Interruption(self):
        print("interrupted")
        

    async def Thinking_Generator(self)-> AsyncGenerator[str, None]:
        async for chunk in self.llm.astream(self.message_history):
            if isinstance(chunk.content,str):
                print(f"🧠: {chunk.content}")
                yield chunk.content

    async def Handle_Output_Audio_i16_Bytes(self, chunk: bytes) -> None:
        if chunk and self.websocket:
            await self.websocket.send_bytes(chunk)
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

app = FastAPI(lifespan=lifespan)

# ----------------- FastAPI app -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ----------------- WebSocket endpoint -----------------
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        app.state.assistant.websocket=websocket
        await app.state.assistant.queue_speech("Hello Dear I am Megan, your super intelligent voice assistant, How can I help you?")
        app.state.assistant.answer_ready.set()
        # Receive raw PCM16 mono, 16kHz, little-endian, any chunk size
        while True:
            data = await websocket.receive_bytes()
            # Push incoming audio straight into assistant
            await app.state.assistant.Put_Input_Audio_i16_Bytes(data)

    except WebSocketDisconnect:
        #saving the message history as a text file
        with open(f"chat_history.txt", "w") as f:
            for message in app.state.assistant.message_history:
                f.write(f"{message.type.upper()}: {message.content}\n")
        print("Web socket disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

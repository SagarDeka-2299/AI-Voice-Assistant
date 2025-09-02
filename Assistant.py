import numpy as np
from numpy.typing import NDArray
from typing import AsyncGenerator, Optional, Callable, Awaitable, List
from asyncio import Queue, create_task, Event, CancelledError, get_running_loop, TimerHandle, run, sleep, set_event_loop_policy
from abc import ABC, abstractmethod
from scipy.signal import resample_poly
import uvloop
set_event_loop_policy(uvloop.EventLoopPolicy())

def i16_bytes_to_f32_ndarray(chunk: bytes) -> NDArray[np.float32]:
    """convert from 16 bit int array into float32 numpy ndarray"""
    return np.frombuffer(chunk, dtype=np.int16).astype(np.float32, copy=False) / 32768.0

def f32_ndarray_to_i16_bytes(arr: NDArray[np.float32]) -> bytes:
    """convert from float32 numpy ndarray into 16 bit int array"""
    return np.clip(arr * 32767, -32768, 32767).astype(np.int16, copy=False).tobytes()

class Accumulator(ABC):
    """It accumulates any value untill some condition is met to call overflow and pass accumulated data"""
    def __init__(self):
        self.data: List=[]
        self.length=0
    async def accumulate(self, chunk):
        overflow_index_at_chunk = self.get_overflow_index(chunk)
        if overflow_index_at_chunk != -1:
            first_split, remaining = chunk[:overflow_index_at_chunk], chunk[overflow_index_at_chunk:]
            self.data.append(first_split)
            self.length += len(first_split)
            await self.on_overflow()
            self.clear()
            await self.accumulate(remaining)
        else:
            self.data.append(chunk)
            self.length += len(chunk)
    def clear(self):
        self.data=[]
        self.length = 0
    @abstractmethod
    def get_overflow_index(self, incoming_chunk)->int:
        """Override this method and return the index on the upcoming chunk, where overflow happened"""
        pass
    @abstractmethod
    async def on_overflow(self):
        """Override this method and do anything with the accumulated data"""
        pass

    async def manual_overflow(self):
        """manually overflow to get all remaining data accumulated"""
        await self.on_overflow()
        self.clear()

    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manual_overflow()


class AudioF32Accumulator(Accumulator):
    """Accumulates float32 numpy array as audio chunks"""
    def __init__(self,callback: Callable[[NDArray[np.float32]],Awaitable[None]],max_size:int=-1):
        self.on_overflow_callback=callback
        self.max_size=max_size
        super().__init__()
    def get_overflow_index(self, incoming_chunk: NDArray[np.float32]):
        if self.max_size==-1 or self.length+len(incoming_chunk)<=self.max_size:
            return -1
        return self.max_size-self.length
    async def on_overflow(self):
        total_chunk=np.concatenate(self.data)
        if len(total_chunk)<self.max_size:
            pad=np.zeros(self.max_size-len(total_chunk), dtype=np.float32)
            total_chunk=np.concatenate([total_chunk,pad])
        await self.on_overflow_callback(total_chunk)

class AudioBytesAccumulatorFixedSize(Accumulator):
    """Accumulates 16 bit int bytes of audio upto a fixed size"""
    def __init__(self,callback: Callable[[bytes],Awaitable[None]], size:int=1024):
        self.on_overflow_callback=callback
        self.size=size
        super().__init__()
    def get_overflow_index(self, incoming_chunk: bytes):
        if self.length+len(incoming_chunk)>self.size:
            return self.size-self.length
        return -1
    async def on_overflow(self):
        total_chunk=b''.join(self.data)
        padding_len=self.size-len(total_chunk)
        padding = b' ' * padding_len
        total_chunk+=padding
        await self.on_overflow_callback(total_chunk)
        

class StrAccumulator(Accumulator):
    """Accumulates string (beyond a minimum length), untill end of sentense is found"""
    def __init__(self, callback:Callable[[str],Awaitable[None]], min_length: int = 10):
        super().__init__()
        self.min_length = min_length
        # self.max_length = max_length
        self.on_overflow_callback=callback
    def get_overflow_index(self, incoming_chunk: str) -> int:
        if self.length + len(incoming_chunk) < self.min_length:
            return -1
        # if self.length + len(incoming_chunk) > self.max_length:
        #     return self.max_length - self.length
        for i, c in enumerate(incoming_chunk):
            if c in ['.','!','?',';',","]:
                return i + 1
        return -1
    async def on_overflow(self):
        joined_str=''.join(self.data)
        await self.on_overflow_callback(joined_str)



class VoiceAssistant(ABC):
    def __init__(
            self, 
            input_sample_rate_khz,
            tts_sample_rate_khz,
            output_sample_rate_khz,
            vad_chunk_duration_ms,
            asr_chunk_duration_ms,
            output_chunk_duration_ms
            ):
        
        self.input_sample_rate_khz=input_sample_rate_khz
        self.tts_sample_rate_khz=tts_sample_rate_khz
        self.output_sample_rate_khz=output_sample_rate_khz
        self.vad_chunk_duration_ms=vad_chunk_duration_ms
        self.asr_chunk_duration_ms=asr_chunk_duration_ms
        self.output_chunk_duration_ms=output_chunk_duration_ms

        self.user_talking=Event()
        self.query_ready=Event()
        self.answer_ready=Event()

        self.current_user_query=Queue(maxsize=1)#one query will be stored, if new segments come in, just merge with it

        self.listener_queue:Queue[bytes]=Queue()
        self.vad_to_asr_speech_chunk_queue:Queue[NDArray[np.float32]]=Queue() #User audio speech segemnt queue, 🔊--> VAD --> Queue[🗣️] --> ASR --> 🔤
        self.llm_to_tts_text_chunk_queue:Queue[str]=Queue() #Thinking output text segment queue, 🔤 --> LLM --> Queue[🔤] -->TTS --> 🗣️
        

        self.input_to_vad_chunk_accumulator=AudioBytesAccumulatorFixedSize(self.listener_queue.put,size=input_sample_rate_khz*vad_chunk_duration_ms*2)

        async def on_asr_speech_chunk_overflow(current_audio_chunk: NDArray[np.float32]):
            await self.vad_to_asr_speech_chunk_queue.put(current_audio_chunk)
            self.user_talking.clear()

        self.vad_to_asr_speech_accumulator=AudioF32Accumulator(callback=on_asr_speech_chunk_overflow,max_size=input_sample_rate_khz*asr_chunk_duration_ms)

        async def on_audio_output_chunk_overflow(chunk: bytes):
            print("putting chunk to output audio queue 📤")
            await self.Handle_Output_Audio_i16_Bytes(chunk=chunk)
            
        self.tts_speech_output_accumulator=AudioBytesAccumulatorFixedSize(callback=on_audio_output_chunk_overflow, size=output_sample_rate_khz*output_chunk_duration_ms*2)

    async def start(self):
        try:
            create_task(self.speech_detector_loop())
            create_task(self.transcriber_loop())
            create_task(self.thinking_loop())
            create_task(self.speech_synthesizer_loop())
        except Exception as e:
            print(f"Exception while starting: {e}")
            raise

    async def speech_detector_loop(self):
        print("speech detection loop start ")
        def on_timout():
            # print("tick tick boom 💥")
            create_task(self.vad_to_asr_speech_accumulator.manual_overflow())

        def reset_timer(old_handle: Optional[TimerHandle]):
            # print("reset timer ⏱️")
            if old_handle:
                old_handle.cancel()
            new_handle=get_running_loop().call_later(0.6,on_timout)
            return new_handle
        
        silence_timer:Optional[TimerHandle]=None
        while True:
            chunk=await self.listener_queue.get()
            chunk_f32=i16_bytes_to_f32_ndarray(chunk)
            is_speech = await self.Detect_Speech(chunk_f32)
            if is_speech:
                # print("🗣️")
                await self.On_Interruption()
                self.user_talking.set()
                silence_timer=reset_timer(old_handle=silence_timer)

            elif not self.user_talking.is_set():
                self.vad_to_asr_speech_accumulator.clear()
            await self.vad_to_asr_speech_accumulator.accumulate(chunk=chunk_f32)


    async def transcriber_loop(self):
        print("transcription loop start")
        while True:
            chunk = await self.vad_to_asr_speech_chunk_queue.get()
            self.query_ready.clear()
            text = await self.Transcribe_Speech(chunk)
            # print(f"transcription 🗣️ ➡️ 🔤 {text}")
            if self.current_user_query.full():
                last_query=self.current_user_query.get_nowait()
                text+=last_query
            await self.current_user_query.put(text)
            if self.vad_to_asr_speech_chunk_queue.empty():
                self.query_ready.set()

                
    async def queue_speech(self, txt:str):
        """use this to put text to talk, can be used from external code to talk manually"""
        if len(txt.strip()):
            await self.llm_to_tts_text_chunk_queue.put(txt)
            self.answer_ready.set()
            await self.On_Brain_Response(txt)



    async def thinking_loop(self):
        print("thinking loop started 🧠 🔄")
        async def thinking_task(txt:str):
            try:
                print(f"llm task started 🤖, query ❓ {txt}")
                ai_response=""
                async with StrAccumulator(callback=self.queue_speech) as str_accumulator:
                    async for chunk in self.Thinking_Generator(txt):
                        ai_response+=chunk
                        await str_accumulator.accumulate(chunk)
            except CancelledError:
                print("🚦thinking stopped, clearing llm_to_tts_text_chunk_queue")

        while True:
            # print("waiting for query to ready, so that, start thinking")
            await self.query_ready.wait()
            # print("query is ready, so start thinking")
            # print("waiting for current user query for the queue")
            query=await self.current_user_query.get()
            # print(f"current user query: {query}")
            current_thinking_task=create_task(thinking_task(query))
            # print("waiting for user to talk, so that, cancel thinking")
            await self.user_talking.wait()
            # print("user talked, so cancelling thinking")
            current_thinking_task.cancel()
            self.answer_ready.clear()
            self.llm_to_tts_text_chunk_queue=Queue()


    async def speech_synthesizer_loop(self):
        print("speech synthesis loop started")
        async def speech_generation_task():
            print("speech generation back task started")
            try:
                while True:
                    text = await self.llm_to_tts_text_chunk_queue.get()
                    # print(f"🔄generating 🗣️:  {text}")
                    async for speech in self.Speech_Generator(text):
                        #resample
                        if self.tts_sample_rate_khz != self.output_sample_rate_khz:
                            speech = resample_poly(speech, up=self.output_sample_rate_khz, down=self.tts_sample_rate_khz).astype(np.float32)
                        speech_i16bytes=f32_ndarray_to_i16_bytes(speech)
                        await self.tts_speech_output_accumulator.accumulate(speech_i16bytes)
                    # print(f"✅generated 🗣️:  {text}")
                    if self.llm_to_tts_text_chunk_queue.empty():
                        await self.tts_speech_output_accumulator.manual_overflow()
                        self.answer_ready.clear()
            except CancelledError:
                print("🚦speech generation stopped")
        while True:
            # print("waiting for answer ready, so that, start speeking")
            await self.answer_ready.wait()
            # print("answer ready, so start speeking")
            speech_synthesiser_task=create_task(speech_generation_task())
            # print("waiting for user to talk, so that, cancel speech generation")
            await self.user_talking.wait()
            # print("user talking, cancelling speech generation")
            speech_synthesiser_task.cancel()


    async def Put_Input_Audio_i16_Bytes(self, chunk: bytes)->None:
        """call this to put audio chunks of unknown size from some audio source"""
        await self.input_to_vad_chunk_accumulator.accumulate(chunk)


    @abstractmethod
    async def On_Brain_Response(self, thought:str):
        """This method should be overridden to handle AI response"""
        pass

    @abstractmethod
    async def On_Interruption(self):
        """This method should be overridden to handle Interruption by user when the AI is talking"""
        pass
        

    @abstractmethod                        
    async def Thinking_Generator(self,query:str)-> AsyncGenerator[str, None]:
        """This method should be overridden to provide LLM thinking output chunks."""
        yield "This is a placeholder for LLM thinking output. Override this method to provide actual LLM output."

    @abstractmethod
    async def Handle_Output_Audio_i16_Bytes(self, chunk: bytes) -> None:
        """This method should be overridden to handle output audio stream chunks."""
        pass

    @abstractmethod
    async def Detect_Speech(self, chunk: NDArray[np.float32])->bool:
        """This method should be overridden to call VAD model of audio chunks."""
        pass

    @abstractmethod
    async def Transcribe_Speech(self, chunk: NDArray[np.float32])->str:
        """This method should be overridden to call ASR model of audio chunks."""
        pass

    @abstractmethod
    async def Speech_Generator(self,sentense:str)->AsyncGenerator[NDArray[np.float32],None]:
        """This method should be overridden to call TTS model."""
        yield np.zeros(100, dtype=np.float32)




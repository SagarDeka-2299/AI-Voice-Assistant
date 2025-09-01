# AI Voice Assistant (Realtime Conversational AI)

This project implements a **real-time voice assistant** capable of live conversation over the web and phone calls.
It uses **asynchronous task orchestration** with producer–consumer patterns, integrating **VAD** (Silero Voice Activity Detection Model), **ASR** (Whisper Automatic Speech Recognition), **TTS** (Coqui XTTS-V2 Text-to-Speech), and **LLMs** (GPT-4o Large Language Models).

Two servers are provided:

* **WebSocket server**: serves a webpage frontend and `/ws` route for live browser audio streaming.
* **Twilio server**: enables **VoIP phone calls** through a Twilio phone number.

---

## Features

* Live **speech detection, transcription, LLM response, and speech synthesis** pipeline.
* Event-driven, fully asynchronous architecture.
* Supports:

  * **Web-based conversations** over WebSocket.
  * **Phone calls** via Twilio Media Streams.
* **OpenAI GPT models** for brain (LLM).
* **Silero VAD** for speech detection.
* **Faster-Whisper** for transcription.
* **Coqui TTS (XTTS v2)** for multilingual voice synthesis.
* Runs efficiently with **CUDA GPU acceleration**.

---

## Setup

### 1. Clone repository

```bash
git clone https://github.com/SagarDeka-2299/AI-Voice-Assistant.git
cd https://github.com/SagarDeka-2299/AI-Voice-Assistant.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Edit `.env.local` as instructed there

### 4. Install cuDNN

Download cuDNN for your CUDA version:
[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

Find installation path:

```bash
find /usr -name "*cudnn*" 2>/dev/null | grep "\.so\.9"
```

Set environment variable:

```bash
export LD_LIBRARY_PATH=<path-to-dir>:$LD_LIBRARY_PATH
```

Example:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

---

## Run

### WebSocket server

```bash
python ws_server.py
```

* Serves frontend at [http://localhost:8000](http://localhost:8000)
* WebSocket endpoint: `ws://localhost:8000/ws`

### Twilio server

```bash
python twilio_server.py
```

* Incoming calls handled via `/incoming-call`
* Outgoing calls via `/call`
* Media stream WebSocket at `/media-stream`

---

## Project Structure

* **Assistant.py** – Core orchestration logic for VAD, ASR, LLM, TTS.
* **ws\_server.py** – WebSocket + frontend server.
* **twilio\_server.py** – Twilio VoIP server.

---

## Contact

* **Email**: [sagardekaofficial@gmail.com](mailto:sagardekaofficial@gmail.com)
* **LinkedIn**: [https://www.linkedin.com/in/sagardeka/](https://www.linkedin.com/in/sagardeka/)

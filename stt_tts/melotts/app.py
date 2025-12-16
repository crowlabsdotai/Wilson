from melo.api import TTS
import warnings
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import pipeline
from datasets import load_dataset
import io
import numpy as np
import soundfile as sf
import whisper
import json
import os
import torch
from contextlib import contextmanager
import sys
import onnxruntime
from onnxruntime_extensions import get_library_path
from pyannote.audio import Pipeline
import logging
from speechbrain.inference.diarization import Speech_Emotion_Diarization
from datetime import datetime
from pyannote.core import Annotation, Segment
import pickle
from transformers import AutoTokenizer
import scipy.signal as sps

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

speed = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# English 
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()

            audio = model.tts_to_file(text, speaker_ids['EN-US'], sdp_ratio=0.4, noise_scale=0.6, speed=speed)
            audio = np.array(audio, dtype=np.float32)
            samplerate = model.hps.data.sampling_rate

            # Resample data
            new_rate = 16000
            number_of_samples = round(len(audio) * float(new_rate) / samplerate)
            audio = sps.resample(audio, number_of_samples)

            await websocket.send_text(json.dumps({"samplerate": new_rate}))

            await websocket.send_bytes(audio.tobytes())
    except WebSocketDisconnect:
        print("TTS Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
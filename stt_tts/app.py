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

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

diarization = True

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def speakers_to_numpy(annotation):
    rttm_data = []

    if isinstance(annotation, Annotation):
        for segment, _, label in annotation.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            duration = end_time - start_time
            speaker_id = label

            rttm_data.append([start_time, duration, speaker_id])

    else:
        raise TypeError("The provided annotation object is not of type 'pyannote.core.Annotation'.")

    rttm_array = np.array(rttm_data, dtype=object) 
    
    return rttm_array

def emotions_to_numpy(data_dict):
    rttm_data = []

    for timestamp, segments in data_dict.items():
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            emotion_label = segment['emotion']

            rttm_data.append([start_time, duration, emotion_label])

    rttm_array = np.array(rttm_data, dtype=object) 
    
    return rttm_array

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

synthesiser = pipeline("text-to-speech", model="microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

whisper_model = whisper.load_model("tiny", device=device)

if diarization:

    pipeline_pyannote = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="add your token")

    classifier = Speech_Emotion_Diarization.from_hparams(source="speechbrain/emotion-diarization-wavlm-large")

@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            audio = audio / 32767.0

            transcription = whisper_model.transcribe(audio)
            
            transcription_json = json.dumps(transcription)
            
            await websocket.send_text(transcription_json)

            if diarization:

                speaker_diarization = pipeline_pyannote({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000})
                logger.info(speaker_diarization)
                await websocket.send_bytes(pickle.dumps(speakers_to_numpy(speaker_diarization)))

                emotion_diarization = classifier.diarize_batch(torch.from_numpy(audio).unsqueeze(0), torch.tensor([1.0]), [datetime.now()])
                logger.info(emotion_diarization)
                await websocket.send_bytes(pickle.dumps(emotions_to_numpy(emotion_diarization)))

    except WebSocketDisconnect:
        print("STT Client disconnected")

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()

            speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
            audio = np.array(speech["audio"], dtype=np.float32)
            samplerate = speech["sampling_rate"]

            await websocket.send_text(json.dumps({"samplerate": samplerate}))

            await websocket.send_bytes(audio.tobytes())
    except WebSocketDisconnect:
        print("TTS Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

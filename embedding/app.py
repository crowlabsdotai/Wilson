import numpy as np
import torch
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import json
from pyannote.audio import Model
from pyannote.audio import Inference

app = FastAPI()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

model_pyannote = Model.from_pretrained("pyannote/embedding", use_auth_token="add your token")
inference = Inference(model_pyannote, window="whole")

def get_wave2vec_embedding(audio: np.ndarray):
    rows = 64
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.numpy()

    reshaped_embeddings = embeddings.reshape(embeddings.shape[1], embeddings.shape[2] // rows, rows)
    max_embeddings = np.max(reshaped_embeddings, axis=2)
    return max_embeddings

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_data = await websocket.receive_bytes()

            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            audio = audio / 32767.0

            embeddings = get_wave2vec_embedding(audio).astype(np.float32)

            await websocket.send_json({"embeddings": embeddings.tolist()})

    except WebSocketDisconnect:
        print("wav2vec Client disconnected")

import torch

@app.websocket("/ws/embedding")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            audio_data = await websocket.receive_bytes()

            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio = audio / 32767.0 

            waveform = torch.tensor([audio])

            embeddings = inference({"waveform": waveform, "sample_rate": 16000}).astype(np.float32)

            await websocket.send_json({"embeddings": embeddings.tolist()})

    except WebSocketDisconnect:
        print("Embedding Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

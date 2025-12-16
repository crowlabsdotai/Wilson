import pickle
import numpy as np
import websockets
import json
import asyncio

def send_raw_audio_and_receive_embeddings(audio_data, samplerate=16000):
    async def _send_and_receive():
        #uri = "ws://localhost:8080/ws/audio"
        uri = "ws://localhost:8080/ws/embedding"

        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_data_int16.tobytes()

        async with websockets.connect(uri, timeout=120) as websocket:
            # Send raw audio bytes over WebSocket
            await websocket.send(audio_bytes)

            # Receive embeddings from the server
            embeddings_data = await websocket.recv()
            embeddings_json = json.loads(embeddings_data)
            embeddings = embeddings_json['embeddings']

            return embeddings

    return asyncio.run(_send_and_receive())

def transcribe_audio_stt_tts(audio_data, diarization=True):
    async def _transcribe():
        uri = "ws://localhost:8000/ws/stt"
        
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_data_int16.tobytes()
        
        async with websockets.connect(uri, timeout=120) as websocket:
            await websocket.send(audio_bytes)
            
            transcription_json = await websocket.recv()
            
            transcription = json.loads(transcription_json)["text"].strip()

            if diarization:

                speakers = await websocket.recv()
                speakers = pickle.loads(speakers)

                emotions = await websocket.recv()
                emotions = pickle.loads(emotions)

            return transcription

    return asyncio.run(_transcribe())

def synthesize_voice_stt_tts(text):
    async def _synthesize():
        #uri = "ws://localhost:8000/ws/tts"        
        uri = "ws://localhost:8888/ws/tts"
        async with websockets.connect(uri, timeout=120) as websocket:
            await websocket.send(text)
            
            samplerate_data = await websocket.recv()
            samplerate_json = json.loads(samplerate_data)
            samplerate = samplerate_json['samplerate']
            
            audio_data = await websocket.recv()
            audio = np.frombuffer(audio_data, dtype=np.float32)
            
            return samplerate, audio

    return asyncio.run(_synthesize())
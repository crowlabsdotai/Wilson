import subprocess
import sounddevice as sd
import numpy as np
import threading
import librosa
import torch
import noisereduce as nr
from colorama import Fore, Style, init
import warnings
import paho.mqtt.client as mqtt
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from collections import deque
from time import sleep
import redis
import os
import chromadb
import ollama
import json
from ascii_magic import AsciiArt, Back
import asyncio
import websockets
import json
import numpy as np
import sys
import numpy as np
import pickle
import random
from scipy.ndimage import gaussian_filter1d
from scipy.io import wavfile
from scipy import signal
import os
import random
import wave
import pyaudio
import time as T
from sklearn.metrics.pairwise import cosine_similarity
from clients import *

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=DeprecationWarning)

SAMPLE_RATE = 16000
CHUNK = 1024
BUFFER_DURATION = 10 
BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION

WOKE = True
current_time_seconds = T.time()
is_playing_tts = {"status": True, "time": current_time_seconds}

my_art = AsciiArt.from_image('./wilson.png')
my_output = my_art.to_ascii(columns=70)
print(my_output)

def get_WOKE():
	global WOKE
	return WOKE

def get_is_playing_tts():
	global is_playing_tts
	return is_playing_tts

def set_WOKE(value):
    global WOKE
    WOKE = value

def set_is_playing_tts(value):
    global is_playing_tts
    is_playing_tts = value

def play_random_audio(folder_path):
    current_time_seconds = T.time()
    is_playing_tts = {"status": True, "time": current_time_seconds}
    set_is_playing_tts(is_playing_tts)
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    random_file = random.choice(wav_files)
    file_path = os.path.join(folder_path, random_file)
    sample_rate, audio_array = wavfile.read(file_path)
    audio_array = correct_audio_block_size(audio_array, 1024)
    play_audio(sample_rate, audio_array)
    #sd.playrec(audio_array, sample_rate, channels=1, blocksize=1024)
    #sd.wait()
    return

def play_audio(sample_rate, audio_array):
    sd.playrec(audio_array, sample_rate, channels=1, blocksize=1024)
    sd.wait()

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def sleep_randomly(min_seconds=1, max_seconds=10):
    duration = random.uniform(min_seconds, max_seconds)
    sleep(duration)

def load_and_preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return audio

def correct_audio_block_size(audio_data, block_size):
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
    
    remainder = len(audio_data) % block_size
    if remainder != 0:
        padding = block_size - remainder
        audio_data = np.pad(audio_data, (0, padding), mode='constant')
    return audio_data

sleepwords = load_and_preprocess_audio("asset/recordings/sleepwords.wav")
wakeupwords = load_and_preprocess_audio("asset/recordings/wakeupwords.wav")

sleepwords_embeddings = send_raw_audio_and_receive_embeddings(sleepwords)
wakeupwords_embeddings = send_raw_audio_and_receive_embeddings(wakeupwords)

def calculate_correlation(audio_embeddings, keyword_embeddings, opposite_keyword_embeddings, threshold):
    audio_embeddings = np.array(audio_embeddings); audio_embeddings /= np.linalg.norm(audio_embeddings)
    keyword_embeddings = np.array(keyword_embeddings); keyword_embeddings /= np.linalg.norm(keyword_embeddings)
    opposite_keyword_embeddings = np.array(opposite_keyword_embeddings); opposite_keyword_embeddings /= np.linalg.norm(opposite_keyword_embeddings)

    correlation = signal.correlate(audio_embeddings, keyword_embeddings)
    max_corr = np.max(np.abs(correlation))
    print(Fore.WHITE + f"Max correlation: {max_corr}")

    opposite_correlation = signal.correlate(audio_embeddings, opposite_keyword_embeddings)
    max_opposite_corr = np.max(np.abs(opposite_correlation))
    print(Fore.WHITE + f"Max opposite correlation: {max_opposite_corr}")

    return (max_corr > threshold and max_opposite_corr < max_corr)

def check_sleep_wilson(within_audio):
    global WOKE
    global is_playing_tts
    within_audio_embeddings = send_raw_audio_and_receive_embeddings(within_audio)
    
    if calculate_correlation(within_audio_embeddings, sleepwords_embeddings, wakeupwords_embeddings, threshold=0.25) and not is_playing_tts["status"]:
        WOKE = False
        print(Fore.RED + "Wilson going to sleep :(")
        play_random_audio('asset/audios/byes')
        current_time_seconds = T.time()
        is_playing_tts = {"status": False, "time": current_time_seconds}
    return

def check_wake_wilson(within_audio):
    global WOKE
    global is_playing_tts
    within_audio_embeddings = send_raw_audio_and_receive_embeddings(within_audio)
    
    if calculate_correlation(within_audio_embeddings, wakeupwords_embeddings, sleepwords_embeddings, threshold=0.25) and not is_playing_tts["status"]:
        WOKE = True
        print(Fore.RED + "Wilson waking up :)")
        play_random_audio('asset/audios/starters')
        current_time_seconds = T.time()
        is_playing_tts = {"status": False, "time": current_time_seconds}
    return
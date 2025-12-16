import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Record audio and save to a WAV file.')
parser.add_argument('output_filename', type=str, help='The output WAV file name (with path).')
args = parser.parse_args()

# Recording parameters
duration = 2  # seconds
sample_rate = 16000  # Hz

# Record audio
print("Recording...")
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished
print("Recording finished.")

# Save the recorded audio to a WAV file
write(args.output_filename, sample_rate, audio_data)
print(f"Audio saved as {args.output_filename}.")


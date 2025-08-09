import os
import time
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer

# Check if the model exists
if not os.path.exists('model'):
    print("Download the Vosk model from https://alphacephei.com/vosk/models and unpack it into 'model' folder")
    exit(1)

# Load the Vosk model
start_time = time.time()
model = Model('model')
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# Initialize the recognizer
recognizer = KaldiRecognizer(model, 16000)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1024
)

print("Say something!")

# Silence detection paramaters
SILENCE_THRESHOLD = 500
last_audio_time = time.time()

# Process audio input
try:
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        if len(data) == 0:
            break

        #convert audio data into Numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Calculate the root-mean-square (RMS) energy of the audio chunk
        epsilon = 1e-10  # A very small value to prevent division by zero or sqrt of zero
        energy = np.sqrt(np.mean(audio_data ** 2) + epsilon)

        # Check if there's any non_silence data
        if energy >SILENCE_THRESHOLD :
            last_audio_time = time.time() #reset time

        # Check for final result
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            print(f"\nYou said: {result}")  # Print final result on a new line
            final_result = result
            break

        # partial_result = recognizer.PartialResult()
        # print(f"\rPartial: {partial_result.strip()}", end="", flush=True)

        #check for timeout

        if time.time() - last_audio_time >= SILENCE_THRESHOLD:
            print("\nSilence detected. Stopping listening...")
            break

finally:
    # Clean up resources
    stream.stop_stream()
    stream.close()
    p.terminate()
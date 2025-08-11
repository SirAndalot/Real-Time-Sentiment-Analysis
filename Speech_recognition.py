import json
import os
import time
import pyaudio
import numpy as np
from vosk import Model, KaldiRecognizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Check if the Vosk model exists
VOSK_MODEL_PATH = "model"
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Vosk model folder '{VOSK_MODEL_PATH}' not found. Please download and extract the model.")
    exit(1)

# Load the Vosk model
vosk_model = Model('model')

# Initialize the recognizer
recognizer = KaldiRecognizer(vosk_model, 16000)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1024
)
stream.start_stream()

print("Say something!")

# Silence detection parameters
SILENCE_THRESHOLD = 300  # Adjust this value based on your microphone sensitivity
MAX_SILENCE_DURATION = 10  # Increased maximum allowed silence duration (in seconds)
last_audio_time = time.time()
sia = SentimentIntensityAnalyzer()

# Process audio input
try:
    full_transcription = ""  # To accumulate partial results
    last_partial_text = ""  # Track the last partial result to avoid repetition
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        if len(data) == 0:
            break

        # Convert audio data into a Numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Calculate the root-mean-square (RMS) energy of the audio chunk
        epsilon = 1e-10  # A very small value to prevent division by zero or sqrt of zero
        energy = np.sqrt(np.mean(audio_data ** 2) + epsilon)

        # Check if there's any non-silence data
        if energy > SILENCE_THRESHOLD:
            last_audio_time = time.time()  # Reset the timer if speech is detected

        # Get partial results
        if recognizer.PartialResult():
            partial_result = recognizer.PartialResult()
            try:
                # Parse the JSON string using json.loads
                parsed_result = json.loads(partial_result)
                partial_text = parsed_result.get("partial", "").strip()  # Safely get the "partial" key

                # Avoid adding repeated or identical partial results
                if partial_text and partial_text != last_partial_text:
                    full_transcription += " " + partial_text
                    last_partial_text = partial_text  # Update the last partial result

            except json.JSONDecodeError:
                print("Error decoding partial result.")

        # Check for final result
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            try:
                # Parse the JSON string using json.loads
                parsed_result = json.loads(result)
                transcribed_text = parsed_result.get("text", "").strip()  # Safely get the "text" key

                # Use only the finalized transcription
                full_transcription = transcribed_text

                # Perform sentiment analysis
                sentiment_score = sia.polarity_scores(full_transcription.strip())

                print(f"\nYou said: {full_transcription.strip()}")
                print(sentiment_score)
                break

            except json.JSONDecodeError:
                print("Error decoding final result.")

        # Check for timeout (silence for too long)
        if time.time() - last_audio_time >= MAX_SILENCE_DURATION:
            print("\nSilence detected. Stopping listening...")
            break

finally:
    # Clean up resources
    stream.stop_stream()
    stream.close()
    p.terminate()
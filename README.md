Real-Time Speech-to-Text with Sentiment Analysis

This Python application transcribes spoken words into text in real-time using the Vosk library and analyzes the sentiment of the text using NLTK.

Features:

  Transcribes speech to text as you speak.
  Analyzes the sentiment of the transcribed text (positive, negative, neutral).
  Automatically stops listening after a period of silence.
  
  Requirements:
  
  Python 3.7+
  Microphone
  Vosk model (any English model) ([download from here](https://alphacephei.com/vosk/models))
  After downloading the model, rename it to "model" and at it to the project
  Libraries: pyaudio, vosk, nltk, numpy

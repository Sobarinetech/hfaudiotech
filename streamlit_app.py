import streamlit as st
from transformers import pipeline
import torch  # Ensure PyTorch is available

# Load the Whisper model
@st.cache_resource
def load_model():
    # Ensure that PyTorch is installed and used for Whisper
    return pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", framework="pt")

# Initialize the Streamlit app
st.title("WAV Audio Transcription App")
st.write("Upload a WAV audio file, and this app will transcribe its content using OpenAI's Whisper model.")

# File uploader for WAV files
audio_file = st.file_uploader("Upload your WAV audio file", type=["wav"])

# Load the Whisper model
whisper_pipeline = load_model()

if audio_file:
    # Display the uploaded audio
    st.audio(audio_file, format="audio/wav")
    st.write("Processing and transcribing your audio...")

    try:
        # Transcribe the audio directly using the Whisper model
        transcription = whisper_pipeline(audio_file)["text"]
        st.write("### Transcription:")
        st.write(transcription)
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")

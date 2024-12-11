import os
import torch
import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_audio(file_path):
    """Transcribe audio using Wav2Vec2."""
    try:
        # Read the audio file
        audio, sample_rate = sf.read(file_path)

        # Ensure audio is 16 kHz mono
        if sample_rate != 16000:
            raise ValueError("Audio must be 16 kHz. Please preprocess your audio file to match this format.")

        # Tokenize and process audio
        input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
        logits = model(input_values).logits

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def main():
    """Streamlit app for audio transcription."""
    st.title("Speech-to-Text Transcription with Wav2Vec2")
    st.write("Upload a .wav audio file in 16 kHz mono format for transcription.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

    if uploaded_file is not None:
        with st.spinner("Transcribing audio..."):
            # Save uploaded file to a temporary path
            temp_file_path = os.path.join("temp_audio", uploaded_file.name)
            os.makedirs("temp_audio", exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Transcribe audio
            transcription = transcribe_audio(temp_file_path)

            if transcription:
                st.success("Transcription completed successfully!")
                st.text_area("Transcription", transcription, height=200)
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()

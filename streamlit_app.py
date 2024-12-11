import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import librosa

# Load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def main():
    st.title("Speech-to-Text Transcription App")

    # File Uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Read the audio file
        audio_bytes = uploaded_file.read()
        audio_array, sampling_rate = librosa.load(audio_bytes, sr=16000)

        # Preprocess the audio
        input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values

        # Generate the transcription
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

        st.text(transcription)

        # Download the transcription
        st.download_button(
            label="Download Transcription",
            data=transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

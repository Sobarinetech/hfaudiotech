import os
import streamlit as st
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Set up the PyAnnote pipeline with HuggingFace token
HUGGINGFACE_ACCESS_TOKEN = "your_huggingface_access_token_here"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_ACCESS_TOKEN
)

def process_audio(file_path):
    """Processes the audio file and performs speaker diarization."""
    try:
        diarization = pipeline(file_path)
        rttm_output = "audio.rttm"
        with open(rttm_output, "w") as rttm:
            diarization.write_rttm(rttm)
        return diarization
    except Exception as e:
        st.error(f"Failed to process audio: {e}")
        return None

def convert_rttm_to_text(diarization, output_path):
    """Converts diarization results to text format with speaker labels."""
    transcription = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        transcription.append(f"Speaker {speaker}: [{segment.start:.2f} - {segment.end:.2f}]\n")
    
    with open(output_path, "w") as txt_file:
        txt_file.write("\n".join(transcription))

    return transcription

def main():
    """Streamlit application for audio transcription."""
    st.title("Call Transcription Tool")
    st.write("Upload your call recording to transcribe:")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with st.spinner("Processing audio file..."):
            # Save uploaded file to a temporary path
            temp_file_path = os.path.join("temp_audio_file", uploaded_file.name)
            os.makedirs("temp_audio_file", exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Convert file to WAV format if needed
            if not temp_file_path.endswith(".wav"):
                audio = AudioSegment.from_file(temp_file_path)
                wav_path = temp_file_path.rsplit(".", 1)[0] + ".wav"
                audio.export(wav_path, format="wav")
                temp_file_path = wav_path

            # Perform diarization
            diarization = process_audio(temp_file_path)
            if diarization:
                # Save transcription to text file
                output_path = temp_file_path.rsplit(".", 1)[0] + "_transcription.txt"
                transcription = convert_rttm_to_text(diarization, output_path)
                
                st.success("Transcription completed successfully!")
                st.download_button(
                    label="Download Transcription", 
                    data="\n".join(transcription), 
                    file_name="transcription.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()

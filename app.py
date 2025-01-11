import streamlit as st
import whisper
import tempfile
import os
from moviepy.editor import VideoFileClip

st.title("🎥 Video a Texto")

def transcribe_video(video_path):
    # Convertir video a audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio.name, verbose=False)
        video.close()
        
        # Cargar modelo y transcribir
        model = whisper.load_model("tiny")
        result = model.transcribe(temp_audio.name)
        
        os.unlink(temp_audio.name)
        return result['text']

uploaded_file = st.file_uploader("Sube un video", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    # Guardar el archivo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    try:
        with st.spinner('Transcribiendo...'):
            transcription = transcribe_video(video_path)
            st.text_area("Transcripción:", transcription, height=300)
            
            # Botón de descarga
            st.download_button(
                "Descargar transcripción",
                transcription,
                "transcripcion.txt",
                "text/plain"
            )
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

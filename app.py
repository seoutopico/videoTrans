import streamlit as st
import whisper_timestamped
import tempfile
from moviepy.editor import VideoFileClip
import os

st.set_page_config(page_title="Video Transcriptor", page_icon="🎥")

st.title("📝 Video a Texto")
st.write("Sube un video y obtén su transcripción con marcas de tiempo")

def process_video(video_path):
    """Procesa el video y retorna la transcripción"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio:
        # Extraer audio
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio.name, verbose=False)
        video.close()
        
        # Transcribir
        model = whisper_timestamped.load_model("tiny", device="cpu")
        result = whisper_timestamped.transcribe(model, temp_audio.name)
        
        return result

def format_time(seconds):
    """Convierte segundos a formato MM:SS"""
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

uploaded_file = st.file_uploader("Escoge un video", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Guardar el archivo temporalmente
    temp_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Procesando video..."):
        try:
            # Procesar el video
            result = process_video(temp_path)
            
            # Mostrar transcripción
            st.subheader("Transcripción:")
            
            # Crear texto con timestamps
            transcription = []
            for segment in result['segments']:
                timestamp = format_time(segment['start'])
                transcription.append(f"[{timestamp}] {segment['text']}")
            
            # Mostrar y permitir descargar
            transcript_text = "\n".join(transcription)
            st.text_area("Texto completo:", transcript_text, height=300)
            
            # Botón de descarga
            st.download_button(
                label="Descargar transcripción",
                data=transcript_text,
                file_name="transcripcion.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Error procesando el video: {str(e)}")
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)

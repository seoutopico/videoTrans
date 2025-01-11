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
    try:
        # Verificar que el archivo existe
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"No se encontró el archivo: {video_path}")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # Cargar el video y verificar que tiene audio
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                raise ValueError("El video no contiene audio")
                
            # Extraer audio
            st.info("Extrayendo audio del video...")
            video.audio.write_audiofile(
                temp_audio.name,
                codec='pcm_s16le',  # Codec más compatible
                ffmpeg_params=["-ac", "1"],  # Convertir a mono
                verbose=False
            )
            video.close()
            
            # Transcribir
            st.info("Transcribiendo audio...")
            model = whisper_timestamped.load_model("tiny", device="cpu")
            result = whisper_timestamped.transcribe(model, temp_audio.name)
            
            # Limpiar
            os.unlink(temp_audio.name)
            return result
            
    except Exception as e:
        st.error(f"Error procesando el video: {str(e)}")
        if 'temp_audio' in locals() and os.path.exists(temp_audio.name):
            os.unlink(temp_audio.name)
        raise

def format_time(seconds):
    """Convierte segundos a formato MM:SS"""
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

# Configurar el uploader para aceptar videos
uploaded_file = st.file_uploader(
    "Escoge un video", 
    type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
    help="Formatos soportados: MP4, AVI, MOV, MKV, WEBM"
)

if uploaded_file is not None:
    # Mostrar información del archivo
    file_details = {
        "Nombre": uploaded_file.name,
        "Tamaño": f"{uploaded_file.size / (1024*1024):.2f} MB"
    }
    st.write("Detalles del archivo:")
    for key, value in file_details.items():
        st.write(f"- {key}: {value}")

    # Guardar el archivo temporalmente
    temp_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Procesando video..."):
        try:
            # Procesar el video
            result = process_video(temp_path)
            
            if result and 'segments' in result:
                # Mostrar transcripción
                st.success("¡Transcripción completada!")
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
            else:
                st.error("No se pudo generar la transcripción")
            
        except Exception as e:
            st.error(f"Error procesando el video: {str(e)}")
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)

st.markdown("""
---
### Notas:
- El video debe contener audio para poder transcribirlo
- Archivos más grandes pueden tardar más en procesarse
- La transcripción funciona mejor con audio claro y sin ruido de fondo
""")

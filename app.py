import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(
    page_title="Video Transcriber",
    page_icon="🎥",
    layout="centered"
)

def extract_audio(video_path, output_path):
    """Extrae el audio de un video y lo guarda como archivo temporal"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()
    audio.close()

def transcribe_audio(audio_path):
    """Transcribe el audio usando whisper"""
    model = whisper.load_model("base")  # Usamos el modelo base para ahorrar recursos
    result = model.transcribe(audio_path)
    return result["text"]

def main():
    st.title("📽️ Transcriptor de Videos")
    st.write("Sube un video para obtener su transcripción")
    
    # Widget para subir archivo
    video_file = st.file_uploader("Sube tu video", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        try:
            # Crear directorio temporal para procesar archivos
            with tempfile.TemporaryDirectory() as temp_dir:
                # Guardar video subido
                video_path = os.path.join(temp_dir, "temp_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(video_file.read())
                
                # Mostrar mensaje de procesamiento
                with st.spinner("Procesando video... Por favor espera."):
                    # Extraer y guardar audio
                    audio_path = os.path.join(temp_dir, "temp_audio.wav")
                    extract_audio(video_path, audio_path)
                    
                    # Transcribir audio
                    transcription = transcribe_audio(audio_path)
                
                # Mostrar resultados
                st.success("¡Transcripción completada!")
                st.write("### Transcripción:")
                st.write(transcription)
                
                # Opción para descargar transcripción
                st.download_button(
                    label="Descargar transcripción",
                    data=transcription,
                    file_name="transcripcion.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Error durante el procesamiento: {str(e)}")
            st.write("Por favor, intenta con un video más corto o en otro formato.")

if __name__ == "__main__":
    main()

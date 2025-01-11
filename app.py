import streamlit as st
import whisper
import os
import tempfile
from moviepy.editor import VideoFileClip

# Configuración de la página
st.set_page_config(page_title="Transcripción de Video", layout="centered")

st.title("🎥 Transcripción de Video a Texto")
st.write("Sube un video y obtén su transcripción en texto con timestamps.")

# Subir archivo de video
uploaded_file = st.file_uploader("Sube un archivo de video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    # Extraer audio usando moviepy
    temp_audio_path = temp_video_path.replace(".mp4", ".mp3")

    st.info("⏳ Procesando el video y extrayendo el audio...")

    try:
        video_clip = VideoFileClip(temp_video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(temp_audio_path, codec="mp3")
        audio_clip.close()
        video_clip.close()
        st.success("✅ Audio extraído con éxito.")
    except Exception as e:
        st.error(f"❌ Error al extraer el audio: {str(e)}")

    # Cargar modelo Whisper
    st.info("⏳ Cargando modelo Whisper...")
    model = whisper.load_model("small")  # Puedes usar "base", "small", "medium", "large"
    st.success("✅ Modelo cargado con éxito.")

    # Transcribir audio
    st.info("⏳ Transcribiendo audio...")
    result = model.transcribe(temp_audio_path)

    # Generar archivo de transcripción con timestamps
    transcription_text = ""
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        transcription_text += f"[{start_time:.2f} - {end_time:.2f}] {text}\n"

    # Guardar transcripción en archivo TXT
    transcription_file_path = temp_audio_path.replace(".mp3", ".txt")
    with open(transcription_file_path, "w", encoding="utf-8") as f:
        f.write(transcription_text)

    st.success("✅ Transcripción completada con éxito.")

    # Mostrar texto en pantalla
    st.text_area("📜 Transcripción:", transcription_text, height=300)

    # Botón para descargar el archivo
    with open(transcription_file_path, "rb") as f:
        st.download_button(
            label="📥 Descargar transcripción",
            data=f,
            file_name="transcripcion.txt",
            mime="text/plain",
        )

    # Limpieza de archivos temporales
    os.remove(temp_video_path)
    os.remove(temp_audio_path)
    os.remove(transcription_file_path)

st.info("📌 Usa este código en Streamlit para compartirlo con la comunidad.")

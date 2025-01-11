import streamlit as st
import whisper
import ffmpeg
import os
import tempfile

st.set_page_config(page_title="Transcripción de Video", layout="centered")

st.title("🎥 Transcripción de Video a Texto")
st.write("Sube un video y obtén su transcripción en texto con timestamps.")

uploaded_file = st.file_uploader("Sube un archivo de video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    temp_audio_path = temp_video_path.replace(".mp4", ".mp3")

    st.info("⏳ Procesando el video y extrayendo el audio...")

    try:
        (
            ffmpeg
            .input(temp_video_path)
            .output(temp_audio_path, format="mp3", acodec="mp3", ar="16k")
            .run(overwrite_output=True, quiet=True)
        )
        st.success("✅ Audio extraído con éxito.")
    except Exception as e:
        st.error(f"❌ Error al extraer el audio: {str(e)}")

    st.info("⏳ Cargando modelo Whisper...")
    model = whisper.load_model("small")
    st.success("✅ Modelo cargado con éxito.")

    st.info("⏳ Transcribiendo audio...")
    result = model.transcribe(temp_audio_path)

    transcription_text = ""
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        transcription_text += f"[{start_time:.2f} - {end_time:.2f}] {text}\n"

    transcription_file_path = temp_audio_path.replace(".mp3", ".txt")
    with open(transcription_file_path, "w", encoding="utf-8") as f:
        f.write(transcription_text)

    st.success("✅ Transcripción completada con éxito.")

    st.text_area("📜 Transcripción:", transcription_text, height=300)

    with open(transcription_file_path, "rb") as f:
        st.download_button(
            label="📥 Descargar transcripción",
            data=f,
            file_name="transcripcion.txt",
            mime="text/plain",
        )

    os.remove(temp_video_path)
    os.remove(temp_audio_path)
    os.remove(transcription_file_path)

st.info("📌 Usa este código en Streamlit para compartirlo con la comunidad.")

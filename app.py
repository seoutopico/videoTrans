import os
import tempfile
from pathlib import Path

import streamlit as st
from moviepy.editor import VideoFileClip
import whisper

st.set_page_config(page_title="Video Transcriber", page_icon="🎥", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    # carga una única vez por proceso
    return whisper.load_model(model_name)

def extract_audio(video_path: str, output_path: str):
    """Extrae el audio de un vídeo a WAV (requiere ffmpeg a nivel sistema)."""
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_path, verbose=False, logger=None)
    audio.close()
    clip.close()

def transcribe_audio(audio_path: str, model_name: str, language_hint: str | None):
    model = load_model(model_name)
    opts = {}
    if language_hint and language_hint != "auto":
        opts["language"] = language_hint
    result = model.transcribe(audio_path, **opts)
    return result.get("text", "").strip()

def main():
    st.title("📽️ Transcriptor de vídeos")
    st.write("Sube un vídeo y obtén la transcripción en texto plano.")

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox(
            "Modelo Whisper",
            ["base", "small", "medium"],
            index=0,
            help="Modelos mayores = +precisión y +tiempo/CPU.",
        )
    with col2:
        language_hint = st.selectbox(
            "Idioma (opcional)",
            ["auto", "es", "en", "pt", "fr", "de", "it", "gl", "ca"],
            index=0,
            help="Deja 'auto' para detección automática.",
        )

    video_file = st.file_uploader("Sube tu vídeo", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                video_path = temp_dir / "input_video"
                video_path.write_bytes(video_file.read())

                st.info("Extrayendo audio…")
                audio_path = temp_dir / "audio.wav"
                extract_audio(str(video_path), str(audio_path))

                with st.spinner("Transcribiendo…"):
                    text = transcribe_audio(str(audio_path), model_name, language_hint)

                st.success("¡Transcripción completada!")
                st.subheader("Transcripción")
                st.write(text if text else "_(vacía)_")

                st.download_button(
                    label="Descargar transcripción",
                    data=text,
                    file_name="transcripcion.txt",
                    mime="text/plain",
                )

        except Exception as e:
            st.error(f"Error durante el procesamiento: {e}")
            st.caption("Prueba con un vídeo más corto o distinto códec.")

if __name__ == "__main__":
    main()

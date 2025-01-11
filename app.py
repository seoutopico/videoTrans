import streamlit as st
import whisper
import tempfile
import os

def main():
    st.title("Transcriptor con Whisper (modelo tiny)")

    video_file = st.file_uploader("Sube tu archivo de vídeo", type=["mp4", "mov", "avi", "mkv"])
    if video_file is not None:
        st.write("Cargando modelo 'tiny' de Whisper...")
        model = whisper.load_model("tiny")

        st.write("Guardando vídeo en archivo temporal...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file_path = tmp_file.name

        st.write("Transcribiendo, espera un momento...")
        result = model.transcribe(tmp_file_path)
        os.remove(tmp_file_path)  # liberamos espacio

        st.write("Transcripción completa:")
        st.text(result["text"])

        segments = result["segments"]
        texto_timestamps = []
        for seg in segments:
            inicio = seg['start']
            fin = seg['end']
            texto_seg = seg['text']
            texto_timestamps.append(f"[{inicio:.2f} - {fin:.2f}] {texto_seg.strip()}")
        texto_final = "\n".join(texto_timestamps)

        st.write("Transcripción con marcas de tiempo:")
        st.text(texto_final)

        st.download_button(
            label="Descargar transcripción",
            data=texto_final,
            file_name="transcripcion.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
